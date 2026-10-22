"""Round-14 inference-only Task-Waypoint-to-motion mapping audit.

The audit separates deployable current-state mappings from offline labels and
counterfactual outcomes.  Frozen predictor inference is allowed; this module
creates no optimizer, loss, gradient, checkpoint, or trainable model.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from wt.maritime.baselines import create_baseline_policy
from wt.round12.contracts import (
    PREDICTOR_SHA256,
    assert_predictor_frozen,
    assert_selected_predictor,
    load_frozen_predictor,
    sha256,
)
from wt.round12.data import ROUND12_ACTION_SOURCE_DIR, load_round12_policy_dataset
from wt.round13.audit import (
    _allowed_difficulty_labels,
    _new_policy,
    _rollout_from_snapshot,
    _snapshot,
    _validation_scenarios,
)
from wt.round7.round7_runner import radial_projection

from .guidance import (
    EPSILON,
    MAPPING_NAMES,
    guidance_action,
    map_task_waypoint_to_motion,
    return_magnitude,
)


SCHEMA = "round14_task_to_motion_mapping_audit_v1"
PROTOCOL_PATH = Path("round_14/configs/task_to_motion_mapping_protocol.json")
ROUND7_OUTPUT = Path("round_07/outputs/round07_formal_20260809_1740")
ROUND12_OUTPUT = Path("round_12/outputs/round12_formal_20260810_161000")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not math.isfinite(float(value)):
            raise ValueError("Round-14 output may not contain NaN or Inf")
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Round-14 output may not contain NaN or Inf")
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_plain(row[key]), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                    else _plain(row.get(key, ""))
                    for key in fields
                }
            )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _protocol(root: Path) -> dict[str, Any]:
    protocol = _read_json(root / PROTOCOL_PATH)
    if protocol.get("schema_version") != "round14_task_to_motion_mapping_protocol_v1":
        raise RuntimeError("Round-14 protocol schema mismatch")
    if protocol.get("allowed_splits") != ["train", "validation"]:
        raise RuntimeError("Round-14 must use exactly train/validation splits")
    if int(protocol.get("task_horizon", -1)) != 10:
        raise RuntimeError("Round-14 task horizon must stay at H=10")
    if int(protocol.get("bootstrap", {}).get("replicates", 0)) < 1000:
        raise RuntimeError("Round-14 bootstrap repetitions are insufficient")
    return protocol


def preflight(root: Path) -> dict[str, Any]:
    """Verify immutable dependencies before reading authorized development data."""
    root = Path(root)
    current = (root / "CURRENT_ROUND.txt").read_text(encoding="utf-8")
    if not current.startswith("round_14\n") or "status=in_progress" not in current:
        raise RuntimeError("CURRENT_ROUND.txt does not authorize Round-14")
    if (root / "round_15").exists():
        raise RuntimeError("Round-14 may not run after creating Round-15")
    manifest = _read_json(root / "round_14/round_manifest.json")
    expected_false = (
        "training_performed",
        "task_to_motion_mapping_audit_performed",
        "neural_policy_training_performed",
        "service_action_extension_performed",
        "untouched_holdout_accessed",
        "hgnn_training_performed",
        "ppo_training_performed",
    )
    for key in expected_false:
        if manifest.get(key) is not False:
            raise RuntimeError(f"Round-14 initial flag must be false: {key}")
    lock13 = _read_json(root / "round_13/ROUND_LOCK.json")
    amendment = _read_json(root / "round_13/ROUND_LOCK_AMENDMENT.json")
    if lock13.get("status") != "locked" or amendment.get("status") != "locked" or not amendment.get("original_round13_lock_unchanged", False):
        raise RuntimeError("Round-13 lock or completion amendment is not accepted")
    lock12 = _read_json(root / "round_12/ROUND_LOCK.json")
    if lock12.get("status") != "locked" or not lock12.get("acceptance_complete"):
        raise RuntimeError("Round-12 lock is not accepted")
    predictor = assert_selected_predictor(root)
    frozen = _read_json(root / ROUND12_OUTPUT / "frozen_policy_checkpoint_manifest.json")
    if int(frozen.get("checkpoint_count", -1)) != 24:
        raise RuntimeError("Round-12 frozen policy checkpoint manifest is incomplete")
    for row in frozen.get("checkpoints", []):
        if sha256(root / ROUND12_OUTPUT / str(row["path"])) != str(row["sha256"]):
            raise RuntimeError("a locked Round-12 policy checkpoint changed")
    return {
        "round13_lock_sha256": sha256(root / "round_13/ROUND_LOCK.json"),
        "round13_amendment_sha256": sha256(root / "round_13/ROUND_LOCK_AMENDMENT.json"),
        "round12_lock_sha256": sha256(root / "round_12/ROUND_LOCK.json"),
        "predictor_path": str(predictor.relative_to(root)),
        "predictor_sha256": sha256(predictor),
        "policy_checkpoint_count": 24,
        "old_development_test_accessed": False,
        "untouched_holdout_accessed": False,
        "round15_created": False,
    }


def _rank(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks without depending on a statistical package."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float]) -> tuple[float, float]:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if x.size < 2 or np.std(x) <= EPSILON or np.std(y) <= EPSILON:
        return 0.0, 0.0
    return float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _cosine(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    if left_norm <= EPSILON or right_norm <= EPSILON:
        return 0.0, 90.0
    cosine = float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))
    return cosine, float(math.degrees(math.acos(cosine)))


def _load_quality_labels(root: Path, allowed_episode_ids: Iterable[str]) -> Mapping[str, str]:
    allowed = {str(item) for item in allowed_episode_ids}
    labels: dict[str, str] = {}
    with (root / ROUND7_OUTPUT / "trajectory_quality_audit.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            episode_id = str(row["episode_id"])
            if episode_id in allowed:
                labels[episode_id] = str(row["quality_tier"])
    if set(labels) != allowed:
        raise RuntimeError("authorized episode quality labels are incomplete")
    return labels


def _geometry_for_split(root: Path, values: Any) -> dict[str, np.ndarray]:
    """Read only current physical arrays for already-authorized transitions."""
    source = root / ROUND12_ACTION_SOURCE_DIR
    manifest = _read_json(source / "dataset_manifest.json")
    allowed_ids = {str(value) for value in values.episode_id.tolist()}
    entries = {str(row["episode_id"]): row for row in manifest.get("episodes", [])}
    max_buoys = int(values.max_buoys)
    count = len(values)
    uav = np.zeros((count, 2), dtype=np.float64)
    buoy = np.zeros((count, max_buoys, 2), dtype=np.float64)
    rates = np.zeros((count, max_buoys), dtype=np.float64)
    queues = np.zeros((count, max_buoys), dtype=np.float64)
    for episode_id in sorted(allowed_ids):
        entry = entries.get(episode_id)
        if entry is None or str(entry.get("split")) not in ("train", "validation"):
            raise PermissionError("Round-14 attempted to open an unregistered episode")
        selected = np.flatnonzero(values.episode_id == episode_id)
        with np.load(source / str(entry["npz_file"]), allow_pickle=False) as archive:
            for destination in selected:
                timestep = int(values.timestep[destination])
                node_count = int(entry["num_buoys"])
                uav[destination] = archive["uav_positions_m"][timestep]
                buoy[destination, :node_count] = archive["buoy_positions_m"][timestep]
                rates[destination, :node_count] = archive["rates_bps"][timestep]
                queues[destination, :node_count] = archive["raw_queue_bits"][timestep]
    return {"uav_position_m": uav, "buoy_positions_m": buoy, "rates_bps": rates, "raw_queue_bits": queues}


def _queue_levels(values: np.ndarray) -> np.ndarray:
    totals = np.asarray(values, dtype=float).sum(axis=1)
    low, high = np.quantile(totals, (1.0 / 3.0, 2.0 / 3.0))
    return np.asarray(["low" if item <= low else ("medium" if item <= high else "high") for item in totals])


def teacher_alignment_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Measure mapping/recorded-action alignment on permitted offline rows."""
    data = load_round12_policy_dataset(root)
    quality = _load_quality_labels(root, np.concatenate((data.train.episode_id, data.validation.episode_id)))
    validation_scenarios = _validation_scenarios(root)
    difficulty = _allowed_difficulty_labels(root, validation_scenarios)
    # Train scenarios are also registered train-only records; their difficulty
    # sidecar is read only by scenario id and never enters a mapping feature.
    allowed_scenarios = {str(item) for split in data.splits.values() for item in split.scenario_id.tolist()}
    with (root / ROUND7_OUTPUT / "scenario_difficulty_audit.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["scenario_id"]) in allowed_scenarios:
                difficulty[str(row["scenario_id"])] = str(row["difficulty_tier"])
    if set(difficulty) != allowed_scenarios:
        raise RuntimeError("authorized train/validation scenario difficulty labels are incomplete")

    rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    selection: dict[str, Any] = {"scores": {}}
    for split_name in ("train", "validation"):
        values = data.splits[split_name]
        geometry = _geometry_for_split(root, values)
        queue_level = _queue_levels(geometry["raw_queue_bits"])
        oracle_actions: dict[str, list[np.ndarray]] = {name: [] for name in MAPPING_NAMES}
        predicted_actions: dict[str, list[np.ndarray]] = {name: [] for name in MAPPING_NAMES}
        for index in range(len(values)):
            mask = values.node_masks[index].astype(bool)
            action = np.asarray(values.actions[index], dtype=float)
            for mapping_name in MAPPING_NAMES:
                predicted = map_task_waypoint_to_motion(
                    mapping_name,
                    values.predicted_phi[index],
                    mask,
                    geometry["uav_position_m"][index],
                    geometry["buoy_positions_m"][index],
                    geometry["rates_bps"][index],
                )
                oracle = map_task_waypoint_to_motion(
                    mapping_name,
                    values.oracle_phi[index],
                    mask,
                    geometry["uav_position_m"][index],
                    geometry["buoy_positions_m"][index],
                    geometry["rates_bps"][index],
                )
                guidance = guidance_action(predicted)
                cosine, angle = _cosine(guidance, action)
                future_cosine, future_angle = _cosine(guidance, guidance_action(oracle))
                rows.append(
                    {
                        "schema_version": SCHEMA,
                        "split": split_name,
                        "record_id": f"{values.episode_id[index]}__t-{int(values.timestep[index]):03d}",
                        "episode_id": str(values.episode_id[index]),
                        "scenario_id": str(values.scenario_id[index]),
                        "timestep": int(values.timestep[index]),
                        "mapping_name": mapping_name,
                        "num_buoys": int(values.num_buoys[index]),
                        "quality_tier": quality[str(values.episode_id[index])],
                        "difficulty_tier": difficulty[str(values.scenario_id[index])],
                        "queue_level": str(queue_level[index]),
                        "teacher_action_norm": float(np.linalg.norm(action)),
                        "teacher_action_nonzero": bool(np.linalg.norm(action) > EPSILON),
                        "guidance_action": guidance.tolist(),
                        "guidance_nonzero": bool(np.linalg.norm(guidance) > EPSILON),
                        "teacher_alignment_cosine": cosine,
                        "teacher_alignment_angle_deg": angle,
                        "future_oracle_diagnostic_cosine": future_cosine,
                        "future_oracle_diagnostic_angle_deg": future_angle,
                        "mapping_inputs_current_only": True,
                    }
                )
                predicted_actions[mapping_name].append(guidance)
                oracle_actions[mapping_name].append(guidance_action(oracle))
        for mapping_name in MAPPING_NAMES:
            predicted_array = np.asarray(predicted_actions[mapping_name], dtype=float)
            oracle_array = np.asarray(oracle_actions[mapping_name], dtype=float)
            future_rows.append(
                {
                    "schema_version": SCHEMA,
                    "split": split_name,
                    "mapping_name": mapping_name,
                    "diagnostic": "predicted_direction_vs_oracle_task_waypoint_direction",
                    "uses_future_outcome_labels": True,
                    "mapping_uses_future_inputs": False,
                    "records": int(len(values)),
                    "mean_direction_cosine": float(np.mean([_cosine(a, b)[0] for a, b in zip(predicted_array, oracle_array)])),
                    "mean_direction_angle_deg": float(np.mean([_cosine(a, b)[1] for a, b in zip(predicted_array, oracle_array)])),
                }
            )
        predicted_return = values.predicted_phi[:, -1]
        oracle_return = values.oracle_phi[:, -1]
        pearson, spearman = _correlation(predicted_return, oracle_return)
        future_rows.append(
            {
                "schema_version": SCHEMA,
                "split": split_name,
                "mapping_name": "return_head",
                "diagnostic": "predicted_standardized_H_step_return_vs_oracle_standardized_H_step_return",
                "uses_future_outcome_labels": True,
                "mapping_uses_future_inputs": False,
                "records": int(len(values)),
                "pearson_r": pearson,
                "spearman_r": spearman,
            }
        )
    summary_rows = _alignment_summary(rows)
    validation_scores = [row for row in summary_rows if row["split"] == "validation" and row["group_dimension"] == "overall"]
    selection["scores"] = {row["mapping_name"]: row["mean_teacher_alignment_cosine"] for row in validation_scores}
    selection["selected_direction_mapping"] = max(
        MAPPING_NAMES, key=lambda name: (float(selection["scores"].get(name, -np.inf)), name)
    )
    return rows, summary_rows, future_rows, selection


def _alignment_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    dimensions = {
        "overall": lambda row: "all",
        "quality_tier": lambda row: str(row["quality_tier"]),
        "difficulty_tier": lambda row: str(row["difficulty_tier"]),
        "num_buoys": lambda row: str(row["num_buoys"]),
        "queue_level": lambda row: str(row["queue_level"]),
    }
    for split in ("train", "validation"):
        for mapping_name in MAPPING_NAMES:
            active = [row for row in rows if row["split"] == split and row["mapping_name"] == mapping_name]
            for dimension, labeler in dimensions.items():
                for label in sorted({labeler(row) for row in active}):
                    subset = [row for row in active if labeler(row) == label]
                    output.append(
                        {
                            "schema_version": SCHEMA,
                            "split": split,
                            "mapping_name": mapping_name,
                            "group_dimension": dimension,
                            "group_value": label,
                            "records": len(subset),
                            "teacher_nonzero_records": int(sum(bool(row["teacher_action_nonzero"]) for row in subset)),
                            "guidance_nonzero_rate": float(np.mean([bool(row["guidance_nonzero"]) for row in subset])),
                            "mean_teacher_alignment_cosine": float(np.mean([float(row["teacher_alignment_cosine"]) for row in subset])),
                            "mean_teacher_alignment_angle_deg": float(np.mean([float(row["teacher_alignment_angle_deg"]) for row in subset])),
                            "mean_future_oracle_diagnostic_cosine": float(np.mean([float(row["future_oracle_diagnostic_cosine"]) for row in subset])),
                        }
                    )
    return output


def _padded_current_info(info: Mapping[str, Any], max_buoys: int) -> dict[str, np.ndarray]:
    actual = int(np.asarray(info["valid_node_mask"], dtype=bool).size)
    if actual > max_buoys:
        raise ValueError("environment exposes more buoys than the frozen mapping width")
    mask = np.zeros(max_buoys, dtype=bool)
    buoy = np.zeros((max_buoys, 2), dtype=float)
    rates = np.zeros(max_buoys, dtype=float)
    mask[:actual] = np.asarray(info["valid_node_mask"], dtype=bool)
    buoy[:actual] = np.asarray(info["buoy_positions_m"], dtype=float)
    rates[:actual] = np.asarray(info["rates_bps"], dtype=float)
    return {
        "valid_node_mask": mask,
        "uav_position_m": np.asarray(info["uav_position_m"], dtype=float),
        "buoy_positions_m": buoy,
        "rates_bps": rates,
    }


def _predict_current_phi(model: Any, observation: np.ndarray, mask: np.ndarray, state_dim: int) -> np.ndarray:
    state = np.zeros((1, state_dim), dtype=np.float32)
    raw = np.asarray(observation, dtype=np.float32)
    if raw.size > state_dim:
        raise ValueError("environment observation exceeds frozen predictor state width")
    state[0, : raw.size] = raw
    with torch.no_grad():
        output = model(torch.from_numpy(state), torch.from_numpy(mask[None, :]))
        phi = torch.cat(
            (output["service_distribution"], output["gross_service_progress"], output["H_step_return"].unsqueeze(-1)),
            dim=-1,
        )
    if phi.requires_grad:
        raise RuntimeError("Round-14 predictor output unexpectedly requires gradients")
    return phi.cpu().numpy()[0].astype(np.float64)


def _snapshot_metadata(scenario: Mapping[str, Any], timestep: int, difficulty: str) -> dict[str, Any]:
    return {
        "scenario_id": str(scenario["scenario_id"]),
        "scenario_seed": int(scenario["scenario_seed"]),
        "num_buoys": int(scenario["num_buoys"]),
        "snapshot_id": f"{scenario['scenario_id']}__t-{int(timestep):03d}",
        "timestep": int(timestep),
        "difficulty_tier": str(difficulty),
    }


def _counterfactual_rows(
    root: Path,
    protocol: Mapping[str, Any],
    selected_mapping: str,
    train_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run paired, same-snapshot first-movement counterfactuals."""
    model = load_frozen_predictor(root, state_dim=44, max_buoys=5)
    assert_predictor_frozen(model)
    scenarios = _validation_scenarios(root)
    difficulty = _allowed_difficulty_labels(root, scenarios)
    counter_rows: list[dict[str, Any]] = []
    magnitude_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for timestep in protocol["counterfactual"]["snapshot_timesteps"]:
            horizon = int(protocol["task_horizon"])
            env, observation, info = _snapshot(scenario, int(timestep), horizon)
            current = _padded_current_info(info, 5)
            phi = _predict_current_phi(model, observation, current["valid_node_mask"], 44)
            metadata = _snapshot_metadata(scenario, int(timestep), difficulty[str(scenario["scenario_id"])])
            qre_action = radial_projection(_new_policy().act(observation, info))
            qre = _rollout_from_snapshot(env, observation, info, qre_action, horizon)
            zero = _rollout_from_snapshot(env, observation, info, np.zeros(2, dtype=float), horizon)
            for mapping_name in MAPPING_NAMES:
                mapped = map_task_waypoint_to_motion(mapping_name, phi, **{
                    "valid_node_mask": current["valid_node_mask"],
                    "uav_position_m": current["uav_position_m"],
                    "buoy_positions_m": current["buoy_positions_m"],
                    "current_rates_bps": current["rates_bps"],
                })
                action = radial_projection(guidance_action(mapped))
                opposite_action = radial_projection(-action)
                variants = {
                    "guidance_full_step": (action, _rollout_from_snapshot(env, observation, info, action, horizon)),
                    "opposite_guidance": (opposite_action, _rollout_from_snapshot(env, observation, info, opposite_action, horizon)),
                    "queue_rate_energy": (qre_action, qre),
                    "zero": (np.zeros(2, dtype=float), zero),
                }
                for action_name, (raw_action, metrics) in variants.items():
                    counter_rows.append(
                        _counterfactual_row(metadata, mapping_name, "full_step", action_name, raw_action, metrics, phi, mapped.node_weights)
                    )
                role_rows.append(
                    {
                        **metadata,
                        "mapping_name": mapping_name,
                        "head_role": "service_progress_direction",
                        "guidance_action": action.tolist(),
                        "node_weights": mapped.node_weights.tolist(),
                        "payload_vs_qre_bits": float(variants["guidance_full_step"][1]["total_service_bits"] - qre["total_service_bits"]),
                        "return_vs_qre": float(variants["guidance_full_step"][1]["H_step_return"] - qre["H_step_return"]),
                        "current_only": True,
                    }
                )
            selected = map_task_waypoint_to_motion(selected_mapping, phi, **{
                "valid_node_mask": current["valid_node_mask"],
                "uav_position_m": current["uav_position_m"],
                "buoy_positions_m": current["buoy_positions_m"],
                "current_rates_bps": current["rates_bps"],
            })
            magnitude_values = {
                "full_step": 1.0,
                "train_fixed_fraction": train_fraction,
                "return_modulated": return_magnitude(phi, 5),
            }
            for mode, magnitude in magnitude_values.items():
                action = radial_projection(guidance_action(selected, magnitude=magnitude))
                metrics = _rollout_from_snapshot(env, observation, info, action, horizon)
                magnitude_rows.append(
                    _counterfactual_row(metadata, selected_mapping, mode, "guidance", action, metrics, phi, selected.node_weights)
                )
                role_rows.append(
                    {
                        **metadata,
                        "mapping_name": selected_mapping,
                        "head_role": "return_magnitude",
                        "magnitude_mode": mode,
                        "magnitude": float(magnitude),
                        "guidance_action": action.tolist(),
                        "payload_vs_full_step_bits": float(metrics["total_service_bits"] - _rollout_from_snapshot(env, observation, info, radial_projection(guidance_action(selected)), horizon)["total_service_bits"]),
                        "return_vs_full_step": float(metrics["H_step_return"] - _rollout_from_snapshot(env, observation, info, radial_projection(guidance_action(selected)), horizon)["H_step_return"]),
                        "current_only": True,
                    }
                )
    assert_predictor_frozen(model)
    if sha256(assert_selected_predictor(root)) != PREDICTOR_SHA256:
        raise RuntimeError("frozen predictor changed during Round-14 inference")
    return counter_rows, magnitude_rows, role_rows


def _counterfactual_row(metadata: Mapping[str, Any], mapping_name: str, magnitude_mode: str, action_name: str, action: np.ndarray, metrics: Mapping[str, Any], phi: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    return {
        **metadata,
        "schema_version": SCHEMA,
        "mapping_name": mapping_name,
        "magnitude_mode": magnitude_mode,
        "action_name": action_name,
        "raw_action": np.asarray(action, dtype=float).tolist(),
        "action_norm": float(np.linalg.norm(action)),
        "phi_task": phi.tolist(),
        "node_weights": weights.tolist(),
        "total_service_bits": float(metrics["total_service_bits"]),
        "H_step_return": float(metrics["H_step_return"]),
        "final_completion": float(metrics["final_completion"]),
        "final_residual_queue_bits": float(metrics["final_residual_queue_bits"]),
        "horizon_steps": int(metrics["horizon_steps"]),
        "all_actions_legal": bool(metrics["all_actions_legal"]),
        "mapping_inputs_current_only": True,
    }


def _bootstrap(values: Mapping[str, Sequence[float]], *, seed: int, repetitions: int) -> dict[str, dict[str, float]]:
    ids = sorted(values)
    if not ids:
        raise ValueError("bootstrap requires at least one scenario")
    scenario_means = np.asarray([np.mean(values[key]) for key in ids], dtype=float)
    generator = np.random.default_rng(seed)
    samples = scenario_means[generator.integers(0, scenario_means.size, size=(repetitions, scenario_means.size))].mean(axis=1)
    return {
        "value": {
            "mean": float(scenario_means.mean()),
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
            "scenario_count": int(scenario_means.size),
        }
    }


def paired_comparisons(counter_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], selected_mapping: str) -> list[dict[str, Any]]:
    """Aggregate first-action comparisons while retaining snapshot pairing by scenario."""
    lookup = {(row["snapshot_id"], row["mapping_name"], row["magnitude_mode"], row["action_name"]): row for row in counter_rows}
    dimensions = {
        "overall": lambda row: "all",
        "difficulty_tier": lambda row: str(row["difficulty_tier"]),
        "num_buoys": lambda row: str(row["num_buoys"]),
    }
    comparisons: list[tuple[str, str, str, str]] = []
    for mapping_name in MAPPING_NAMES:
        comparisons.append((mapping_name, "guidance_full_step", "queue_rate_energy", "guidance_minus_queue_rate_energy"))
        comparisons.append((mapping_name, "guidance_full_step", "opposite_guidance", "guidance_minus_opposite"))
        comparisons.append((mapping_name, "guidance_full_step", "zero", "guidance_minus_zero"))
    metrics = {
        "Delta_payload_bits": ("total_service_bits", 1.0),
        "Delta_completion": ("final_completion", 1.0),
        "Delta_residual_queue_bits": ("final_residual_queue_bits", -1.0),
        "Delta_return": ("H_step_return", 1.0),
    }
    output: list[dict[str, Any]] = []
    seed_base = int(protocol["bootstrap"]["seed"])
    for mapping_name, left_name, right_name, label in comparisons:
        left_rows = [row for row in counter_rows if row["mapping_name"] == mapping_name and row["magnitude_mode"] == "full_step" and row["action_name"] == left_name]
        for dimension, labeler in dimensions.items():
            for group in sorted({labeler(row) for row in left_rows}):
                selected_rows = [row for row in left_rows if labeler(row) == group]
                for metric, (field, sign) in metrics.items():
                    grouped: dict[str, list[float]] = {}
                    for left in selected_rows:
                        right = lookup[(left["snapshot_id"], mapping_name, "full_step", right_name)]
                        grouped.setdefault(str(left["scenario_id"]), []).append(sign * (float(left[field]) - float(right[field])))
                    result = _bootstrap(grouped, seed=seed_base + len(output), repetitions=int(protocol["bootstrap"]["replicates"]))["value"]
                    output.append(
                        {
                            "schema_version": SCHEMA,
                            "mapping_name": mapping_name,
                            "comparison": label,
                            "metric": metric,
                            "group_dimension": dimension,
                            "group_value": group,
                            "paired_snapshot_count": len(selected_rows),
                            "bootstrap_unit": "scenario_id; paired states retained within each scenario",
                            "paired_mean": result["mean"],
                            "bootstrap_ci_low": result["ci_low"],
                            "bootstrap_ci_high": result["ci_high"],
                            "scenario_count": result["scenario_count"],
                        }
                    )
    return output


def magnitude_comparisons(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    lookup = {(row["snapshot_id"], row["magnitude_mode"]): row for row in rows}
    metrics = {
        "Delta_payload_bits": ("total_service_bits", 1.0),
        "Delta_completion": ("final_completion", 1.0),
        "Delta_residual_queue_bits": ("final_residual_queue_bits", -1.0),
        "Delta_return": ("H_step_return", 1.0),
    }
    output: list[dict[str, Any]] = []
    for mode in ("train_fixed_fraction", "return_modulated"):
        for metric, (field, sign) in metrics.items():
            grouped: dict[str, list[float]] = {}
            left_rows = [row for row in rows if row["magnitude_mode"] == mode]
            for left in left_rows:
                full = lookup[(left["snapshot_id"], "full_step")]
                grouped.setdefault(str(left["scenario_id"]), []).append(sign * (float(left[field]) - float(full[field])))
            result = _bootstrap(grouped, seed=int(protocol["bootstrap"]["seed"]) + len(output), repetitions=int(protocol["bootstrap"]["replicates"]))["value"]
            output.append(
                {
                    "schema_version": SCHEMA,
                    "comparison": f"{mode}_minus_full_step",
                    "metric": metric,
                    "paired_mean": result["mean"],
                    "bootstrap_ci_low": result["ci_low"],
                    "bootstrap_ci_high": result["ci_high"],
                    "scenario_count": result["scenario_count"],
                    "paired_snapshot_count": len(left_rows),
                    "bootstrap_unit": "scenario_id; paired states retained within each scenario",
                }
            )
    return output


def mapping_gate(paired: Sequence[Mapping[str, Any]], selected_mapping: str) -> dict[str, Any]:
    rows = [
        row for row in paired
        if row["mapping_name"] == selected_mapping
        and row["comparison"] == "guidance_minus_queue_rate_energy"
        and row["group_dimension"] == "overall"
    ]
    def normalized(row: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(row)
        for field in ("paired_mean", "bootstrap_ci_low", "bootstrap_ci_high"):
            value[field] = float(value[field])
        for field in ("paired_snapshot_count", "scenario_count"):
            value[field] = int(value[field])
        return value

    by_metric = {str(row["metric"]): normalized(row) for row in rows}
    payload = by_metric["Delta_payload_bits"]
    completion = by_metric["Delta_completion"]
    residual = by_metric["Delta_residual_queue_bits"]
    ret = by_metric["Delta_return"]
    nondegrading = all(float(row["bootstrap_ci_low"]) >= -EPSILON for row in (completion, residual, ret))
    stable_rows = [
        row for row in paired
        if row["mapping_name"] == selected_mapping
        and row["comparison"] == "guidance_minus_queue_rate_energy"
        and row["metric"] == "Delta_payload_bits"
        and row["group_dimension"] in ("difficulty_tier", "num_buoys")
    ]
    group_stable = bool(stable_rows) and all(float(row["paired_mean"]) >= -EPSILON for row in stable_rows)
    stable_worse = any(float(row["bootstrap_ci_high"]) < -EPSILON for row in (payload, completion, residual, ret))
    if float(payload["paired_mean"]) > 0.0 and float(payload["bootstrap_ci_low"]) > 0.0 and nondegrading and group_stable:
        gate = "MAP-A"
        interpretation = "positive payload with a strictly positive paired CI, nondegrading secondary outcomes, and stable registered groups"
    elif stable_worse:
        gate = "MAP-C"
        interpretation = "stable payload harm or a secondary-outcome safety regression"
    else:
        gate = "MAP-B"
        interpretation = "diagnostic signal is uncertain; payload CI crosses zero or registered groups are unstable"
    return {
        "schema_version": SCHEMA,
        "task_to_motion_mapping_gate": gate,
        "selected_direction_mapping": selected_mapping,
        "primary_comparison": "guidance_full_step minus queue_rate_energy from identical validation snapshots",
        "payload": payload,
        "completion": completion,
        "residual_queue": residual,
        "return": ret,
        "secondary_outcomes_nondegrading": nondegrading,
        "group_stability": group_stable,
        "stable_worse": stable_worse,
        "interpretation": interpretation,
        "not_a_service_action_extension": True,
        "old_development_test_accessed": False,
        "untouched_holdout_accessed": False,
    }


def _train_fixed_fraction(root: Path) -> float:
    data = load_round12_policy_dataset(root)
    fractions = np.linalg.norm(data.train.actions.astype(float), axis=1)
    return float(np.clip(np.median(fractions), 0.0, 1.0))


def _source_tree(root: Path) -> Mapping[str, str]:
    source = root / "src/wt/round14"
    return {path.relative_to(source).as_posix(): sha256(path) for path in sorted(source.rglob("*.py"))}


def run_mapping_audit(root: Path, output: Path) -> dict[str, Any]:
    """Materialize all Round-14 mapping evidence inside one new output folder."""
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(f"Round-14 output already exists: {output}")
    protocol = _protocol(root)
    preflight_audit = preflight(root)
    output.mkdir(parents=True, exist_ok=False)
    teacher_rows, alignment_summary, future_rows, selection = teacher_alignment_rows(root)
    selected_mapping = str(selection["selected_direction_mapping"])
    fraction = _train_fixed_fraction(root)
    counter_rows, raw_magnitude_rows, role_rows = _counterfactual_rows(root, protocol, selected_mapping, fraction)
    paired = paired_comparisons(counter_rows, protocol, selected_mapping)
    magnitude = magnitude_comparisons(raw_magnitude_rows, protocol)
    gate = mapping_gate(paired, selected_mapping)
    selected = {
        "schema_version": SCHEMA,
        "selected_direction_mapping": selected_mapping,
        "selection_basis": "highest validation recorded-action cosine; no outcome-based coefficient search",
        "validation_teacher_cosines": selection["scores"],
        "full_step_fraction": 1.0,
        "train_fixed_fraction": fraction,
        "return_magnitude": "sigmoid(predicted standardized H_step_return), current prediction only",
        "fallback": "zero movement when weighted direction cancels or all valid geometry vectors are zero",
        "public_action_interface": "unchanged normalized [delta_x, delta_y]",
        "mapping_gate": gate["task_to_motion_mapping_gate"],
        "deployment_ready": gate["task_to_motion_mapping_gate"] == "MAP-A",
    }
    _write_csv(output / "mapping_teacher_alignment.csv", teacher_rows)
    _write_csv(output / "mapping_quality_stratified_alignment.csv", alignment_summary)
    _write_csv(output / "mapping_future_relevance.csv", future_rows)
    _write_csv(output / "mapping_counterfactual_metrics.csv", counter_rows)
    _write_csv(output / "mapping_paired_comparison.csv", paired)
    _write_csv(output / "return_magnitude_modulation_audit.csv", magnitude)
    _write_csv(output / "task_waypoint_head_role_audit.csv", role_rows)
    _write_json(output / "task_to_motion_mapping_gate.json", gate)
    _write_json(output / "selected_task_to_motion_mapping.json", selected)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL_PATH),
        "preflight": preflight_audit,
        "source_tree": _source_tree(root),
        "source_hashes": {
            "round10_task_waypoint_metadata": sha256(root / "round_10/outputs/round10_task_waypoint_20260810_0110/task_waypoint_metadata.jsonl"),
            "round10_task_waypoint_arrays": sha256(root / "round_10/outputs/round10_task_waypoint_20260810_0110/task_waypoint_samples_v1.npz"),
            "round7_development_manifest": sha256(root / ROUND12_ACTION_SOURCE_DIR / "dataset_manifest.json"),
        },
        "selected_mapping": selected,
        "counts": {
            "teacher_alignment_rows": len(teacher_rows),
            "counterfactual_rows": len(counter_rows),
            "paired_comparison_rows": len(paired),
            "magnitude_rows": len(magnitude),
            "counterfactual_snapshot_count": len({row["snapshot_id"] for row in counter_rows}),
        },
        "training_performed": False,
        "neural_policy_training_performed": False,
        "service_action_extension_performed": False,
        "old_development_test_accessed": False,
        "untouched_holdout_accessed": False,
    }
    _write_json(output / "round14_mapping_manifest.json", manifest)
    _write_json(output / "run_manifest.json", {**manifest, "output": str(output.relative_to(root))})
    return {
        "output": output,
        "gate": gate,
        "selected": selected,
        "manifest": manifest,
        "preflight": preflight_audit,
    }


__all__ = [
    "SCHEMA",
    "_alignment_summary",
    "_bootstrap",
    "_counterfactual_row",
    "_geometry_for_split",
    "_protocol",
    "_train_fixed_fraction",
    "mapping_gate",
    "magnitude_comparisons",
    "paired_comparisons",
    "preflight",
    "run_mapping_audit",
    "teacher_alignment_rows",
]
