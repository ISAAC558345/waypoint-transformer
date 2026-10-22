"""Frozen Round-28 current-state planning-need representation audit.

This module trains nothing.  It reads only pre-existing allowed-split Joint-H1
trajectory states, restores those physical snapshots, and applies the unchanged
Round-27 same-snapshot H5-versus-H1 utility protocol.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from wt.governance_state_sync import approved_lifecycle_overlay_sha256, verify_round22_governance_state_sync_amendment
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv, assignment_link_metrics, gain_matrix, joint_aware_matching
from wt.round19.planner import joint_plan
from wt.round20.scenarios import EPISODE_HORIZON, development_scenarios
from wt.round22.data import FrozenDeepSetsPredictor, VALIDATION, load_policy_sequences
from wt.round22.models import PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER
from wt.round22.training import TrainingConfig, load_checkpoint, predict
from wt.round27.audit import verify_frozen_state as verify_round27_frozen_state


ROOT = Path(__file__).resolve().parents[3]
ROUND = ROOT / "round_28"
OUTPUT = ROUND / "outputs" / "round28_planning_need_representation_20260830_1624"
FIGURES = ROUND / "figures" / "round28_planning_need_representation_20260830_1624"
CONFIG_PATH = ROUND / "configs" / "round28_planning_need_representation_config.json"
R20 = ROOT / "round_20" / "outputs" / "round20_multi_uav_dataset_20260828_2100"
R22 = ROOT / "round_22" / "outputs" / "round22_deepsets_task_waypoint_policy_20260829_1153"
R24 = ROOT / "round_24" / "outputs" / "round24_multiseed_stability_20260829_1515"
EXPECTED_R27 = "84B1F2993F2DA19C88874C92F33C287E174C3CD9830A7E43E78C73818B8A7267"
ALLOWED = frozenset(("train", "validation"))

BASE_FEATURES = (
    "preference_cosine_similarity", "top_choice_overlap_strength", "top1_top2_score_margin",
    "average_preference_entropy", "assignment_regret", "mean_isr", "max_isr", "isr_asymmetry",
    "shared_sum_rate_loss_fraction", "total_normalized_queue", "top1_queue_share", "top2_queue_share",
    "queue_entropy", "queue_imbalance", "unserved_high_queue_mass", "queue_to_current_capacity_ratio",
    "buoy_spatial_spread_m", "high_queue_buoy_spread_m", "distance_to_preferred_buoy_m",
    "distance_to_best_unassigned_high_value_buoy_m", "assigned_minus_alternative_distance_m",
    "uav_uav_distance_m", "safety_margin_m", "distinct_preferred_buoy_count",
    "current_assignment_diversity", "assignment_score_asymmetry", "assignment_score_difference_normalized",
    "current_distinct_served_buoy_count", "queue_to_h5_current_capacity_ratio",
    "unassigned_queue_to_h5_current_capacity_ratio", "top_unserved_opportunity_ratio",
)
COMPOSITES = (
    "coordination_pressure", "interference_queue_pressure", "deferred_service_pressure", "unified_planning_pressure",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise RuntimeError(f"unexpected empty Round-28 output: {path.name}")
    fields = sorted({key for row in values for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in values:
            writer.writerow({key: json.dumps(_plain(value), sort_keys=True) if isinstance(value, (dict, list, tuple, np.ndarray)) else value for key, value in row.items()})


def _feature_rows(path: Path) -> list[dict[str, Any]]:
    """Read a previously completed active-round feature table without new replay."""
    text_fields = {"state_id", "scenario_id", "split", "utility_class", "continuation_policy", "current_assignment", "current_preferred_buoys"}
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in text_fields:
                    row[key] = value
                elif value in {"True", "False"}:
                    row[key] = value == "True"
                else:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        row[key] = value
            rows.append(row)
    return rows


def verify_frozen_state() -> dict[str, Any]:
    """Fail closed before decoding any protected Round-20 row."""
    if _digest(ROOT / "round_27" / "ROUND_LOCK.json") != EXPECTED_R27:
        raise RuntimeError("Round-27 original lock changed")
    amendment = _load(ROOT / "round_27" / "ROUND_LOCK_AMENDMENT.json")
    expected = {
        "src/wt/governance_state_sync.py": ROOT / "src" / "wt" / "governance_state_sync.py",
        "tests/test_round24_seed_extension.py": ROOT / "tests" / "test_round24_seed_extension.py",
        "tests/test_round21_hgnn_task_waypoint.py": ROOT / "tests" / "test_round21_hgnn_task_waypoint.py",
        "tests/test_governance_state_sync.py": ROOT / "tests" / "test_governance_state_sync.py",
    }
    if amendment.get("amendment_type") != "legal_round28_successor_governance_extension" or amendment.get("original_round27_lock_sha256") != EXPECTED_R27:
        raise RuntimeError("Round-27 legal-successor overlay is invalid")
    affected = amendment.get("affected_paths", {})
    if any(_digest(path) != approved_lifecycle_overlay_sha256(name, affected.get(name, {}).get("amended_sha256", "")) for name, path in expected.items()):
        raise RuntimeError("Round-27 legal-successor overlay hashes are invalid")
    prohibited = ("scientific_artifacts_modified", "model_checkpoints_modified", "dataset_modified", "metrics_modified", "fresh_dev_rerun", "untouched_eval_accessed", "original_round27_lock_modified")
    if any(amendment.get(name) is not False for name in prohibited):
        raise RuntimeError("Round-27 amendment modifies protected evidence")
    prior = verify_round27_frozen_state()
    governance = verify_round22_governance_state_sync_amendment()
    return {
        "passed": True, "round27_original_lock_sha256": EXPECTED_R27,
        "round27_frozen_chain": prior, "round22_governance_verified": bool(governance["passed"]),
        "fresh_new_rollout_count": 0, "round20_development_test_reuse_count": 0,
        "untouched_eval_access_count": 0,
    }


def _scenario_map() -> dict[str, Any]:
    return {item.scenario_id: item for item in development_scenarios() if item.split in ALLOWED}


def _allowed_h1_states(config: Mapping[str, Any], scenarios: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Filter raw lines before JSON decoding, excluding every protected split."""
    wanted = {int(value) for value in config["state_sampling"]["timesteps"]}
    selected: list[dict[str, Any]] = []
    with (R20 / "multi_uav_transitions.jsonl").open(encoding="utf-8") as handle:
        for raw in handle:
            if '"behavior_generator":"joint_h1"' not in raw:
                continue
            if '"split":"train"' not in raw and '"split":"validation"' not in raw:
                continue
            row = json.loads(raw)
            if row.get("split") not in ALLOWED or row.get("behavior_generator") != "joint_h1":
                raise RuntimeError("Round-28 decoded a prohibited source row")
            if int(row["timestep"]) in wanted:
                selected.append(row)
    expected = {(scenario_id, timestep) for scenario_id in scenarios for timestep in wanted}
    actual = {(str(row["scenario_id"]), int(row["timestep"])) for row in selected}
    if actual != expected or len(selected) != len(actual):
        raise RuntimeError("stored Joint-H1 state corpus does not match the pre-registered deterministic sample")
    return sorted(selected, key=lambda row: (str(row["split"]), str(row["scenario_id"]), int(row["timestep"])))


def _state_signature(info: Mapping[str, Any], scenario: Any) -> tuple[bytes, bytes, bytes, bytes, int, float]:
    return (
        np.asarray(info["uav_positions_m"], dtype=np.float64).tobytes(),
        np.asarray(info["raw_queue_bits"], dtype=np.float64).tobytes(),
        np.asarray(info["remaining_energy_per_uav"], dtype=np.float64).tobytes(),
        np.asarray(info["buoy_positions_m"], dtype=np.float64).tobytes(),
        int(info["time_step"]), float(scenario.interference_overlap_factor),
    )


def _restore_env(scenario: Any, graph: Mapping[str, Any], timestep: int) -> tuple[SharedSpectrumMultiUAVEnv, dict[str, Any]]:
    """Restore only current physical state from a stored H1 trajectory snapshot."""
    env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options())
    env.reset(seed=scenario.seed)
    base = env._base
    uav = np.asarray(graph["uav_nodes"], dtype=np.float64)[:env.num_uavs]
    buoy = np.asarray(graph["buoy_nodes"], dtype=np.float64)[:env.num_buoys]
    base._uav_positions_m = uav[:, :2].copy()
    base._uav_velocities_mps = uav[:, 2:4].copy()
    base._remaining_energy = uav[:, 4].copy()
    base._queue_bits = buoy[:, 2].copy()
    base._time_step = int(timestep)
    base._has_reset, base._episode_done = True, False
    base._last_assignment = np.asarray(graph["uav_to_buoy_assignment"], dtype=np.int64)[:env.num_uavs].copy()
    info = {
        "uav_positions_m": base._uav_positions_m.copy(), "raw_queue_bits": base._queue_bits.copy(),
        "remaining_energy_per_uav": base._remaining_energy.copy(),
        "buoy_positions_m": np.asarray(scenario.buoy_positions_m, dtype=np.float64).copy(),
        "time_step": int(timestep),
    }
    if not (np.all(np.isfinite(base._uav_positions_m)) and np.all(np.isfinite(base._queue_bits)) and np.all(np.isfinite(base._remaining_energy))):
        raise RuntimeError("stored H1 physical snapshot is non-finite")
    return env, info


def _direct_rates(positions: np.ndarray, buoys: np.ndarray, env: SharedSpectrumMultiUAVEnv) -> tuple[np.ndarray, np.ndarray]:
    gains, _ = gain_matrix(positions, buoys, env.config)
    direct = float(env.config.total_bandwidth_hz) * np.log2(1.0 + float(env.config.buoy_tx_power_w) * gains / float(env.config.noise_power_w))
    return gains, direct


def _entropy(values: np.ndarray) -> float:
    mass = np.asarray(values, dtype=np.float64)
    total = float(np.sum(mass))
    if total <= 1.0e-12 or mass.size <= 1:
        return 0.0
    prob = mass / total
    return float(-np.sum(np.where(prob > 0.0, prob * np.log(prob), 0.0)) / math.log(mass.size))


def _weighted_spread(positions: np.ndarray, weights: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    distances, factors = [], []
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            distances.append(float(np.linalg.norm(positions[left] - positions[right])))
            factors.append(float(weights[left] * weights[right]))
    return float(np.average(distances, weights=factors)) if float(np.sum(factors)) > 1.0e-12 else float(np.mean(distances))


def _exclusive_score(weighted: np.ndarray) -> tuple[float, np.ndarray]:
    m_count, k_count = weighted.shape
    best_score, best = -math.inf, None
    for candidate in itertools.permutations(range(k_count), m_count):
        score = float(sum(weighted[uav, buoy] for uav, buoy in enumerate(candidate)))
        if score > best_score + 1.0e-12 or (math.isclose(score, best_score, abs_tol=1.0e-12) and (best is None or candidate < best)):
            best_score, best = score, candidate
    if best is None:
        raise RuntimeError("exclusive current assignment failed")
    return best_score, np.asarray(best, dtype=np.int64)


def _feature_row(scenario: Any, graph: Mapping[str, Any], timestep: int) -> dict[str, Any]:
    env, info = _restore_env(scenario, graph, timestep)
    positions = np.asarray(info["uav_positions_m"], dtype=np.float64)
    queues = np.asarray(info["raw_queue_bits"], dtype=np.float64)
    buoys = np.asarray(info["buoy_positions_m"], dtype=np.float64)
    gains, direct = _direct_rates(positions, buoys, env)
    normalized_queue = np.clip(queues / float(env.config.queue_capacity_bits), 0.0, 1.0)
    weighted = normalized_queue[None, :] * direct
    top_indices = np.argmax(weighted, axis=1)
    descending = np.sort(weighted, axis=1)[:, ::-1]
    top, runner = descending[:, 0], descending[:, 1]
    norms = np.linalg.norm(weighted, axis=1)
    cosine = float(np.dot(weighted[0], weighted[1]) / max(float(norms[0] * norms[1]), 1.0e-12))
    top_overlap = float(min(top) / max(float(np.max(top)), 1.0e-12))
    margin = float(np.mean((top - runner) / np.maximum(top, 1.0e-12)))
    preference_entropy = float(np.mean([_entropy(row) for row in weighted]))
    independent = float(np.sum(np.max(weighted, axis=1)))
    exclusive, _ = _exclusive_score(weighted)
    regret = float(max(independent - exclusive, 0.0) / max(abs(independent), 1.0e-12))
    assignment, _, assignment_scores, _ = joint_aware_matching(queues, gains, env.config, float(scenario.interference_overlap_factor))
    _, _, _, isr, selected_rates = assignment_link_metrics(assignment, gains, env.config, float(scenario.interference_overlap_factor))
    _, _, _, _, zero_isr_rates = assignment_link_metrics(assignment, gains, env.config, 0.0)
    shared_rate = float(np.sum(selected_rates))
    zero_rate = float(np.sum(zero_isr_rates))
    unserved = np.ones(env.num_buoys, dtype=bool)
    unserved[assignment[assignment >= 0]] = False
    capacity = float(np.sum(selected_rates) * env.config.delta_t)
    h5_capacity = 5.0 * capacity
    pair_distance = float(np.linalg.norm(positions[0] - positions[1]))
    alternatives, preferred_distances, assigned_minus_alt = [], [], []
    assigned_set = {int(value) for value in assignment if value >= 0}
    for uav in range(env.num_uavs):
        preferred_distances.append(float(np.linalg.norm(positions[uav] - buoys[top_indices[uav]])))
        choices = [buoy for buoy in range(env.num_buoys) if buoy not in assigned_set or buoy == int(assignment[uav])]
        alternatives_for_uav = [buoy for buoy in choices if buoy != int(assignment[uav])]
        if alternatives_for_uav:
            alternative = max(alternatives_for_uav, key=lambda buoy: (weighted[uav, buoy], -buoy))
            alt_distance = float(np.linalg.norm(positions[uav] - buoys[alternative]))
            alternatives.append(alt_distance)
            assigned_distance = float(np.linalg.norm(positions[uav] - buoys[int(assignment[uav])])) if assignment[uav] >= 0 else 0.0
            assigned_minus_alt.append(assigned_distance - alt_distance)
        else:
            alternatives.append(0.0); assigned_minus_alt.append(0.0)
    assignment_values = np.asarray([assignment_scores[uav, buoy] if buoy >= 0 else 0.0 for uav, buoy in enumerate(assignment)], dtype=np.float64)
    top_unserved = float(np.max(weighted[:, unserved])) / max(float(np.max(weighted)), 1.0e-12) if np.any(unserved) else 0.0
    r27_competition = float(np.mean(np.clip(1.0 - (top - runner) / np.maximum(top, 1.0e-12), 0.0, 1.0)))
    return {
        "preference_cosine_similarity": cosine, "same_top_buoy_flag": bool(top_indices[0] == top_indices[1]),
        "top_choice_overlap_strength": top_overlap, "top1_top2_score_margin": margin,
        "average_preference_entropy": preference_entropy, "independent_current_score": independent,
        "exclusive_current_score": exclusive, "assignment_regret": regret,
        "mean_isr": float(np.mean(isr)), "max_isr": float(np.max(isr)), "isr_asymmetry": float(abs(isr[0] - isr[1])),
        "shared_sum_rate_loss_fraction": float(max(zero_rate - shared_rate, 0.0) / max(zero_rate, 1.0e-12)),
        "total_normalized_queue": float(np.sum(normalized_queue)), "top1_queue_share": float(np.max(queues) / max(float(np.sum(queues)), 1.0e-12)),
        "top2_queue_share": float(np.sum(np.sort(queues)[-2:]) / max(float(np.sum(queues)), 1.0e-12)),
        "queue_entropy": _entropy(queues), "queue_imbalance": float((np.max(queues) - np.min(queues)) / float(env.config.queue_capacity_bits)),
        "unserved_high_queue_mass": float(np.sum(queues[unserved]) / float(env.config.queue_capacity_bits)),
        "queue_to_current_capacity_ratio": float(np.sum(queues) / max(capacity, 1.0e-12)),
        "buoy_spatial_spread_m": _weighted_spread(buoys, np.ones(env.num_buoys)),
        "high_queue_buoy_spread_m": _weighted_spread(buoys, normalized_queue),
        "distance_to_preferred_buoy_m": float(np.mean(preferred_distances)),
        "distance_to_best_unassigned_high_value_buoy_m": float(np.mean(alternatives)),
        "assigned_minus_alternative_distance_m": float(np.mean(assigned_minus_alt)),
        "uav_uav_distance_m": pair_distance, "safety_margin_m": pair_distance - float(env.config.multi_uav_min_separation_m),
        "distinct_preferred_buoy_count": float(len(set(int(value) for value in top_indices))),
        "current_assignment_diversity": float(len(assigned_set) / max(env.num_uavs, 1)),
        "assignment_score_asymmetry": float(abs(assignment_values[0] - assignment_values[1]) / max(float(np.sum(np.abs(assignment_values))), 1.0e-12)),
        "assignment_score_difference_normalized": float(abs(assignment_values[0] - assignment_values[1]) / max(float(np.max(np.abs(assignment_values))), 1.0e-12)),
        "current_distinct_served_buoy_count": float(len(assigned_set)),
        "queue_to_h5_current_capacity_ratio": float(np.sum(queues) / max(h5_capacity, 1.0e-12)),
        "unassigned_queue_to_h5_current_capacity_ratio": float(np.sum(queues[unserved]) / max(h5_capacity, 1.0e-12)),
        "top_unserved_opportunity_ratio": top_unserved,
        "interference_conflict_current": bool(float(scenario.interference_overlap_factor) > 0.0 and float(np.max(isr)) >= float(env.options.conflict_isr_threshold)),
        "duplicate_intent_current": bool(top_indices[0] == top_indices[1]),
        "assignment_conflict_current": bool(top_indices[0] == top_indices[1] and assignment[0] >= 0 and assignment[1] >= 0 and assignment[0] != assignment[1]),
        "round27_queue_competition_score": r27_competition, "current_assignment": assignment.tolist(),
        "current_preferred_buoys": top_indices.tolist(), "num_buoys": int(env.num_buoys),
        "eta": float(scenario.interference_overlap_factor),
    }


def _branch(scenario: Any, graph: Mapping[str, Any], timestep: int, *, first_horizon: int, total_horizon: int) -> dict[str, Any]:
    env, info = _restore_env(scenario, graph, timestep)
    signature = _state_signature(info, scenario)
    payload, episode_return = 0.0, 0.0
    decision = joint_plan(env, info, horizon=first_horizon)
    action = decision.action.copy()
    applied_first: np.ndarray | None = None
    for step in range(total_horizon):
        if step:
            action = joint_plan(env, info, horizon=1).action
        _, reward, terminated, truncated, after = env.step(action, matching_mode="joint_aware")
        if step == 0:
            applied_first = np.asarray(after["normalized_applied_action"], dtype=np.float64)
        payload += float(np.sum(np.asarray(after["transmitted_payload_bits"], dtype=np.float64)))
        episode_return += float(reward)
        info = after
        if terminated or (truncated and step + 1 != total_horizon):
            raise RuntimeError("registered counterfactual horizon terminated unexpectedly")
    return {
        "snapshot_signature": signature, "payload_bits": payload, "episode_return": episode_return,
        "residual_queue_bits": float(np.sum(np.asarray(info["raw_queue_bits"], dtype=np.float64))),
        "applied_first_action": applied_first, "first_horizon": first_horizon,
        "future_trace_accessed": bool(decision.future_trace_accessed),
    }


def _utility_rows(pool: Sequence[Mapping[str, Any]], scenarios: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    protocol = config["counterfactual_protocol"]
    tolerance = float(protocol["numerical_utility_tolerance_bits"])
    rows: list[dict[str, Any]] = []
    for item in pool:
        scenario = scenarios[str(item["scenario_id"])]
        timestep, graph = int(item["timestep"]), item["graph_state_t"]
        feature = _feature_row(scenario, graph, timestep)
        h5 = _branch(scenario, graph, timestep, first_horizon=int(protocol["planner_first_horizon"]), total_horizon=int(protocol["primary_horizon"]))
        h1 = _branch(scenario, graph, timestep, first_horizon=int(protocol["reference_first_horizon"]), total_horizon=int(protocol["primary_horizon"]))
        if h5["snapshot_signature"] != h1["snapshot_signature"] or h5["future_trace_accessed"] or h1["future_trace_accessed"]:
            raise RuntimeError("Round-28 primary counterfactual is not frozen causal same-snapshot evidence")
        stored = np.asarray(item["joint_action_applied_t"], dtype=np.float64)
        h1_error = float(np.sqrt(np.mean((np.asarray(h1["applied_first_action"]) - stored) ** 2)))
        if h1_error > 1.0e-9:
            raise RuntimeError("reconstructed frozen H1 state does not reproduce its stored first action")
        utility = float(h5["payload_bits"] - h1["payload_bits"])
        label = "useful" if utility > tolerance else "harmful" if utility < -tolerance else "neutral"
        h10_available = timestep + int(protocol["secondary_horizon"]) <= EPISODE_HORIZON
        h10_utility, h10_payload, h10_reference = 0.0, 0.0, 0.0
        if h10_available:
            h10 = _branch(scenario, graph, timestep, first_horizon=int(protocol["secondary_horizon"]), total_horizon=int(protocol["secondary_horizon"]))
            h1_10 = _branch(scenario, graph, timestep, first_horizon=int(protocol["reference_first_horizon"]), total_horizon=int(protocol["secondary_horizon"]))
            if h10["snapshot_signature"] != h1_10["snapshot_signature"] or h10["future_trace_accessed"] or h1_10["future_trace_accessed"]:
                raise RuntimeError("Round-28 secondary counterfactual is invalid")
            h10_utility, h10_payload, h10_reference = float(h10["payload_bits"] - h1_10["payload_bits"]), float(h10["payload_bits"]), float(h1_10["payload_bits"])
        rows.append({
            "state_id": f"{scenario.scenario_id}__t{timestep:02d}", "scenario_id": scenario.scenario_id,
            "split": scenario.split, "timestep": timestep, "utility_payload": utility,
            "utility_return": float(h5["episode_return"] - h1["episode_return"]),
            "utility_queue": float(h1["residual_queue_bits"] - h5["residual_queue_bits"]),
            "utility_class": label, "planner_payload_bits": float(h5["payload_bits"]),
            "reference_payload_bits": float(h1["payload_bits"]), "planner_residual_queue_bits": float(h5["residual_queue_bits"]),
            "reference_residual_queue_bits": float(h1["residual_queue_bits"]),
            "h5_to_h1_first_action_rmse": float(np.sqrt(np.mean((np.asarray(h5["applied_first_action"]) - np.asarray(h1["applied_first_action"])) ** 2))),
            "stored_h1_action_reproduction_rmse": h1_error, "same_snapshot_verified": True,
            "only_first_action_intervention": True, "continuation_policy": "frozen_causal_joint_h1",
            "continuation_horizon": int(protocol["continuation_horizon"]), "secondary_h10_available": bool(h10_available),
            "secondary_h10_utility_payload": h10_utility, "secondary_h10_payload_bits": h10_payload,
            "secondary_h1_payload_bits": h10_reference, **feature,
        })
    return rows


def _normalize(rows: Sequence[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    train = [row for row in rows if row["split"] == "train"]
    epsilon = float(config["normalization"]["epsilon"])
    parameters: dict[str, dict[str, float]] = {}
    for feature in BASE_FEATURES:
        values = np.asarray([float(row[feature]) for row in train], dtype=np.float64)
        low, high = float(np.min(values)), float(np.max(values))
        parameters[feature] = {"train_min": low, "train_max": high, "range": high - low}
        for row in rows:
            row[f"norm_{feature}"] = float(np.clip((float(row[feature]) - low) / max(high - low, epsilon), 0.0, 1.0))
    return {"round": "round_28", "method": "train_minmax", "fit_split": "train", "validation_used_for_fit": False, "epsilon": epsilon, "features": parameters}


def _composites(rows: Sequence[dict[str, Any]], config: Mapping[str, Any]) -> None:
    definitions = config["composites"]
    for row in rows:
        for name in ("coordination_pressure", "interference_queue_pressure", "deferred_service_pressure"):
            parts, weights = definitions[name]["components"], definitions[name]["weights"]
            row[name] = float(sum(float(weight) * float(row[f"norm_{part}"]) for part, weight in zip(parts, weights)))
        parts, weights = definitions["unified_planning_pressure"]["components"], definitions["unified_planning_pressure"]["weights"]
        row["unified_planning_pressure"] = float(sum(float(weight) * float(row[part]) for part, weight in zip(parts, weights)))


def _rank(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    order = np.argsort(data, kind="mergesort")
    result = np.empty(len(data), dtype=np.float64)
    cursor = 0
    while cursor < len(data):
        end = cursor + 1
        while end < len(data) and data[order[end]] == data[order[cursor]]:
            end += 1
        result[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return 0.0 if a.size < 3 or np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12 else float(np.corrcoef(a, b)[0, 1])


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_rank(left), _rank(right))


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    score, label = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)
    positive, negative = int(np.sum(label)), int(np.sum(~label))
    if positive == 0 or negative == 0:
        return 0.5
    return float((np.sum(_rank(score)[label]) - positive * (positive + 1) / 2.0) / (positive * negative))


def _effect(rows: Sequence[Mapping[str, Any]], feature: str) -> dict[str, float]:
    useful = np.asarray([float(row[feature]) for row in rows if row["utility_class"] == "useful"], dtype=np.float64)
    other = np.asarray([float(row[feature]) for row in rows if row["utility_class"] != "useful"], dtype=np.float64)
    if not useful.size or not other.size:
        return {"mean_difference": 0.0, "standardized_effect": 0.0, "rank_biserial": 0.0, "auc_raw": 0.5}
    pooled = math.sqrt(max(((useful.size - 1) * float(np.var(useful, ddof=1)) + (other.size - 1) * float(np.var(other, ddof=1))) / max(useful.size + other.size - 2, 1), 0.0))
    auc = _auc([float(row[feature]) for row in rows], [row["utility_class"] == "useful" for row in rows])
    return {"mean_difference": float(np.mean(useful) - np.mean(other)), "standardized_effect": float((np.mean(useful) - np.mean(other)) / max(pooled, 1.0e-12)), "rank_biserial": float(2.0 * auc - 1.0), "auc_raw": auc}


def _cluster_bootstrap(rows: Sequence[Mapping[str, Any]], function: Callable[[Sequence[Mapping[str, Any]]], float], *, seed: int, resamples: int) -> tuple[float, float]:
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row["scenario_id"])].append(row)
    keys = sorted(by_scenario)
    if not keys:
        return 0.0, 0.0
    rng, values = np.random.default_rng(seed), np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        selected = rng.integers(0, len(keys), size=len(keys))
        sample = [row for key_index in selected for row in by_scenario[keys[int(key_index)]]]
        values[index] = function(sample)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _separability(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    resamples, seed = int(config["statistics"]["bootstrap_resamples"]), int(config["statistics"]["bootstrap_seed"])
    directions: dict[str, float] = {}
    for feature_index, feature in enumerate(BASE_FEATURES + COMPOSITES):
        train = [row for row in rows if row["split"] == "train"]
        train_effect = _effect(train, feature)
        directions[feature] = 1.0 if train_effect["mean_difference"] >= 0.0 else -1.0
        for split_index, split in enumerate(("train", "validation")):
            population = [row for row in rows if row["split"] == split]
            effect = _effect(population, feature)
            sign = directions[feature]
            low, high = _cluster_bootstrap(population, lambda sample, field=feature: _effect(sample, field)["mean_difference"], seed=seed + feature_index * 10 + split_index, resamples=resamples)
            auc_directed = effect["auc_raw"] if sign > 0.0 else 1.0 - effect["auc_raw"]
            harmful = _effect([row for row in population if row["utility_class"] in {"useful", "harmful"}], feature)
            result.append({
                "feature": feature, "split": split, "positive_class": "useful", "negative_class": "neutral_or_harmful",
                "useful_state_count": sum(row["utility_class"] == "useful" for row in population),
                "nonuseful_state_count": sum(row["utility_class"] != "useful" for row in population),
                "mean_difference_useful_minus_nonuseful": effect["mean_difference"],
                "mean_difference_ci95_low": low, "mean_difference_ci95_high": high,
                "standardized_effect": effect["standardized_effect"], "mann_whitney_rank_biserial": effect["rank_biserial"],
                "roc_auc_raw": effect["auc_raw"], "score_direction_from_train": "higher" if sign > 0.0 else "lower",
                "roc_auc_equivalent_train_oriented": auc_directed,
                "useful_vs_harmful_mean_difference": harmful["mean_difference"],
                "harmful_state_count": sum(row["utility_class"] == "harmful" for row in population),
            })
    return result


def _utility_correlations(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    resamples, seed = int(config["statistics"]["bootstrap_resamples"]), int(config["statistics"]["bootstrap_seed"] + 1000)
    for feature_index, feature in enumerate(BASE_FEATURES + COMPOSITES):
        for split_index, split in enumerate(("train", "validation")):
            population = [row for row in rows if row["split"] == split]
            values, utility = [float(row[feature]) for row in population], [float(row["utility_payload"]) for row in population]
            pearson, spearman = _pearson(values, utility), _spearman(values, utility)
            pearson_ci = _cluster_bootstrap(population, lambda sample, field=feature: _pearson([float(row[field]) for row in sample], [float(row["utility_payload"]) for row in sample]), seed=seed + feature_index * 10 + split_index, resamples=resamples)
            spearman_ci = _cluster_bootstrap(population, lambda sample, field=feature: _spearman([float(row[field]) for row in sample], [float(row["utility_payload"]) for row in sample]), seed=seed + feature_index * 10 + split_index + 2, resamples=resamples)
            result.append({"feature": feature, "split": split, "state_count": len(population), "pearson": pearson, "pearson_ci95_low": pearson_ci[0], "pearson_ci95_high": pearson_ci[1], "spearman": spearman, "spearman_ci95_low": spearman_ci[0], "spearman_ci95_high": spearman_ci[1], "bootstrap_cluster": "scenario_id"})
    return result


def _generalization(separability: Sequence[Mapping[str, Any]], correlations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    sep = {(str(row["feature"]), str(row["split"])): row for row in separability}
    corr = {(str(row["feature"]), str(row["split"])): row for row in correlations}
    result = []
    for feature in BASE_FEATURES + COMPOSITES:
        train_sep, validation_sep = sep[(feature, "train")], sep[(feature, "validation")]
        train_corr, validation_corr = corr[(feature, "train")], corr[(feature, "validation")]
        effect_stable = float(train_sep["mean_difference_useful_minus_nonuseful"]) * float(validation_sep["mean_difference_useful_minus_nonuseful"]) > 0.0
        corr_stable = float(train_corr["spearman"]) * float(validation_corr["spearman"]) > 0.0
        result.append({"feature": feature, "train_effect": train_sep["mean_difference_useful_minus_nonuseful"], "validation_effect": validation_sep["mean_difference_useful_minus_nonuseful"], "train_spearman": train_corr["spearman"], "validation_spearman": validation_corr["spearman"], "effect_direction_stable": effect_stable, "utility_correlation_direction_stable": corr_stable, "feature_direction_stable": bool(effect_stable and corr_stable), "validation_auc_train_oriented": validation_sep["roc_auc_equivalent_train_oriented"]})
    return result


def _thresholds(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    train = [row for row in rows if row["split"] == "train"]
    output: dict[str, Any] = {"round": "round_28", "fit_split": "train", "validation_used_for_fit": False, "primary_threshold": "p75", "secondary_diagnostic_threshold": "median", "composites": {}}
    for feature in COMPOSITES:
        values = np.asarray([float(row[feature]) for row in train], dtype=np.float64)
        output["composites"][feature] = {"train_p75": float(np.percentile(values, 75.0)), "train_median": float(np.median(values))}
    return output


def _trigger_metrics(population: Sequence[Mapping[str, Any]], active: Sequence[Mapping[str, Any]], *, score: str, threshold_kind: str, threshold: float, split: str, seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    inactive = [row for row in population if row not in active]
    useful = [row for row in population if row["utility_class"] == "useful"]
    useful_active = [row for row in active if row["utility_class"] == "useful"]
    harmful = [row for row in active if row["utility_class"] == "harmful"]
    ci = _cluster_bootstrap(active, lambda sample: float(np.mean([float(row["utility_payload"]) for row in sample])) if sample else 0.0, seed=seed, resamples=int(config["statistics"]["bootstrap_resamples"])) if active else (0.0, 0.0)
    return {
        "score": score, "threshold_kind": threshold_kind, "threshold": threshold, "threshold_fit_split": "train", "split": split,
        "population_count": len(population), "activation_count": len(active), "activation_rate": len(active) / max(len(population), 1),
        "active_scenario_count": len({str(row["scenario_id"]) for row in active}),
        "precision_useful": len(useful_active) / max(len(active), 1), "recall_useful": len(useful_active) / max(len(useful), 1),
        "specificity": sum(row["utility_class"] != "useful" for row in inactive) / max(len(inactive), 1),
        "average_active_utility_payload": float(np.mean([float(row["utility_payload"]) for row in active])) if active else 0.0,
        "active_utility_ci95_low": ci[0], "active_utility_ci95_high": ci[1],
        "average_inactive_utility_payload": float(np.mean([float(row["utility_payload"]) for row in inactive])) if inactive else 0.0,
        "harmful_rate_active": len(harmful) / max(len(active), 1), "bootstrap_cluster": "scenario_id",
    }


def _composite_metrics(rows: Sequence[dict[str, Any]], thresholds: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics, grouped = [], []
    seed = int(config["statistics"]["bootstrap_seed"] + 2000)
    for score_index, score in enumerate(COMPOSITES):
        for kind in ("p75", "median"):
            threshold = float(thresholds["composites"][score][f"train_{kind}"])
            field = f"trigger_{score}_{kind}"
            for row in rows:
                row[field] = bool(float(row[score]) >= threshold)
            for split_index, split in enumerate(("train", "validation")):
                population = [row for row in rows if row["split"] == split]
                active = [row for row in population if row[field]]
                metrics.append(_trigger_metrics(population, active, score=score, threshold_kind=kind, threshold=threshold, split=split, seed=seed + score_index * 20 + split_index * 3 + (0 if kind == "p75" else 1), config=config))
            if kind == "p75":
                validation = [row for row in rows if row["split"] == "validation"]
                for group_type, values in (("num_buoys", (3, 5)), ("eta", (0.0, 0.25, 0.5, 1.0))):
                    for value in values:
                        population = [row for row in validation if float(row[group_type]) == float(value)]
                        active = [row for row in population if row[field]]
                        grouped.append({"group_type": group_type, "group_value": value, **_trigger_metrics(population, active, score=score, threshold_kind=kind, threshold=threshold, split="validation", seed=seed + score_index * 100 + len(grouped), config=config)})
    return metrics, grouped


def _round27_revalidation(rows: Sequence[dict[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    train = [row for row in rows if row["split"] == "train"]
    threshold = float(np.percentile([float(row["round27_queue_competition_score"]) for row in train], 75.0))
    result = []
    for row in rows:
        row["round27_A_interference_conflict"] = bool(row["interference_conflict_current"])
        row["round27_B_duplicate_intent"] = bool(row["duplicate_intent_current"])
        row["round27_C_queue_interference"] = bool(row["interference_conflict_current"] and float(row["round27_queue_competition_score"]) >= threshold)
        row["round27_D_assignment_conflict"] = bool(row["assignment_conflict_current"])
    for split_index, split in enumerate(("train", "validation")):
        population = [row for row in rows if row["split"] == split]
        for trigger_index, name in enumerate(("A_interference_conflict", "B_duplicate_intent", "C_queue_interference", "D_assignment_conflict")):
            active = [row for row in population if row[f"round27_{name}"]]
            result.append({"trigger": name, "queue_competition_train_p75": threshold, **_trigger_metrics(population, active, score=f"round27_{name}", threshold_kind="train_p75" if name.startswith("C_") else "none", threshold=threshold, split=split, seed=int(config["statistics"]["bootstrap_seed"]) + 3000 + split_index * 10 + trigger_index, config=config)})
    return result


def _pca(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = [f"norm_{feature}" for feature in BASE_FEATURES]
    train = [row for row in rows if row["split"] == "train"]
    center = np.mean(np.asarray([[float(row[field]) for field in fields] for row in train], dtype=np.float64), axis=0)
    matrix = np.asarray([[float(row[field]) for field in fields] for row in train], dtype=np.float64) - center
    _, singular, vectors = np.linalg.svd(matrix, full_matrices=False)
    components = vectors[:2]
    explained = (singular[:2] ** 2) / max(float(np.sum(singular ** 2)), 1.0e-12)
    result = []
    for row in rows:
        score = (np.asarray([float(row[field]) for field in fields]) - center) @ components.T
        result.append({"state_id": row["state_id"], "scenario_id": row["scenario_id"], "split": row["split"], "timestep": row["timestep"], "utility_class": row["utility_class"], "utility_payload": row["utility_payload"], "pc1": float(score[0]), "pc2": float(score[1]), "pca_fit_split": "train", "pc1_explained_variance_ratio": float(explained[0]), "pc2_explained_variance_ratio": float(explained[1])})
    return result


def _policy_relations(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation = [row for row in rows if row["split"] == "validation"]
    wanted = {(str(row["scenario_id"]), int(row["timestep"])) for row in validation}
    predictor = FrozenDeepSetsPredictor()
    sequence = load_policy_sequences(VALIDATION, predictor)
    indices = [index for index, (scenario, timestep, behavior) in enumerate(zip(sequence.metadata["scenario_id"], sequence.metadata["timestep"], sequence.metadata["behavior_generator"])) if str(behavior) == "joint_h1" and (str(scenario), int(timestep)) in wanted]
    if {(str(sequence.metadata["scenario_id"][index]), int(sequence.metadata["timestep"][index])) for index in indices} != wanted:
        raise RuntimeError("frozen validation policies do not cover the Round-28 H1 state sample")
    manifest22, manifest24 = _load(R22 / "policy_checkpoint_manifest.json"), _load(R24 / "checkpoint_manifest.json")
    records = manifest22["checkpoints"] + manifest24["new_round24_checkpoints"]
    by_family = {family: sorted([record for record in records if record["family"] == family], key=lambda record: int(record["seed"])) for family in (PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER)}
    if any({int(record["seed"]) for record in by_family[family]} != set(range(8)) for family in by_family):
        raise RuntimeError("Round-28 requires frozen eight-seed policy families")
    actions: dict[str, dict[tuple[str, int], list[np.ndarray]]] = {family: defaultdict(list) for family in by_family}
    for family, family_records in by_family.items():
        for record in family_records:
            loaded_family, model, normalizer, payload = load_checkpoint(ROOT / record["checkpoint"])
            if loaded_family != family or int(payload["seed"]) != int(record["seed"]):
                raise RuntimeError("frozen policy checkpoint identity changed")
            predicted = predict(family, model, normalizer.apply(sequence), TrainingConfig())
            for index in indices:
                key = (str(sequence.metadata["scenario_id"][index]), int(sequence.metadata["timestep"][index]))
                actions[family][key].append(np.asarray(predicted[index], dtype=np.float64))
    paired = list(csv.DictReader((R24 / "eight_seed_validation_paired.csv").open(encoding="utf-8", newline="")))
    payload_delta: dict[str, list[float]] = defaultdict(list)
    for row in paired:
        payload_delta[str(row["scenario_id"])].append(float(row["delta_cumulative_payload_bits"]))
    task, seed = [], []
    for row in validation:
        key = (str(row["scenario_id"]), int(row["timestep"]))
        task_actions, state_actions = actions[PREDICTED_TRANSFORMER][key], actions[STATE_ONLY_TRANSFORMER][key]
        if len(task_actions) != 8 or len(state_actions) != 8 or key[0] not in payload_delta:
            raise RuntimeError("frozen policy relation evidence is incomplete")
        disagreement = [float(np.sqrt(np.mean((task_actions[left] - task_actions[right]) ** 2))) for left in range(8) for right in range(left + 1, 8)]
        difference = [float(np.sqrt(np.mean((task_actions[index] - state_actions[index]) ** 2))) for index in range(8)]
        common = {"state_id": row["state_id"], "scenario_id": key[0], "timestep": key[1], "split": "validation", "planner_utility_payload": float(row["utility_payload"]), "unified_planning_pressure": float(row["unified_planning_pressure"]), "taskwp_seed_action_disagreement_rmse": float(np.mean(disagreement)), "seed_count": 8}
        seed.append(dict(common))
        task.append({**common, "taskwp_minus_state_action_rmse": float(np.mean(difference)), "taskwp_minus_state_payload_bits_mean": float(np.mean(payload_delta[key[0]]))})
    for target, field in ((seed, "taskwp_seed_action_disagreement_rmse"), (task, "taskwp_minus_state_action_rmse"), (task, "taskwp_minus_state_payload_bits_mean")):
        evidence = [row for row in target if row["state_id"] != "__summary__"]
        pearson, spearman = _pearson([float(row["planner_utility_payload"]) for row in evidence], [float(row[field]) for row in evidence]), _spearman([float(row["planner_utility_payload"]) for row in evidence], [float(row[field]) for row in evidence])
        target.append({"state_id": "__summary__", "scenario_id": "__summary__", "timestep": -1, "split": "validation", "planner_utility_payload": 0.0, "unified_planning_pressure": 0.0, field: 0.0, "seed_count": 8, "pearson": pearson, "spearman": spearman})
    return seed, task


def _figures(rows: Sequence[Mapping[str, Any]], separability: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]], grouped: Sequence[Mapping[str, Any]], revalidation: Sequence[Mapping[str, Any]], pca: Sequence[Mapping[str, Any]], seed: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    validation_metrics = [row for row in metrics if row["split"] == "validation" and row["threshold_kind"] == "p75"]
    def save(number: int, name: str, draw: Callable[[Any], None]) -> None:
        fig, axis = plt.subplots(figsize=(6.4, 4.2)); draw(axis); fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(FIGURES / f"{number:02d}_{name}.{suffix}", dpi=140)
        plt.close(fig)
    utility = [float(row["utility_payload"]) for row in rows]
    save(1, "expanded_utility_distribution", lambda ax: (ax.hist(utility, bins=24, color="#4477aa"), ax.set(xlabel="U5 payload bits", ylabel="state count")))
    classes = ("useful", "neutral", "harmful")
    save(2, "utility_classes", lambda ax: (ax.bar(classes, [sum(row["utility_class"] == label for row in rows) for label in classes], color=["#44aa77", "#999999", "#cc6677"]), ax.set(ylabel="state count")))
    for number, feature, title in ((3, "preference_cosine_similarity", "preference overlap"), (4, "assignment_regret", "assignment regret"), (5, "max_isr", "ISR pressure"), (6, "queue_to_h5_current_capacity_ratio", "queue-capacity tension"), (7, "unserved_high_queue_mass", "deferred-service pressure"), (8, "coordination_pressure", "coordination pressure"), (9, "interference_queue_pressure", "interference-queue pressure"), (10, "deferred_service_pressure", "deferred-service score"), (11, "unified_planning_pressure", "unified planning pressure")):
        save(number, title.replace(" ", "_"), lambda ax, field=feature, label=title: (ax.scatter([float(row[field]) for row in rows], utility, c=[{"useful":"#44aa77","neutral":"#999999","harmful":"#cc6677"}[row["utility_class"]] for row in rows], s=18), ax.set(xlabel=label, ylabel="U5 payload bits")))
    sep_train = [row for row in separability if row["split"] == "train" and row["feature"] in COMPOSITES]
    sep_val = [row for row in separability if row["split"] == "validation" and row["feature"] in COMPOSITES]
    save(12, "train_validation_feature_effect", lambda ax: (ax.bar(np.arange(4)-.18, [float(row["standardized_effect"]) for row in sep_train], .36, label="train"), ax.bar(np.arange(4)+.18, [float(row["standardized_effect"]) for row in sep_val], .36, label="validation"), ax.set(xticks=np.arange(4), xticklabels=["CP","IQP","DSP","UPP"], ylabel="standardized effect"), ax.legend()))
    save(13, "trigger_active_inactive_utility", lambda ax: (ax.bar([row["score"] for row in validation_metrics], [float(row["average_active_utility_payload"]) for row in validation_metrics], color="#4477aa"), ax.tick_params(axis="x", rotation=25), ax.set(ylabel="active U5 payload bits")))
    save(14, "harmful_rate_by_trigger", lambda ax: (ax.bar([row["score"] for row in validation_metrics], [float(row["harmful_rate_active"]) for row in validation_metrics], color="#cc6677"), ax.tick_params(axis="x", rotation=25), ax.set(ylabel="harmful active rate")))
    selected = validation_metrics[0]["score"] if validation_metrics else "coordination_pressure"
    k_rows = [row for row in grouped if row["score"] == selected and row["group_type"] == "num_buoys"]
    eta_rows = [row for row in grouped if row["score"] == selected and row["group_type"] == "eta"]
    save(15, "k3_k5_trigger_utility", lambda ax: (ax.bar([str(row["group_value"]) for row in k_rows], [float(row["average_active_utility_payload"]) for row in k_rows], color="#4477aa"), ax.set(xlabel="K", ylabel="active U5 payload bits")))
    save(16, "eta_tier_trigger_utility", lambda ax: (ax.bar([str(row["group_value"]) for row in eta_rows], [float(row["average_active_utility_payload"]) for row in eta_rows], color="#4477aa"), ax.set(xlabel="eta", ylabel="active U5 payload bits")))
    save(17, "pca_utility_classes", lambda ax: (ax.scatter([float(row["pc1"]) for row in pca], [float(row["pc2"]) for row in pca], c=[{"useful":"#44aa77","neutral":"#999999","harmful":"#cc6677"}[str(row["utility_class"])] for row in pca], s=18), ax.set(xlabel="train-fit PC1", ylabel="train-fit PC2")))
    rv = [row for row in revalidation if row["split"] == "validation"]
    save(18, "round27_trigger_revalidation", lambda ax: (ax.bar([str(row["trigger"])[0] for row in rv], [float(row["average_active_utility_payload"]) for row in rv], color="#aa4499"), ax.set(xlabel="Round-27 trigger", ylabel="active U5 payload bits")))
    seed_rows = [row for row in seed if row["state_id"] != "__summary__"]
    save(19, "utility_seed_disagreement", lambda ax: (ax.scatter([float(row["planner_utility_payload"]) for row in seed_rows], [float(row["taskwp_seed_action_disagreement_rmse"]) for row in seed_rows], color="#ddcc77", s=20), ax.set(xlabel="U5 payload bits", ylabel="Task-WP seed action disagreement")))
    return sorted(path.name for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"})


def _write_report(rows: Sequence[Mapping[str, Any]], separability: Sequence[Mapping[str, Any]], correlations: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]], gate: Mapping[str, Any], readiness: Mapping[str, Any]) -> None:
    train_corr = [row for row in correlations if row["split"] == "train" and row["feature"] in BASE_FEATURES]
    strongest = max(train_corr, key=lambda row: abs(float(row["spearman"])))
    validation_corr = next(row for row in correlations if row["split"] == "validation" and row["feature"] == strongest["feature"])
    validation_sep = next(row for row in separability if row["split"] == "validation" and row["feature"] == strongest["feature"])
    primary = [row for row in metrics if row["split"] == "validation" and row["threshold_kind"] == "p75"]
    best = max(primary, key=lambda row: float(row["average_active_utility_payload"]))
    counts = {label: sum(row["utility_class"] == label for row in rows) / len(rows) for label in ("useful", "neutral", "harmful")}
    lines = ["# Round-28 Completion Report", "", f"1. Expanded diagnostic states: {len(rows)}.", f"2. Train / validation states: {sum(row['split']=='train' for row in rows)} / {sum(row['split']=='validation' for row in rows)}.", f"3. Planner-useful ratio: {counts['useful']:.6f}.", f"4. Neutral ratio: {counts['neutral']:.6f}.", f"5. Harmful ratio: {counts['harmful']:.6f}.", f"6. Strongest train-selected single current feature: {strongest['feature']}.", f"7. Train / validation Spearman: {float(strongest['spearman']):.6f} / {float(validation_corr['spearman']):.6f}.", f"8. Validation AUC equivalent: {float(validation_sep['roc_auc_equivalent_train_oriented']):.6f}.", "9. Preference-overlap evidence is recorded in planning_feature_separability.csv.", "10. Assignment-regret evidence is recorded in planning_feature_separability.csv.", "11. ISR-pressure evidence is recorded in planning_feature_separability.csv.", "12. Queue-capacity-tension evidence is recorded in planning_feature_separability.csv."]
    labels = (("Coordination Pressure", "coordination_pressure"), ("IQP", "interference_queue_pressure"), ("DSP", "deferred_service_pressure"), ("UPP", "unified_planning_pressure"))
    for index, (label, score) in enumerate(labels, start=13):
        row = next(item for item in primary if item["score"] == score)
        lines.append(f"{index}. {label} validation active utility + CI: {float(row['average_active_utility_payload']):.6f} [{float(row['active_utility_ci95_low']):.6f}, {float(row['active_utility_ci95_high']):.6f}].")
    lines.extend([f"17. Best composite by pre-registered validation active utility ranking: {best['score']}.", f"18. Its activation rate: {float(best['activation_rate']):.6f}.", f"19. Its precision / recall: {float(best['precision_useful']):.6f} / {float(best['recall_useful']):.6f}.", f"20. Its harmful rate: {float(best['harmful_rate_active']):.6f}.", "21. K3/K5 robustness is recorded in composite_trigger_grouped_metrics.csv.", "22. Eta=.25/.5 robustness is recorded in composite_trigger_grouped_metrics.csv.", "23. Expanded Round-27 A/B/C/D revalidation is recorded in round27_trigger_revalidation.csv.", "24. Train-fit PCA class structure is recorded in planning_need_pca_audit.csv.", "25. Planning utility / seed-disagreement relation is recorded in planning_utility_seed_disagreement_v2.csv.", "26. Planning pressure / frozen Task-WP relation is recorded in planning_pressure_taskwp_relation.csv.", f"27. Representation Gate: {gate['gate']}.", f"28. causal_planning_need_separable: {gate['causal_planning_need_separable']}.", f"29. minimal_trigger_model_training_ready: {gate['minimal_trigger_model_training_ready']}.", f"30. Round-28 recommendation: {readiness['recommendation']}.", "31. Fresh new rollout count: 0.", "32. Untouched-evaluation access count: 0.", "33. Round-29 created: false."])
    (ROUND / "report" / "round28_completion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gates(metrics: Sequence[Mapping[str, Any]], grouped: Sequence[Mapping[str, Any]], generalization: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    criteria = config["representation_gate"]
    general = {str(row["feature"]): row for row in generalization}
    strict, directional = [], []
    for score in COMPOSITES:
        validation = next(row for row in metrics if row["score"] == score and row["split"] == "validation" and row["threshold_kind"] == "p75")
        groups = [row for row in grouped if row["score"] == score]
        k_ok = all(float(next(row for row in groups if row["group_type"] == "num_buoys" and float(row["group_value"]) == value)["average_active_utility_payload"]) > 0.0 for value in (3.0, 5.0))
        eta_ok = all(float(next(row for row in groups if row["group_type"] == "eta" and float(row["group_value"]) == value)["average_active_utility_payload"]) > 0.0 for value in (0.25, 0.5))
        stable = bool(general[score]["feature_direction_stable"] and float(general[score]["validation_auc_train_oriented"]) >= float(criteria["auc_min"]) and float(validation["activation_rate"]) >= float(config["threshold_protocol"]["minimum_validation_activation_rate"]) and int(validation["activation_count"]) >= int(config["threshold_protocol"]["minimum_validation_active_states"]) and float(validation["average_active_utility_payload"]) > float(criteria["active_utility_min_bits"]) and float(validation["active_utility_ci95_low"]) > float(criteria["active_utility_ci_lower_min_bits"]) and float(validation["harmful_rate_active"]) <= float(criteria["harmful_rate_max"]) and k_ok and eta_ok)
        weak = bool(general[score]["feature_direction_stable"] and float(general[score]["validation_auc_train_oriented"]) >= float(criteria["directionally_stable_auc_min"]))
        if stable: strict.append(score)
        if weak: directional.append(score)
    gate_name = "REP-A" if strict else "REP-B" if directional else "REP-C"
    gate = {"round": "round_28", "gate": gate_name, "strict_stable_composites": strict, "directionally_stable_composites": directional, "causal_planning_need_separable": bool(strict), "minimal_trigger_model_training_ready": bool(strict), "bridge_used_as_trigger": False, "training_performed": False}
    trigger = {"round": "round_28", "gate": gate_name, "minimal_trigger_model_training_ready": bool(strict), "training_authorized_this_round": False, "eligible_future_model": "tiny_interpretable_trigger_only" if strict else "none", "criteria": {"stable_composite": bool(strict), "train_validation_direction": bool(directional)}}
    recommendation = "R28-A_future_tiny_interpretable_trigger_model_only" if strict else "R28-B_continue_current_state_representation_study_no_training" if directional else "R28-C_stop_general_planner_trigger_route"
    readiness = {"round": "round_28", "gate": "R28-A" if strict else "R28-B" if directional else "R28-C", "recommendation": recommendation, "training_performed": False, "trigger_model_training_performed": False, "fresh_new_rollout_count": 0, "untouched_eval_access_count": 0, "round29_created": False}
    return gate, trigger, readiness


def run_round28() -> dict[str, Any]:
    config, frozen = _load(CONFIG_PATH), verify_frozen_state()
    scenarios = _scenario_map(); pool = _allowed_h1_states(config, scenarios)
    sampling = {"round": "round_28", **config["state_sampling"], "episode_horizon": EPISODE_HORIZON, "primary_horizon": config["counterfactual_protocol"]["primary_horizon"], "secondary_horizon": config["counterfactual_protocol"]["secondary_horizon"], "source_file": (R20 / "multi_uav_transitions.jsonl").relative_to(ROOT).as_posix(), "source_file_sha256": _digest(R20 / "multi_uav_transitions.jsonl")}
    corpus = {"round": "round_28", "selection_precedes_utility_labels": True, "state_count": len(pool), "split_counts": {split: sum(row["split"] == split for row in pool) for split in sorted(ALLOWED)}, "states": [{"state_id": f"{row['scenario_id']}__t{int(row['timestep']):02d}", "scenario_id": row["scenario_id"], "split": row["split"], "timestep": row["timestep"], "source_behavior_generator": row["behavior_generator"]} for row in pool]}
    _json(OUTPUT / "round28_state_sampling_protocol.json", sampling); _json(OUTPUT / "planner_utility_state_corpus_manifest.json", corpus)
    feature_table = OUTPUT / "planning_need_feature_table.csv"
    if feature_table.is_file():
        rows = _feature_rows(feature_table)
        expected_ids = {f"{row['scenario_id']}__t{int(row['timestep']):02d}" for row in pool}
        if len(rows) != len(pool) or {str(row["state_id"]) for row in rows} != expected_ids:
            raise RuntimeError("active Round-28 feature-table resume identity check failed")
        normalization, thresholds = _load(OUTPUT / "planning_need_feature_normalization.json"), _load(OUTPUT / "round28_trigger_thresholds.json")
    else:
        rows = _utility_rows(pool, scenarios, config)
        normalization = _normalize(rows, config); _json(OUTPUT / "planning_need_feature_normalization.json", normalization)
        _composites(rows, config)
        thresholds = _thresholds(rows, config); _json(OUTPUT / "round28_trigger_thresholds.json", thresholds)
        _csv(feature_table, rows)
    separability = _separability(rows, config); correlations = _utility_correlations(rows, config); generalization = _generalization(separability, correlations)
    _csv(OUTPUT / "planning_feature_separability.csv", separability); _csv(OUTPUT / "planning_feature_utility_correlation.csv", correlations); _csv(OUTPUT / "feature_generalization_audit.csv", generalization)
    composite_spec = {"round": "round_28", "source": "pre-registered config before validation", "normalization": normalization["method"], "definitions": config["composites"], "validation_driven_weights": False}
    (OUTPUT / "planning_composite_score_spec.md").write_text("# Planning composite score specification\n\nAll four scores use the pre-registered train-minmax components and equal weights in `round28_planning_need_representation_config.json`. No validation outcome selects a component or weight.\n", encoding="utf-8")
    metrics, grouped = _composite_metrics(rows, thresholds, config); revalidation = _round27_revalidation(rows, config)
    _csv(OUTPUT / "composite_trigger_metrics.csv", metrics); _csv(OUTPUT / "composite_trigger_grouped_metrics.csv", grouped); _csv(OUTPUT / "round27_trigger_revalidation.csv", revalidation)
    pca = _pca(rows); _csv(OUTPUT / "planning_need_pca_audit.csv", pca)
    seed, task = _policy_relations(rows, config); _csv(OUTPUT / "planning_utility_seed_disagreement_v2.csv", seed); _csv(OUTPUT / "planning_pressure_taskwp_relation.csv", task)
    gate, trigger, readiness = _gates(metrics, grouped, generalization, config)
    _json(OUTPUT / "planning_need_representation_gate.json", gate); _json(OUTPUT / "trigger_model_readiness_gate.json", trigger); _json(OUTPUT / "round28_readiness_gate.json", readiness)
    feature_spec = "# Current-observable planning-need features\n\nTask preference is normalized current queue times current direct rate; assignment regret is independent-score minus injective exclusive-score over independent-score. Interference features use current deterministic joint-aware matching only. Queue/capacity, geometry, role, and H5 static-capacity proxies use no future state or planner action. Bridge, behavior, quality, Task-Waypoint targets, future values, and planner actions are excluded from deployable features.\n"
    (OUTPUT / "planning_need_feature_spec.md").write_text(feature_spec, encoding="utf-8")
    access = {"round": "round_28", "allowed_splits_decoded": ["train", "validation"], "source_behavior_generator": "joint_h1", "stored_h1_state_rows_decoded": len(pool), "development_test_rows_decoded": 0, "fresh_new_rollout_count": 0, "round20_development_test_reuse_count": 0, "round24_fresh_access_count": 0, "untouched_eval_access_count": 0, "future_trigger_feature_access_count": 0, "new_scenarios_created": 0, "policy_checkpoint_inference_only": True, "bridge_used_as_trigger": False, "training_performed": False, "trigger_model_training_performed": False, "counterfactual_diagnostic_branch_count": 2 * len(rows) + 2 * sum(bool(row["secondary_h10_available"]) for row in rows)}
    _json(OUTPUT / "access_audit.json", access)
    figures = _figures(rows, separability, metrics, grouped, revalidation, pca, seed)
    run = {"round": "round_28", "training_performed": False, "trigger_model_training_performed": False, "planning_need_representation_audit": True, "planner_utility_replay_performed": True, "formal_sample_counts": {"state_pool": len(rows), "train": sum(row["split"] == "train" for row in rows), "validation": sum(row["split"] == "validation" for row in rows), "counterfactual_branches": int(access["counterfactual_diagnostic_branch_count"])}, "figure_file_count": len(figures), "frozen_state": frozen, "fresh_new_rollout_count": 0, "untouched_eval_access_count": 0, "round29_created": False}
    _json(OUTPUT / "run_manifest.json", run)
    _write_report(rows, separability, correlations, metrics, gate, readiness)
    (ROUND / "logs" / "implementation_log.md").write_text("# Round-28 implementation log\n\nRead only stored Round-20 train/validation Joint-H1 snapshots, frozen planners, and frozen policy checkpoints. Counterfactual branches are same-snapshot diagnostic replays, not new scenarios or fresh rollouts. No optimizer, backward pass, classifier, model training, checkpoint write, development-test decode, or untouched-evaluation access occurred.\n", encoding="utf-8")
    return {"rows": rows, "gate": gate, "readiness": readiness, "figures": figures, "access": access}


def main() -> None:
    run_round28()


if __name__ == "__main__":
    main()
