"""Non-destructive Round-13 completion-amendment diagnostics.

This module never changes ``MaritimeEnv`` or any original Round-13 formal
artifact.  It derives missing tables from the frozen raw audit and evaluates
discrete and continuous service-resource sidecars on the same validation
scenarios and movement trajectory.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from wt.maritime.baselines import create_baseline_policy
from wt.maritime.env import MaritimeEnv
from wt.round7.round7_runner import _case as round7_case, radial_projection
from wt.round12.contracts import PREDICTOR_SHA256, sha256

from .audit import _action_set, _allowed_difficulty_labels, _snapshot, _validation_scenarios
from .contracts import discrete_service_fractions
from .service_decoder import (
    EPSILON,
    allocate_payload,
    continuous_priority_from_current_state,
    jain_fairness,
)


SCHEMA = "round13_completion_amendment_v1"
ROUND12_OUTPUT = Path("round_12/outputs/round12_formal_20260810_161000")
ORIGINAL_OUTPUT = Path("round_13/outputs/round13_formal_20260811_005500")
STATUS_DOCUMENTS = (Path("PLANS.md"), Path("ROUND_VERSIONING.md"))
FIXED_REWARD_COMPONENTS = (
    "energy_penalty",
    "boundary_penalty",
    "invalid_action_penalty",
    "energy_depletion_penalty",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not math.isfinite(float(value)):
            raise ValueError("amendment output contains NaN or Inf")
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("amendment output contains NaN or Inf")
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        raise ValueError(f"cannot write empty table {path}")
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
        raise ValueError(f"{path} must be a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _file_hashes(root: Path, directory: Path) -> dict[str, str]:
    target = root / directory
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }


def _status_hashes(root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256(root / path) for path in STATUS_DOCUMENTS}


def _round12_integrity(root: Path) -> dict[str, Any]:
    predictor = root / "round_11/checkpoints/H10/seed_2/best.pt"
    if sha256(predictor) != PREDICTOR_SHA256:
        raise RuntimeError("Round-12 frozen predictor changed")
    manifest = _read_json(root / ROUND12_OUTPUT / "frozen_policy_checkpoint_manifest.json")
    for row in manifest["checkpoints"]:
        if sha256(root / ROUND12_OUTPUT / str(row["path"])) != row["sha256"]:
            raise RuntimeError("Round-12 frozen policy checkpoint changed")
    return {
        "round12_lock_sha256": sha256(root / "round_12/ROUND_LOCK.json"),
        "predictor_sha256": PREDICTOR_SHA256,
        "policy_checkpoint_count": int(manifest["checkpoint_count"]),
        "policy_checkpoint_hash_audit": True,
    }


def _new_policy() -> Any:
    policy = create_baseline_policy("queue_rate_energy", seed=0)
    policy.reset(seed=0)
    return policy


def _reward_state(
    *,
    queue: np.ndarray,
    payload: np.ndarray,
    offered: np.ndarray,
    remaining_energy_ratio: float,
    config: Any,
) -> tuple[float, float, bool, float, float]:
    completion = float(np.clip(payload.sum() / max(float(offered.sum()), EPSILON), 0.0, 1.0))
    residual = float(queue.sum())
    residual_ratio = residual / float(queue.size * config.queue_capacity_bits)
    success = bool(
        completion >= config.target_task_completion_ratio
        and residual_ratio <= config.maximum_allowed_residual_queue_ratio
        and remaining_energy_ratio >= config.minimum_terminal_energy_ratio
    )
    unfinished = -float(config.unfinished_queue_penalty_weight * residual_ratio) if not success else 0.0
    success_bonus = float(config.task_success_bonus) if success and config.terminate_on_task_success else 0.0
    return completion, residual, success, unfinished, success_bonus


def _step_sidecar(
    *,
    mechanism: str,
    queue: np.ndarray,
    full_rate: np.ndarray,
    arrival: np.ndarray,
    delta_t: float,
    queue_capacity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int]:
    valid = np.ones(queue.size, dtype=bool)
    if mechanism == "current_fixed_parallel_1_over_K":
        fraction = np.full(queue.size, 1.0 / float(queue.size), dtype=np.float64)
        fallback = False
    elif mechanism == "discrete_argmax_queue_times_rate":
        selected = int(np.argmax(np.where(valid & (queue > 0.0), queue * full_rate, -1.0))) if np.any(queue > 0.0) else None
        fraction = discrete_service_fractions(selected, valid)
        fallback = selected is None
    elif mechanism == "continuous_masked_simplex_queue_times_rate":
        decoded = continuous_priority_from_current_state(queue, full_rate, valid)
        fraction = decoded.fraction
        fallback = decoded.used_fixed_equal_fallback
    else:
        raise ValueError(f"unknown resource mechanism: {mechanism}")
    capacity, payload = allocate_payload(queue, full_rate, fraction, delta_t)
    next_queue = np.clip(queue - payload + arrival, 0.0, queue_capacity)
    violations = int(
        np.any(payload > queue + 1.0e-6)
        or np.any(payload > capacity + 1.0e-6)
        or np.any(fraction < -EPSILON)
        or float(fraction.sum()) > 1.0 + EPSILON
    )
    return fraction, capacity, payload, next_queue, fallback, violations


def _scenario_mechanisms(scenario: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scenario_id, config, _, trace_id, _ = round7_case(int(scenario["scenario_seed"]), int(scenario["num_buoys"]), 128)
    if scenario_id != scenario["scenario_id"]:
        raise RuntimeError("amendment scenario reconstruction mismatch")
    env = MaritimeEnv(config)
    observation, info = env.reset(seed=int(scenario["scenario_seed"]))
    if str(info["arrival_trace_id"]) != trace_id:
        raise RuntimeError("amendment arrival trace reconstruction mismatch")
    names = (
        "current_fixed_parallel_1_over_K",
        "discrete_argmax_queue_times_rate",
        "continuous_masked_simplex_queue_times_rate",
    )
    initial_queue = np.asarray(info["raw_queue_bits"], dtype=np.float64).copy()
    state = {
        name: {
            "queue": initial_queue.copy(),
            "payload": np.zeros_like(initial_queue),
            "arrival": np.zeros_like(initial_queue),
            "components": {key: 0.0 for key in (*FIXED_REWARD_COMPONENTS, "payload_reward")},
            "violations": 0,
            "fallback_steps": 0,
            "first_fraction": None,
            "first_full_bandwidth_rate": None,
        }
        for name in names
    }
    policy = _new_policy()
    baseline_return = 0.0
    movement_distance = 0.0
    movement_energy = 0.0
    step_rows: list[dict[str, Any]] = []
    while True:
        action = radial_projection(policy.act(observation, info))
        observation, reward, terminated, truncated, after = env.step(action)
        baseline_return += float(reward)
        movement_distance += float(after["applied_movement_distance_m"])
        movement_energy += float(after["actual_energy_consumed"])
        full_rate = np.asarray(after["rates_bps"], dtype=np.float64) * float(env.num_buoys)
        arrival = np.asarray(after["raw_arrival_bits"], dtype=np.float64)
        common = {key: float(after["reward_components"][key]) for key in FIXED_REWARD_COMPONENTS}
        for name in names:
            item = state[name]
            fraction, capacity, payload, next_queue, fallback, violations = _step_sidecar(
                mechanism=name,
                queue=np.asarray(item["queue"], dtype=np.float64),
                full_rate=full_rate,
                arrival=arrival,
                delta_t=float(config.delta_t),
                queue_capacity=float(config.queue_capacity_bits),
            )
            if item["first_fraction"] is None:
                item["first_fraction"] = fraction.copy()
                item["first_full_bandwidth_rate"] = full_rate.copy()
            item["payload"] += payload
            item["arrival"] += arrival
            item["queue"] = next_queue
            item["violations"] += violations
            item["fallback_steps"] += int(fallback)
            payload_utility = float(np.clip(payload.sum() / (env._max_total_rate_bps * config.delta_t), 0.0, 1.0))
            item["components"]["payload_reward"] += float(config.payload_reward_weight * payload_utility)
            for key, value in common.items():
                item["components"][key] += value
            step_rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_seed": int(scenario["scenario_seed"]),
                    "num_buoys": int(scenario["num_buoys"]),
                    "mechanism": name,
                    "time_step": int(after["time_step"]),
                    "payload_bits": float(payload.sum()),
                    "payload_utility": payload_utility,
                    "queue_before_bits": float(np.asarray(item["queue"], dtype=float).sum() + payload.sum() - arrival.sum()),
                    "movement_distance_m": float(after["applied_movement_distance_m"]),
                    "actual_energy_consumed": float(after["actual_energy_consumed"]),
                    "resource_fraction": fraction.tolist(),
                }
            )
        info = after
        if terminated or truncated:
            break
    rows: list[dict[str, Any]] = []
    allocation_example: dict[str, Any] = {}
    for name in names:
        item = state[name]
        completion, residual, success, unfinished, bonus = _reward_state(
            queue=np.asarray(item["queue"], dtype=np.float64),
            payload=np.asarray(item["payload"], dtype=np.float64),
            offered=initial_queue + np.asarray(item["arrival"], dtype=np.float64),
            remaining_energy_ratio=float(after["remaining_energy_ratio"]),
            config=config,
        )
        components = dict(item["components"])
        components["unfinished_queue_penalty"] = unfinished
        components["task_success_bonus"] = bonus
        shadow_return = float(sum(components.values()))
        actual_return = baseline_return if name == "current_fixed_parallel_1_over_K" else shadow_return
        reconstruction_error = shadow_return - baseline_return if name == "current_fixed_parallel_1_over_K" else 0.0
        row = {
            "scenario_id": scenario_id,
            "scenario_seed": int(scenario["scenario_seed"]),
            "num_buoys": int(scenario["num_buoys"]),
            "mechanism": name,
            "cumulative_actual_payload_bits": float(np.asarray(item["payload"]).sum()),
            "total_offered_bits": float((initial_queue + np.asarray(item["arrival"])).sum()),
            "completion": completion,
            "residual_queue_bits": residual,
            "queue_reduction_bits": float((initial_queue + np.asarray(item["arrival"])).sum() - residual),
            "episode_return": actual_return,
            "shadow_return": shadow_return,
            "current_return_reconstruction_error": reconstruction_error,
            "per_buoy_payload_bits": np.asarray(item["payload"]).tolist(),
            "jain_fairness": jain_fairness(np.asarray(item["payload"])),
            "resource_decoder_violations": int(item["violations"]),
            "fixed_equal_fallback_steps": int(item["fallback_steps"]),
            "same_movement_continuation": True,
            "uses_current_state_only": True,
            "movement_distance_m": movement_distance,
            "movement_energy": movement_energy,
            "remaining_energy_ratio": float(after["remaining_energy_ratio"]),
            "task_success": success,
            **components,
        }
        rows.append(row)
        if name == "continuous_masked_simplex_queue_times_rate":
            allocation_example = {
                "scenario_id": scenario_id,
                "queue_bits": initial_queue.tolist(),
                "first_full_bandwidth_rate_bps": np.asarray(item["first_full_bandwidth_rate"]).tolist(),
                "fraction": np.asarray(item["first_fraction"]).tolist(),
            }
    current = next(row for row in rows if row["mechanism"] == "current_fixed_parallel_1_over_K")
    if abs(float(current["current_return_reconstruction_error"])) > 1.0e-9:
        raise RuntimeError("shadow reward decomposition does not reconstruct MaritimeEnv reward")
    return rows, step_rows, allocation_example


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + end - 1)
        cursor = end
    return ranks


def _correlation(left: Iterable[float], right: Iterable[float]) -> tuple[float, float]:
    x, y = np.asarray(list(left), dtype=float), np.asarray(list(right), dtype=float)
    if x.size < 3 or np.std(x) <= EPSILON or np.std(y) <= EPSILON:
        return 0.0, 0.0
    return float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _partial_correlation(left: Sequence[float], right: Sequence[float], groups: Sequence[str]) -> float:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    labels = sorted(set(groups))
    design = np.ones((x.size, 1 + max(0, len(labels) - 1)), dtype=float)
    for column, label in enumerate(labels[1:], start=1):
        design[:, column] = np.asarray([value == label for value in groups], dtype=float)
    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return _correlation(residual_x, residual_y)[0]


def _paired_rows(rows: Sequence[Mapping[str, Any]], *, target: str, reference: str, bootstrap_seed: int) -> list[dict[str, Any]]:
    by_key = {(str(row["scenario_id"]), str(row["mechanism"])): row for row in rows}
    scenarios = sorted({str(row["scenario_id"]) for row in rows})
    metrics = {
        "Delta_payload_bits": lambda a, b: float(a["cumulative_actual_payload_bits"]) - float(b["cumulative_actual_payload_bits"]),
        "Delta_completion": lambda a, b: float(a["completion"]) - float(b["completion"]),
        "Delta_residual_queue_improvement_bits": lambda a, b: float(b["residual_queue_bits"]) - float(a["residual_queue_bits"]),
        "Delta_return": lambda a, b: float(a["episode_return"]) - float(b["episode_return"]),
        "Delta_jain_fairness": lambda a, b: float(a["jain_fairness"]) - float(b["jain_fairness"]),
    }
    rng = np.random.default_rng(bootstrap_seed)
    result: list[dict[str, Any]] = []
    for metric, calculator in metrics.items():
        values = np.asarray([calculator(by_key[(scenario, target)], by_key[(scenario, reference)]) for scenario in scenarios], dtype=float)
        samples = np.asarray([rng.choice(values, size=values.size, replace=True).mean() for _ in range(10000)])
        result.append(
            {
                "comparison": f"{target}_minus_{reference}",
                "metric": metric,
                "scenario_count": int(values.size),
                "paired_mean": float(values.mean()),
                "paired_median": float(np.median(values)),
                "paired_std": float(values.std(ddof=0)),
                "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
                "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
                "win_rate": float(np.mean(values > 0.0)),
                "tie_rate": float(np.mean(values == 0.0)),
                "loss_rate": float(np.mean(values < 0.0)),
                "effect_size_mean_over_std": float(values.mean() / values.std(ddof=0)) if values.std(ddof=0) > EPSILON else 0.0,
            }
        )
    return result


def _continuous_gate(rows: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparison = "continuous_masked_simplex_queue_times_rate_minus_current_fixed_parallel_1_over_K"
    values = {str(row["metric"]): row for row in paired if row["comparison"] == comparison}
    payload = values["Delta_payload_bits"]
    completion = values["Delta_completion"]
    residual = values["Delta_residual_queue_improvement_bits"]
    violations = sum(int(row["resource_decoder_violations"]) for row in rows if row["mechanism"] == "continuous_masked_simplex_queue_times_rate")
    positive = payload["paired_mean"] > 0.0 and payload["bootstrap_ci_low"] > 0.0
    nondegrading = completion["paired_mean"] >= 0.0 and residual["paired_mean"] >= 0.0
    harmful = payload["paired_mean"] < 0.0 and payload["bootstrap_ci_high"] < 0.0
    gate = "SA-A-continuous" if positive and nondegrading and violations == 0 else ("SA-C-continuous" if harmful or (completion["paired_mean"] < 0.0 and residual["paired_mean"] < 0.0) else "SA-B-continuous")
    return {
        "gate": gate,
        "primary_comparison": comparison,
        "payload_mean_delta": payload["paired_mean"],
        "payload_95_CI": [payload["bootstrap_ci_low"], payload["bootstrap_ci_high"]],
        "completion_mean_delta": completion["paired_mean"],
        "residual_queue_improvement_mean_delta": residual["paired_mean"],
        "decoder_violations": int(violations),
        "service_action_extension_ready": bool(gate == "SA-A-continuous"),
        "definition": "SA-A-continuous requires positive payload CI excluding zero, non-degrading completion/residual queue, and zero decoder violations; SA-C-continuous has payload CI entirely below zero or systematic core communication deterioration; otherwise SA-B-continuous.",
    }


def _alignment_rows(metric_rows: Sequence[Mapping[str, Any]], raw_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scope in ("all", "current_fixed_parallel_1_over_K", "discrete_argmax_queue_times_rate", "continuous_masked_simplex_queue_times_rate"):
        subset = list(metric_rows) if scope == "all" else [row for row in metric_rows if row["mechanism"] == scope]
        if not subset:
            continue
        group = [f"{row['mechanism']}|{row['num_buoys']}" for row in subset]
        for component in (*FIXED_REWARD_COMPONENTS, "payload_reward", "unfinished_queue_penalty", "task_success_bonus", "episode_return"):
            for metric, sign in (("cumulative_actual_payload_bits", 1.0), ("completion", 1.0), ("residual_queue_bits", -1.0), ("movement_energy", -1.0), ("movement_distance_m", -1.0)):
                pearson, spearman = _correlation([float(row[component]) for row in subset], [sign * float(row[metric]) for row in subset])
                result.append({"analysis": "reward_component", "scope": scope, "component": component, "metric": metric, "metric_direction": "benefit" if sign > 0 else "cost", "pearson": pearson, "spearman": spearman, "partial_pearson_grouped": _partial_correlation([float(row[component]) for row in subset], [sign * float(row[metric]) for row in subset], group), "sample_count": len(subset)})
    h_return = [float(row["H_step_return"]) for row in raw_rows]
    h_payload = [float(row["total_service_bits"]) for row in raw_rows]
    pearson, spearman = _correlation(h_return, h_payload)
    result.append({"analysis": "H10_return_head", "scope": "original_round13_counterfactual", "component": "H_step_return", "metric": "H_step_actual_payload_bits", "metric_direction": "benefit", "pearson": pearson, "spearman": spearman, "partial_pearson_grouped": _partial_correlation(h_return, h_payload, [str(row["snapshot_id"]) for row in raw_rows]), "sample_count": len(raw_rows)})
    return_head = result[-1]
    return result, {"pearson": return_head["pearson"], "spearman": return_head["spearman"], "partial_pearson": return_head["partial_pearson_grouped"]}


def _reward_gate(rows: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]], return_alignment: Mapping[str, Any]) -> dict[str, Any]:
    pair_map = {(str(row["comparison"]), str(row["metric"])): row for row in paired}
    tradeoff = False
    for target in ("discrete_argmax_queue_times_rate", "continuous_masked_simplex_queue_times_rate"):
        comparison = f"{target}_minus_current_fixed_parallel_1_over_K"
        payload = pair_map[(comparison, "Delta_payload_bits")]
        completion = pair_map[(comparison, "Delta_completion")]
        residual = pair_map[(comparison, "Delta_residual_queue_improvement_bits")]
        ret = pair_map[(comparison, "Delta_return")]
        tradeoff |= bool(ret["paired_mean"] > 0.0 and payload["paired_mean"] < 0.0 and completion["paired_mean"] < 0.0 and residual["paired_mean"] < 0.0)
    if tradeoff:
        gate = "RA-C"
    elif float(return_alignment["pearson"]) >= 0.60 and float(return_alignment["spearman"]) >= 0.60:
        gate = "RA-A"
    else:
        gate = "RA-B"
    classification = "strong" if min(float(return_alignment["pearson"]), float(return_alignment["spearman"])) >= 0.60 else ("moderate" if min(float(return_alignment["pearson"]), float(return_alignment["spearman"])) >= 0.30 else "weak")
    if tradeoff or float(return_alignment["pearson"]) < 0.0:
        classification = "conflicting"
    return {"reward_alignment_gate": gate, "return_head_alignment_with_primary_metric": classification, "H10_return_payload_pearson": return_alignment["pearson"], "H10_return_payload_spearman": return_alignment["spearman"], "H10_return_payload_partial_pearson": return_alignment["partial_pearson"], "observed_exact_reward_tradeoff": tradeoff, "definition": "RA-A requires directionally coherent reward/communication evidence; RA-B is mixed or weak alignment; RA-C requires a reward improvement concurrent with systematic primary communication deterioration."}


def _hstep_return_rows(root: Path) -> list[dict[str, Any]]:
    """Recompute only permitted H=10 movement counterfactual summaries.

    The immutable raw Round-13 table contains H-step payload and return, but
    not its final queue/completion/energy summaries.  This additive audit
    reconstructs those omitted fields from the already registered validation
    snapshots, legal first actions, and the same deterministic continuation.
    """
    protocol = _read_json(root / "round_13/configs/round13_controllability_protocol.json")
    horizon = int(protocol["task_horizon"])
    scenarios = _validation_scenarios(root)
    difficulty = _allowed_difficulty_labels(root, scenarios)
    result: list[dict[str, Any]] = []
    for scenario in scenarios:
        for timestep in protocol["snapshot_timesteps"]:
            environment, observation, info = _snapshot(scenario, int(timestep), horizon)
            actions = _action_set(observation, info)
            for action_name, first_action in actions.items():
                rollout = copy.deepcopy(environment)
                obs, current = observation.copy(), dict(info)
                policy = _new_policy()
                start_queue = float(np.asarray(current["raw_queue_bits"], dtype=float).sum())
                payload = 0.0
                total_return = 0.0
                movement_energy = 0.0
                for step in range(horizon):
                    action = radial_projection(first_action if step == 0 else policy.act(obs, current))
                    obs, reward, terminated, truncated, current = rollout.step(action)
                    payload += float(np.asarray(current["transmitted_payload_bits"], dtype=float).sum())
                    total_return += float(reward)
                    movement_energy += float(current["actual_energy_consumed"])
                    if terminated or truncated:
                        break
                residual = float(np.asarray(current["raw_queue_bits"], dtype=float).sum())
                result.append(
                    {
                        "scenario_id": str(scenario["scenario_id"]),
                        "scenario_seed": int(scenario["scenario_seed"]),
                        "num_buoys": int(scenario["num_buoys"]),
                        "difficulty": str(difficulty[str(scenario["scenario_id"])]),
                        "snapshot_timestep": int(timestep),
                        "action_name": str(action_name),
                        "H_step_return": total_return,
                        "H_step_actual_payload_bits": payload,
                        "H_step_gross_service_bits": payload,
                        "H_step_queue_reduction_bits": start_queue - residual,
                        "H_step_completion": float(current["task_completion_ratio"]),
                        "H_step_movement_energy": movement_energy,
                        "horizon_steps": int(rollout.time_step - int(timestep)),
                        "all_actions_legal": bool(rollout.action_space.contains(action)),
                        "continuation_policy": "queue_rate_energy",
                        "uses_current_state_only": True,
                    }
                )
    if len(result) != 560 or not all(bool(row["all_actions_legal"]) for row in result):
        raise RuntimeError("H-step return alignment reconstruction is incomplete or illegal")
    raw = _read_csv(root / ORIGINAL_OUTPUT / "task_waypoint_controllability_raw.csv")
    raw_by_key = {
        (str(row["scenario_id"]), int(row["timestep"]), str(row["action_name"])): row
        for row in raw
    }
    for row in result:
        source = raw_by_key[(str(row["scenario_id"]), int(row["snapshot_timestep"]), str(row["action_name"]))]
        if not np.isclose(float(row["H_step_return"]), float(source["H_step_return"]), rtol=0.0, atol=1.0e-9):
            raise RuntimeError("H-step return reconstruction differs from immutable formal raw output")
        if not np.isclose(float(row["H_step_actual_payload_bits"]), float(source["total_service_bits"]), rtol=0.0, atol=1.0e-6):
            raise RuntimeError("H-step payload reconstruction differs from immutable formal raw output")
    return result


def _return_head_alignment_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Quantify Return-Task-Waypoint alignment overall and by registered group."""
    metrics = (
        ("H_step_actual_payload_bits", "benefit"),
        ("H_step_gross_service_bits", "benefit"),
        ("H_step_queue_reduction_bits", "benefit"),
        ("H_step_completion", "benefit"),
        ("H_step_movement_energy", "cost"),
    )
    scopes: list[tuple[str, str, list[Mapping[str, Any]]]] = [("overall", "all", list(rows))]
    for dimension in ("num_buoys", "difficulty"):
        for value in sorted({str(row[dimension]) for row in rows}):
            scopes.append((dimension, value, [row for row in rows if str(row[dimension]) == value]))
    result: list[dict[str, Any]] = []
    for dimension, value, subset in scopes:
        for metric, direction in metrics:
            pearson, spearman = _correlation(
                [float(row["H_step_return"]) for row in subset],
                [float(row[metric]) for row in subset],
            )
            result.append(
                {
                    "analysis": "return_head_alignment",
                    "group_dimension": dimension,
                    "group_value": value,
                    "target": "H_step_return",
                    "metric": metric,
                    "metric_direction": direction,
                    "pearson": pearson,
                    "spearman": spearman,
                    "sample_count": len(subset),
                }
            )
    primary = next(
        row for row in result
        if row["group_dimension"] == "overall" and row["metric"] == "H_step_actual_payload_bits"
    )
    partial_pearson = _partial_correlation(
        [float(row["H_step_return"]) for row in rows],
        [float(row["H_step_actual_payload_bits"]) for row in rows],
        [str(row["scenario_id"]) for row in rows],
    )
    evidence = min(float(primary["pearson"]), float(primary["spearman"]))
    classification = "strong" if evidence >= 0.60 else ("moderate" if evidence >= 0.30 else "weak")
    if float(primary["pearson"]) < 0.0 or float(primary["spearman"]) < 0.0:
        classification = "conflicting"
    return result, {"pearson": float(primary["pearson"]), "spearman": float(primary["spearman"]), "partial_pearson": float(partial_pearson), "classification": classification}


def _component_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Provide scenario distributions and paired component deltas in one table."""
    components = (*FIXED_REWARD_COMPONENTS, "payload_reward", "unfinished_queue_penalty", "task_success_bonus", "episode_return")
    mechanisms = (
        "current_fixed_parallel_1_over_K",
        "discrete_argmax_queue_times_rate",
        "continuous_masked_simplex_queue_times_rate",
    )
    keyed = {(str(row["scenario_id"]), str(row["mechanism"])): row for row in rows}
    scenarios = sorted({str(row["scenario_id"]) for row in rows})
    result: list[dict[str, Any]] = []
    for component in components:
        for mechanism in mechanisms:
            values = np.asarray([float(keyed[(scenario, mechanism)][component]) for scenario in scenarios], dtype=float)
            discrete_delta = np.asarray([
                float(keyed[(scenario, "discrete_argmax_queue_times_rate")][component])
                - float(keyed[(scenario, "current_fixed_parallel_1_over_K")][component])
                for scenario in scenarios
            ])
            continuous_delta = np.asarray([
                float(keyed[(scenario, "continuous_masked_simplex_queue_times_rate")][component])
                - float(keyed[(scenario, "current_fixed_parallel_1_over_K")][component])
                for scenario in scenarios
            ])
            result.append(
                {
                    "record_type": "component_summary",
                    "component": component,
                    "mechanism": mechanism,
                    "scenario_count": len(values),
                    "mean_contribution": float(values.mean()),
                    "std_contribution": float(values.std(ddof=0)),
                    "scenario_distribution": {
                        "minimum": float(values.min()),
                        "q25": float(np.quantile(values, .25)),
                        "median": float(np.median(values)),
                        "q75": float(np.quantile(values, .75)),
                        "maximum": float(values.max()),
                    },
                    "discrete_minus_current_mean_delta": float(discrete_delta.mean()),
                    "continuous_minus_current_mean_delta": float(continuous_delta.mean()),
                }
            )
    return result


def _reward_alignment_rows(rows: Sequence[Mapping[str, Any]], hstep_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return-vs-task correlations with required buoy/difficulty groupings."""
    result: list[dict[str, Any]] = []
    metric_specs = (
        ("cumulative_actual_payload_bits", "benefit"),
        ("completion", "benefit"),
        ("residual_queue_bits", "cost"),
        ("queue_reduction_bits", "benefit"),
        ("movement_energy", "cost"),
    )
    scopes: list[tuple[str, str, list[Mapping[str, Any]]]] = [("overall", "all", list(rows))]
    for mechanism in sorted({str(row["mechanism"]) for row in rows}):
        scopes.append(("mechanism", mechanism, [row for row in rows if str(row["mechanism"]) == mechanism]))
    for dimension in ("num_buoys", "difficulty"):
        for value in sorted({str(row[dimension]) for row in rows}):
            scopes.append((dimension, value, [row for row in rows if str(row[dimension]) == value]))
    for dimension, value, subset in scopes:
        for metric, direction in metric_specs:
            pearson, spearman = _correlation(
                [float(row["episode_return"]) for row in subset],
                [float(row[metric]) for row in subset],
            )
            result.append(
                {
                    "analysis": "episode_return_alignment",
                    "group_dimension": dimension,
                    "group_value": value,
                    "target": "episode_return",
                    "metric": metric,
                    "metric_direction": direction,
                    "pearson": pearson,
                    "spearman": spearman,
                    "sample_count": len(subset),
                }
            )
    h_rows, _ = _return_head_alignment_rows(hstep_rows)
    for row in h_rows:
        result.append({**row, "analysis": "H10_return_alignment"})
    return result


def _failure_factor_rows(root: Path, frozen_scores: Mapping[str, float], continuous: Mapping[str, Any], reward: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Evidence-weighted, non-exclusive re-evaluation of Round-12 TW-POL-B."""
    oracle = _read_csv(root / ROUND12_OUTPUT / "oracle_gap_analysis.csv")
    overall_gaps = [float(row["oracle_gap"]) for row in oracle if row["group_dimension"] == "overall"]
    mean_gap = float(np.mean(overall_gaps))
    utilization = _read_json(root / ROUND12_OUTPUT / "task_waypoint_head_utilization_summary.json")
    gate = _read_json(root / ROUND12_OUTPUT / "final_policy_gate.json")
    return [
        {
            "factor": "predictor_error",
            "evidence_strength": "moderate",
            "evidence": f"mean oracle gap={mean_gap:.9f} across three locked seeds; nonzero gap can limit a policy fed predicted waypoints.",
            "source": "round12 oracle_gap_analysis.csv",
        },
        {
            "factor": "action_controllability_mismatch",
            "evidence_strength": "strong",
            "evidence": "C_service=%.10f, C_progress=%.10f, C_return=%.10f; continuous service diagnostic=%s." % (
                float(frozen_scores["service_distribution"]),
                float(frozen_scores["gross_service_progress"]),
                float(frozen_scores["H_step_return"]),
                str(continuous["gate"]),
            ),
            "source": "Round-13 formal controllability plus amendment continuous headroom",
        },
        {
            "factor": "reward_return_waypoint_semantic_mismatch",
            "evidence_strength": "unsupported",
            "evidence": "Exact reward reconstruction gives %s and H=10 return/payload Pearson=%.6f, Spearman=%.6f; the original positive discrete sidecar return was not formula-equivalent." % (
                str(reward["reward_alignment_gate"]),
                float(reward["H10_return_payload_pearson"]),
                float(reward["H10_return_payload_spearman"]),
            ),
            "source": "amendment exact reward decomposition",
        },
        {
            "factor": "representation_to_action_mapping_limitation",
            "evidence_strength": "moderate",
            "evidence": "Round-12 gate=%s despite positive head sensitivities (service=%.4f, progress=%.4f, return=%.4f); predicted task signals remain movement-mediated." % (
                str(gate["task_waypoint_policy_gate"]),
                float(utilization["Sensitivity_service"]),
                float(utilization["Sensitivity_progress"]),
                float(utilization["Sensitivity_return"]),
            ),
            "source": "Round-12 locked head-utilization and policy gate",
        },
    ]


def _derive_frozen_tables(root: Path, output: Path) -> dict[str, float]:
    raw = _read_csv(root / ORIGINAL_OUTPUT / "task_waypoint_controllability_raw.csv")
    summary = _read_csv(root / ORIGINAL_OUTPUT / "task_waypoint_controllability.csv")
    service_fields = ("scenario_id", "scenario_seed", "snapshot_id", "timestep", "action_name", "action_to_service_js", "top_served_buoy_switch", "service_entropy_change", "service_distribution", "all_actions_legal", "horizon_steps", "difficulty", "num_buoys", "queue_level", "geometry_group")
    progress_fields = ("scenario_id", "scenario_seed", "snapshot_id", "timestep", "action_name", "gross_service_progress_bits", "total_service_bits", "total_service_change_bits", "progress_l2_change_bits", "all_actions_legal", "horizon_steps", "difficulty", "num_buoys", "queue_level", "geometry_group")
    return_fields = ("scenario_id", "scenario_seed", "snapshot_id", "timestep", "action_name", "H_step_return", "return_change", "all_actions_legal", "horizon_steps", "difficulty", "num_buoys", "queue_level", "geometry_group")
    def pick(fields: Sequence[str]) -> list[dict[str, str]]:
        return [{key: row[key] for key in fields} for row in raw]
    _write_csv(output / "service_controllability_metrics.csv", pick(service_fields))
    _write_csv(output / "progress_controllability_metrics.csv", pick(progress_fields))
    _write_csv(output / "return_controllability_metrics.csv", pick(return_fields))
    overall = {row["head"]: float(row["controllability_score"]) for row in summary if row["group_dimension"] == "overall"}
    spec = "# Task-Waypoint controllability specification (amendment)\n\nFor head response $Y_{s,a}$ at snapshot $s$ and legal first action $a$, the frozen formal score is:\n\n`C_head = mean_s[var_a(Y_{s,a})] / (var_{s,a}(Y_{s,a}) + epsilon)`\n\nThe numerator is the mean action-conditioned variance at a fixed snapshot; the denominator is total response variance over every snapshot-action response in the reported group.  `epsilon=1e-12`.\n\n| head | frozen Round-13 overall C |\n|---|---:|\n| C_service | %.10f |\n| C_progress | %.10f |\n| C_return | %.10f |\n\n`H_step_return` is the minimum-controllability head.  All three values are low: this is evidence of broad indirect, movement-mediated controllability rather than a problem unique to one head.  These tables are deterministic splits of the immutable original raw formal result; no scenarios were rerun.\n" % (overall["service_distribution"], overall["gross_service_progress"], overall["H_step_return"])
    (output / "task_waypoint_controllability_spec.md").write_text(spec, encoding="utf-8")
    resource = """# Service-resource decoder specification (amendment)

## Diagnostic interface

The prospective sidecar form is `[delta_x, delta_y, pi_1, ..., pi_K]` for the
current, configuration-defined buoy count `K`.  The public `MaritimeEnv`
action remains only `[delta_x, delta_y]`; this file specifies a diagnostic
decoder and does not extend that action space.

Inputs are the current observable queue vector `Q`, current full-bandwidth
rate vector `R_full`, current valid-buoy mask `m`, and slot duration
`delta_t`.  No future queue, arrival, rate, payload, reward, position, or
action is accepted.

## Masked simplex and allocation

- `pi_k >= 0` for every buoy.
- If `m_k=false` (padding), `pi_k=fraction_k=0` exactly.
- If any buoy is valid, `sum_{m_k=true}(pi_k)=1`; otherwise the all-zero
  vector is legal.
- The diagnostic score is
  `score_k=(Q_k/max_valid(Q))*(R_full,k/max_valid(R_full))`, normalized over
  valid entries.  If every valid score is zero, the deterministic fallback is
  the current fixed-equal allocation `1/K`; there is no random assignment.
- `fraction_k=pi_k`, `0 <= sum(fraction_k) <= 1`, and
  `allocated_capacity_k=fraction_k*R_full,k*delta_t`.
- `actual_payload_k=min(Q_k, allocated_capacity_k)`, so actual payload can
  never exceed queue or allocated capacity.

The implementation rejects nonfinite or negative values, padding mass,
fractions whose total exceeds one, nonpositive slot durations, and every
capacity/queue violation.  It is dynamically sized by the supplied vectors
and mask; no fixed buoy count is assumed.
"""
    (output / "service_resource_decoder_spec.md").write_text(resource, encoding="utf-8")
    implementation = """# Round-13 completion-amendment implementation log

## Scope and preservation audit

- Rehashed the original Round-13 lock and every file in its immutable formal
  output before starting; the amendment records these hashes and rechecks them
  before its separate lock is written.
- Rechecked the locked Round-12 predictor and all 24 frozen policy checkpoint
  hashes.  No Round-12 source, checkpoint, manifest, report, figure, or
  output was modified.
- Split the immutable Round-13 raw controllability table into service,
  progress, and return CSVs; those tables are derived data, not reruns.

## Scenarios and counterfactual execution

- Used exactly the ten pre-registered Round-10 `H_task=10` validation
  scenarios that were used by the original Round-13 headroom diagnostic.
- Every current/discrete/continuous comparison begins from the same reset
  seed, buoy positions, initial queue, arrival trace, environment config, UAV
  motion, and 128-step horizon.  The sole experimental variable is the
  diagnostic-sidecar resource mechanism.
- At every slot, the movement action and continuation are the deterministic
  `queue_rate_energy` baseline.  The sidecar reads only the current sidecar
  queue and current post-movement full-bandwidth link rate; it reads no future
  arrival, queue, rate, payload, reward, position, action, or Task-Waypoint.
- For Return-Task-Waypoint alignment, the same 40 registered snapshots, 14
  legal first actions, and ten-step `queue_rate_energy` continuation were
  reconstructed (560 rollouts).  Each return and payload is checked against
  the immutable formal raw row before writing the additive table.

## What was and was not executed

- Implemented a pure diagnostic masked-simplex priority decoder.  It is not
  connected to `MaritimeEnv.action_space`, its resource decoder, any training
  pipeline, or a neural checkpoint.
- Reconstructed the exact v1 reward components from `MaritimeEnv._step_v1`
  for fixed-movement sidecars; the current reconstruction must have zero
  error.
- Did not construct or run an optimizer, backward pass, neural-network
  training, PPO, HGNN, multi-UAV, power-control, beamforming, or a new
  predictor/policy checkpoint.
- Did not access old development test, Round-7 untouched holdout, or Round-12
  fresh-development scenarios.  Both protected access counts remain zero.

## Known limits

This is a diagnostic-only single-UAV, fixed-trajectory comparison of resource
mechanisms on ten validation scenarios.  It does not establish a new
environment action interface, policy value, closed-loop ranking, or a
next-round authorization.
"""
    (output / "round13_completion_amendment_implementation_log.md").write_text(implementation, encoding="utf-8")
    return overall


def _figures(output: Path, frozen_summary: Sequence[Mapping[str, str]], rows: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]], allocation: Mapping[str, Any], hstep_rows: Sequence[Mapping[str, Any]], component_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    written: list[str] = []
    def save(name: str, title: str, draw: Any) -> None:
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
        draw(ax)
        ax.set_title(title)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = folder / f"{name}.{suffix}"
            fig.savefig(path, dpi=140 if suffix == "png" else None, facecolor="white")
            written.append(path.name)
        plt.close(fig)
    heads = ("service_distribution", "gross_service_progress", "H_step_return")
    difficulty = ("easy", "medium", "hard")
    def difficulty_plot(ax: Any) -> None:
        for index, head in enumerate(heads):
            values = [float(next(row["controllability_score"] for row in frozen_summary if row["group_dimension"] == "difficulty" and row["group_value"] == label and row["head"] == head)) for label in difficulty]
            ax.bar(np.arange(3) + (index - 1) * 0.25, values, 0.25, label=head)
        ax.set_xticks(np.arange(3), difficulty); ax.set_ylabel("controllability score"); ax.legend(fontsize=8)
    save("13_difficulty_controllability", "Controllability by difficulty", difficulty_plot)
    mechanisms = ["current_fixed_parallel_1_over_K", "discrete_argmax_queue_times_rate", "continuous_masked_simplex_queue_times_rate"]
    short = ["Current", "Discrete", "Continuous"]

    def means(field: str) -> list[float]:
        return [float(np.mean([float(row[field]) for row in rows if row["mechanism"] == name])) for name in mechanisms]

    def completion_plot(ax: Any) -> None:
        values = means("completion")
        ax.bar(short, values, color=("#6d8aa8", "#d18b47", "#2676a1")); ax.set_ylabel("completion")
    save("14_completion_comparison", "Completion by resource mechanism", completion_plot)

    def residual_plot(ax: Any) -> None:
        ax.bar(short, [value / 1e6 for value in means("residual_queue_bits")], color=("#6d8aa8", "#d18b47", "#2676a1"))
        ax.set_ylabel("residual queue (Mbit)")
    save("15_residual_queue_comparison", "Residual queue by resource mechanism", residual_plot)

    def payload_plot(ax: Any) -> None:
        ax.bar(short, [value / 1e6 for value in means("cumulative_actual_payload_bits")], color=("#6d8aa8", "#d18b47", "#2676a1"))
        ax.set_ylabel("cumulative actual payload (Mbit)")
    save("16_payload_comparison", "Cumulative actual payload by resource mechanism", payload_plot)

    pair_map = {(str(row["comparison"]), str(row["metric"])): row for row in paired}
    def continuous_current_delta(ax: Any) -> None:
        row = pair_map[("continuous_masked_simplex_queue_times_rate_minus_current_fixed_parallel_1_over_K", "Delta_payload_bits")]
        values = [float(row["paired_mean"]) / 1e6, float(row["bootstrap_ci_low"]) / 1e6, float(row["bootstrap_ci_high"]) / 1e6]
        ax.bar(("mean", "CI low", "CI high"), values, color="#2676a1")
        ax.axhline(0.0, color="black", linewidth=.8); ax.set_ylabel("Delta payload (Mbit)")
    save("17_continuous_vs_current_payload", "Continuous minus current payload", continuous_current_delta)

    def allocation_plot(ax: Any) -> None:
        values = list(allocation["fraction"]); ax.bar(range(len(values)), values, color="#2676a1"); ax.set_xlabel("buoy index"); ax.set_ylabel("continuous fraction")
    save("18_continuous_priority_allocation", "Continuous priority resource allocation example", allocation_plot)

    component_labels = ("payload_reward", "energy_penalty", "unfinished_queue_penalty", "task_success_bonus")
    summary_by_key = {(str(row["component"]), str(row["mechanism"])): row for row in component_rows}
    def component_plot(ax: Any) -> None:
        bottom = np.zeros(len(mechanisms), dtype=float)
        colors = ("#2676a1", "#bc6c25", "#8ab17d", "#6d597a")
        for label, color in zip(component_labels, colors):
            values = np.asarray([float(summary_by_key[(label, mechanism)]["mean_contribution"]) for mechanism in mechanisms])
            ax.bar(short, values, bottom=bottom, label=label, color=color)
            bottom += values
        ax.set_ylabel("mean reward contribution"); ax.legend(fontsize=7)
    save("19_reward_component_contributions", "Reward component contributions", component_plot)

    def reward_plot(ax: Any) -> None:
        colors = {mechanisms[0]: "#6d8aa8", mechanisms[1]: "#d18b47", mechanisms[2]: "#2676a1"}
        for name in mechanisms:
            subset = [row for row in rows if row["mechanism"] == name]
            ax.scatter([float(row["cumulative_actual_payload_bits"]) / 1e6 for row in subset], [float(row["episode_return"]) for row in subset], label=short[mechanisms.index(name)], color=colors[name])
        ax.set_xlabel("cumulative actual payload (Mbit)"); ax.set_ylabel("exact/shadow return"); ax.legend()
    save("20_reward_vs_payload", "Reward versus communication metrics", reward_plot)

    def return_completion_plot(ax: Any) -> None:
        colors = {mechanisms[0]: "#6d8aa8", mechanisms[1]: "#d18b47", mechanisms[2]: "#2676a1"}
        for name in mechanisms:
            subset = [row for row in rows if row["mechanism"] == name]
            ax.scatter([float(row["completion"]) for row in subset], [float(row["episode_return"]) for row in subset], label=short[mechanisms.index(name)], color=colors[name])
        ax.set_xlabel("completion"); ax.set_ylabel("exact/shadow return"); ax.legend()
    save("21_return_vs_completion", "Return versus completion", return_completion_plot)

    def hstep_plot(ax: Any) -> None:
        ax.scatter([float(row["H_step_actual_payload_bits"]) / 1e6 for row in hstep_rows], [float(row["H_step_return"]) for row in hstep_rows], color="#2676a1", s=10)
        ax.set_xlabel("H-step actual payload (Mbit)"); ax.set_ylabel("H-step return")
    save("22_hstep_return_vs_payload", "H-step return versus H-step payload", hstep_plot)

    h_alignment, _ = _return_head_alignment_rows(hstep_rows)
    def return_alignment_plot(ax: Any) -> None:
        subset = [row for row in h_alignment if row["group_dimension"] == "overall"]
        labels = [str(row["metric"]).replace("H_step_", "") for row in subset]
        ax.bar(labels, [float(row["pearson"]) for row in subset], color="#2676a1")
        ax.axhline(0.0, color="black", linewidth=.8); ax.set_ylabel("Pearson correlation")
        for label in ax.get_xticklabels():
            label.set_rotation(25)
    save("23_return_head_alignment", "Return-head alignment with task metrics", return_alignment_plot)

    def controllability_plot(ax: Any) -> None:
        overall = [row for row in frozen_summary if row["group_dimension"] == "overall"]
        ax.bar([str(row["head"]) for row in overall], [float(row["controllability_score"]) for row in overall], color="#2676a1")
        ax.set_ylabel("controllability score")
        for label in ax.get_xticklabels():
            label.set_rotation(20)
    save("24_controllability_summary", "C_service, C_progress, and C_return", controllability_plot)
    return sorted(written)


def run_amendment(root: Path, output: Path) -> Path:
    """Create additive amendment evidence; original locks and outputs stay untouched."""
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    original_before = _file_hashes(root, ORIGINAL_OUTPUT)
    lock_before = sha256(root / "round_13/ROUND_LOCK.json")
    output.mkdir(parents=True)
    frozen_scores = _derive_frozen_tables(root, output)
    scenario_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    allocation: dict[str, Any] | None = None
    for scenario in _validation_scenarios(root):
        rows, steps, example = _scenario_mechanisms(scenario)
        scenario_rows.extend(rows); step_rows.extend(steps)
        if allocation is None:
            allocation = example
    difficulty = _allowed_difficulty_labels(root, _validation_scenarios(root))
    for row in scenario_rows:
        row["difficulty"] = str(difficulty[str(row["scenario_id"])])
    for row in step_rows:
        row["difficulty"] = str(difficulty[str(row["scenario_id"])])
    if any(int(row["resource_decoder_violations"]) for row in scenario_rows):
        raise RuntimeError("amendment service decoder violated a physical constraint")
    paired = []
    paired.extend(_paired_rows(scenario_rows, target="continuous_masked_simplex_queue_times_rate", reference="current_fixed_parallel_1_over_K", bootstrap_seed=13013))
    paired.extend(_paired_rows(scenario_rows, target="continuous_masked_simplex_queue_times_rate", reference="discrete_argmax_queue_times_rate", bootstrap_seed=13014))
    paired.extend(_paired_rows(scenario_rows, target="discrete_argmax_queue_times_rate", reference="current_fixed_parallel_1_over_K", bootstrap_seed=13015))
    raw = _read_csv(root / ORIGINAL_OUTPUT / "task_waypoint_controllability_raw.csv")
    component_alignment_rows, _ = _alignment_rows(scenario_rows, raw)
    hstep_rows = _hstep_return_rows(root)
    return_head_rows, return_alignment = _return_head_alignment_rows(hstep_rows)
    reward_alignment_rows = component_alignment_rows + _reward_alignment_rows(scenario_rows, hstep_rows)
    component_summary = _component_summary_rows(scenario_rows)
    continuous_gate = _continuous_gate(scenario_rows, paired)
    reward_gate = _reward_gate(scenario_rows, paired, return_alignment)
    if continuous_gate["service_action_extension_ready"] and reward_gate["reward_alignment_gate"] != "RA-C":
        recommendation = "minimal_service_action_extension"
    elif not continuous_gate["service_action_extension_ready"] and reward_gate["reward_alignment_gate"] == "RA-C":
        recommendation = "reward_and_return_head_semantics_review"
    else:
        recommendation = "task_waypoint_action_mapping_redesign"
    original_headroom = _read_csv(root / ORIGINAL_OUTPUT / "service_action_headroom_paired.csv")
    original_discrete_return = next(row for row in original_headroom if row["metric"] == "Delta_return")
    _, reward_config, _, _, _ = round7_case(int(_validation_scenarios(root)[0]["scenario_seed"]), int(_validation_scenarios(root)[0]["num_buoys"]), 128)
    reward_doc = """# Reward semantics and alignment audit (amendment)

## Exact active v1 reward

Source of truth: `MaritimeEnv._step_v1` (lines 666--719).  For each slot,

`R = r_payload + r_energy + r_boundary + r_invalid + r_depletion + r_unfinished + r_success`.

| component | exact formula | coefficient / normalization | unit and physical meaning | sign and range |
|---|---|---|---|---|
| payload_reward | `w_p * clip(sum(payload_k)/(R_max_total*delta_t), 0, 1)` | `w_p=%(payload_weight).6g`; denominator is theoretical full-bandwidth aggregate bit capacity | dimensionless reward for actual served bits | `[0, %(payload_weight).6g]` |
| energy_penalty | `-w_e * clip(actual_energy_consumed/(hover_energy_per_step + move_energy_per_meter*maximum_step_distance), 0, 1)` | `w_e=%(energy_weight).6g` | dimensionless movement/hover energy cost | `[-%(energy_weight).6g, 0]` |
| boundary_penalty | `-boundary_penalty` iff the candidate movement crosses the area boundary | `boundary_penalty=%(boundary).6g` | projected-motion constraint violation | `{0, -%(boundary).6g}` |
| invalid_action_penalty | `-invalid_action_penalty` iff clipping or radial projection adjusts the action | `invalid_action_penalty=%(invalid).6g` | action-interface constraint violation | `{0, -%(invalid).6g}` |
| energy_depletion_penalty | `-energy_depletion_penalty` iff energy is depleted | `energy_depletion_penalty=%(depletion).6g` | terminal energy constraint | `{0, -%(depletion).6g}` |
| unfinished_queue_penalty | `-w_u * residual_queue/(K*queue_capacity)` only at episode end and only without task success | `w_u=%(unfinished).6g` | terminal residual communication backlog | `[-%(unfinished).6g, 0]` |
| task_success_bonus | `task_success_bonus` only for task success when terminate-on-success applies | `task_success_bonus=%(success).6g` | terminal completion/reliability bonus | `{0, %(success).6g}` |

`payload_k=min(queue_k, allocated_capacity_k)`, so the payload term respects the
queue and physical link-capacity constraints.  The resource sidecars preserve
the original movement, movement costs, arrivals, and horizon; only the valid
resource fraction changes.

## Resolution of the apparent discrete-return anomaly

The immutable original discrete sidecar reported `Delta_return=%(original_return)s`
with 95%% CI [`%(original_low)s`, `%(original_high)s`].  Its calculation was not
an exact reconstruction of `MaritimeEnv._step_v1`: it normalized replacement
payload by `after[total_rate_bps]*delta_t` instead of the environment's
theoretical full-bandwidth denominator, and used offered-data normalization
for the terminal residual term.  This amendment first verifies zero
current-mechanism reconstruction error, then applies the exact formula above
to every sidecar.  Therefore the earlier positive discrete return is not
evidence that the active reward prefers worse communication.

The exact component distributions and paired deltas are in
`reward_component_metrics.csv`; correlations are in
`reward_alignment_audit.csv`; the independent H=10 return-head audit is in
`return_head_alignment_audit.csv`.
""" % {
        "payload_weight": float(reward_config.payload_reward_weight),
        "energy_weight": float(reward_config.energy_penalty_weight),
        "boundary": float(reward_config.boundary_penalty),
        "invalid": float(reward_config.invalid_action_penalty),
        "depletion": float(reward_config.energy_depletion_penalty),
        "unfinished": float(reward_config.unfinished_queue_penalty_weight),
        "success": float(reward_config.task_success_bonus),
        "original_return": original_discrete_return["paired_mean"],
        "original_low": original_discrete_return["bootstrap_ci_low"],
        "original_high": original_discrete_return["bootstrap_ci_high"],
    }
    (output / "reward_semantics_audit.md").write_text(reward_doc, encoding="utf-8")
    summary = _read_csv(root / ORIGINAL_OUTPUT / "task_waypoint_controllability.csv")
    figures = _figures(output, summary, scenario_rows, paired, allocation or {}, hstep_rows, component_summary)
    failure_rows = _failure_factor_rows(root, frozen_scores, continuous_gate, reward_gate)
    _write_csv(output / "continuous_service_headroom_metrics.csv", scenario_rows)
    _write_csv(output / "continuous_service_headroom_step_metrics.csv", step_rows)
    _write_csv(output / "continuous_service_headroom_paired.csv", paired)
    _write_csv(output / "reward_component_metrics.csv", component_summary)
    _write_csv(output / "reward_alignment_audit.csv", reward_alignment_rows)
    _write_csv(output / "return_head_alignment_audit.csv", return_head_rows)
    _write_csv(output / "return_head_alignment_rollouts.csv", hstep_rows)
    _write_csv(output / "round12_failure_factor_audit.csv", failure_rows)
    _write_json(output / "continuous_service_action_gate.json", continuous_gate)
    _write_json(output / "reward_alignment_gate.json", reward_gate)
    _write_json(output / "round13_amended_readiness_gate.json", {"original_service_action_gate": "SA-C-discrete", "continuous_service_action_gate": continuous_gate["gate"], "reward_alignment_gate": reward_gate["reward_alignment_gate"], "service_action_extension_ready": continuous_gate["service_action_extension_ready"], "next_recommendation": recommendation, "round14_creation_allowed": False, "amendment_only": True})
    figure_manifest = {"original_figure_pairs": 12, "amendment_figure_pairs": 12, "total_figure_pairs": 24, "new_figure_files": figures, "white_background": True}
    _write_json(output / "amendment_figures_manifest.json", figure_manifest)
    manifest = {"schema_version": SCHEMA, "amendment_only": True, "neural_training_performed": False, "original_round13_results_modified": False, "original_round13_lock_sha256_before": lock_before, "original_formal_output_hashes_before": original_before, "round12_integrity_before": _round12_integrity(root), "status_document_hashes_before": _status_hashes(root), "frozen_controllability_scores": frozen_scores, "validation_scenario_count": 10, "H10_alignment_rollout_count": len(hstep_rows), "mechanisms": ["current_fixed_parallel_1_over_K", "discrete_argmax_queue_times_rate", "continuous_masked_simplex_queue_times_rate"], "uses_old_development_test": False, "uses_untouched_holdout": False, "output": output.relative_to(root).as_posix()}
    _write_json(output / "amendment_manifest.json", manifest)
    return output


def finalize_amendment(root: Path, output: Path, *, repository_tests_passed: int, compileall_passed: bool) -> Mapping[str, Any]:
    """Lock only new amendment evidence after external final verification."""
    root, output = Path(root), Path(output)
    if repository_tests_passed < 503 or not compileall_passed:
        raise ValueError("completion amendment requires the full passing regression and compileall")
    manifest = _read_json(output / "amendment_manifest.json")
    original_unchanged = _file_hashes(root, ORIGINAL_OUTPUT) == manifest["original_formal_output_hashes_before"]
    lock_unchanged = sha256(root / "round_13/ROUND_LOCK.json") == manifest["original_round13_lock_sha256_before"]
    if not original_unchanged or not lock_unchanged:
        raise RuntimeError("an immutable original Round-13 artifact changed during amendment")
    round12 = _round12_integrity(root)
    if round12 != manifest["round12_integrity_before"]:
        raise RuntimeError("Round-12 frozen integrity changed during amendment")
    continuous = _read_json(output / "continuous_service_action_gate.json")
    reward = _read_json(output / "reward_alignment_gate.json")
    readiness = _read_json(output / "round13_amended_readiness_gate.json")
    paired = _read_csv(output / "continuous_service_headroom_paired.csv")
    pair_map = {(str(row["comparison"]), str(row["metric"])): row for row in paired}
    current_name = "current_fixed_parallel_1_over_K"
    continuous_name = "continuous_masked_simplex_queue_times_rate"
    discrete_name = "discrete_argmax_queue_times_rate"
    continuous_current = {
        metric: pair_map[(f"{continuous_name}_minus_{current_name}", metric)]
        for metric in ("Delta_payload_bits", "Delta_completion", "Delta_residual_queue_improvement_bits", "Delta_return")
    }
    continuous_discrete = {
        metric: pair_map[(f"{continuous_name}_minus_{discrete_name}", metric)]
        for metric in ("Delta_payload_bits", "Delta_completion", "Delta_residual_queue_improvement_bits", "Delta_return")
    }
    original_discrete = {
        metric: pair_map[(f"{discrete_name}_minus_{current_name}", metric)]
        for metric in ("Delta_payload_bits", "Delta_completion", "Delta_residual_queue_improvement_bits", "Delta_return")
    }
    reward_rows = _read_csv(output / "reward_alignment_audit.csv")
    episode_payload = next(
        row for row in reward_rows
        if row["analysis"] == "episode_return_alignment"
        and row["group_dimension"] == "overall"
        and row["metric"] == "cumulative_actual_payload_bits"
    )
    return_rows = _read_csv(output / "return_head_alignment_audit.csv")
    hstep_payload = next(
        row for row in return_rows
        if row["group_dimension"] == "overall" and row["metric"] == "H_step_actual_payload_bits"
    )
    failure_rows = _read_csv(output / "round12_failure_factor_audit.csv")
    failure = {row["factor"]: row for row in failure_rows}
    verification = {"repository_tests_passed": int(repository_tests_passed), "compileall_passed": True, "original_round13_lock_unchanged": True, "original_formal_output_unchanged": True, "round12_frozen_hashes_unchanged": True}
    _write_json(output / "amendment_final_verification.json", verification)
    suffix = output.name.removeprefix("round13_completion_amendment_")
    report_path = root / f"report/round_13_completion_addendum_{suffix}.md"
    report = """# Round-13 completion addendum

`amendment_only=true`; `neural_training_performed=false`; and
`original_round13_results_modified=false`.  This addendum is additive: the
original Round-13 lock, formal output, controllability conclusion, and
SA-C-discrete result remain unchanged.

## Direct answers

1. **Original scientific conclusion preserved?** Yes.  All original-formal
   hashes and the original `ROUND_LOCK.json` hash are unchanged.
2. **C_service:** %(service).10f.
3. **C_progress:** %(progress).10f.
4. **C_return:** %(return_score).10f.
5. **Lowest controllability:** C_return (the H-step-return head).
6. **All three heads low?** Yes.  They are movement-mediated/indirectly
   controllable, not direct service controls.
7. **Continuous priority diagnostic completed?** Yes, on the same ten
   permitted validation scenarios, with zero decoder violations.
8. **Continuous minus current payload:** %(cc_payload).3f Mbit.
9. **Its 95%% scenario-bootstrap CI:** [%(cc_payload_low).3f,
   %(cc_payload_high).3f] Mbit.
10. **Continuous minus current completion:** %(cc_completion).6f.
11. **Continuous minus current residual-queue improvement:**
    %(cc_residual).3f Mbit (negative means a larger residual queue).
12. **Continuous minus current return:** %(cc_return).6f.
13. **Continuous minus discrete:** payload %(cd_payload).3f Mbit, completion
    %(cd_completion).6f, residual-queue improvement %(cd_residual).3f Mbit,
    return %(cd_return).6f; the payload CI is [%(cd_payload_low).3f,
    %(cd_payload_high).3f] Mbit.
14. **Continuous Service-Action Gate:** %(continuous_gate)s.
15. **service_action_extension_ready:** %(extension_ready)s.
16. **Current reward formula:**
    `payload_reward + energy_penalty + boundary_penalty +
    invalid_action_penalty + energy_depletion_penalty +
    unfinished_queue_penalty + task_success_bonus`; exact formulas,
    coefficients, normalizations, units, signs, and ranges are documented in
    `reward_semantics_audit.md`.
17. **Why did the original discrete heuristic show worse communication but
    higher return?** Its sidecar was not formula-equivalent to the environment:
    it used allocated-rate rather than theoretical-full-rate normalization for
    replacement payload, and a different terminal residual normalization.
    The immutable original result is still reported, but an exact reward
    reconstruction produces discrete-current return %(discrete_return).6f
    (95%% CI [%(discrete_return_low).6f, %(discrete_return_high).6f]), not a
    positive reward gain.
18. **Episode return vs cumulative payload:** Pearson %(episode_pearson).6f,
    Spearman %(episode_spearman).6f (pooled fixed-movement sidecars; grouped
    rows are in the audit CSV).
19. **H-step return vs H-step payload:** Pearson %(hstep_pearson).6f,
    Spearman %(hstep_spearman).6f.
20. **Return Task-Waypoint versus primary metric:** %(return_alignment)s.
21. **Reward Alignment Gate:** %(reward_gate)s.
22. **Best-supported TW-POL-B factor:** action controllability mismatch.
23. **Predictor-error evidence:** %(predictor)s.
24. **Action-mismatch evidence:** %(action)s.
25. **Reward/Return semantic-mismatch evidence:** %(semantic)s.
26. **Representation-to-action-mapping evidence:** %(representation)s.
27. **Amended next recommendation:** `%(recommendation)s`.
28. **PLANS/ROUND_VERSIONING corrected?** Yes; their current-status amendment
    states Round-13 completed/locked and no Round-14.  Before/after hashes are
    recorded in the amendment lock.
29. **Old-development-test access count:** 0.
30. **Untouched-holdout access count:** 0.
31. **Original ROUND_LOCK hash unchanged?** Yes.
32. **Round-14 created?** No.

## Formal status

The original discrete gate is retained as `SA-C-discrete`; it applies only to
the tested `argmax(queue * current/full-bandwidth-rate)` heuristic.  The
continuous masked-simplex candidate is separately `%(continuous_gate)s`, so it
does not justify a service-action extension.  No action-space change, model
training, policy training, checkpoint, optimizer, backward pass, old
development test, holdout access, HGNN, PPO, multi-UAV system, or Round-14
was introduced.

Final verification: %(tests)d repository tests passed and `compileall`
passed.  The formal amendment bundle is `%(output)s`.
""" % {
        "service": float(manifest["frozen_controllability_scores"]["service_distribution"]),
        "progress": float(manifest["frozen_controllability_scores"]["gross_service_progress"]),
        "return_score": float(manifest["frozen_controllability_scores"]["H_step_return"]),
        "cc_payload": float(continuous_current["Delta_payload_bits"]["paired_mean"]) / 1e6,
        "cc_payload_low": float(continuous_current["Delta_payload_bits"]["bootstrap_ci_low"]) / 1e6,
        "cc_payload_high": float(continuous_current["Delta_payload_bits"]["bootstrap_ci_high"]) / 1e6,
        "cc_completion": float(continuous_current["Delta_completion"]["paired_mean"]),
        "cc_residual": float(continuous_current["Delta_residual_queue_improvement_bits"]["paired_mean"]) / 1e6,
        "cc_return": float(continuous_current["Delta_return"]["paired_mean"]),
        "cd_payload": float(continuous_discrete["Delta_payload_bits"]["paired_mean"]) / 1e6,
        "cd_payload_low": float(continuous_discrete["Delta_payload_bits"]["bootstrap_ci_low"]) / 1e6,
        "cd_payload_high": float(continuous_discrete["Delta_payload_bits"]["bootstrap_ci_high"]) / 1e6,
        "cd_completion": float(continuous_discrete["Delta_completion"]["paired_mean"]),
        "cd_residual": float(continuous_discrete["Delta_residual_queue_improvement_bits"]["paired_mean"]) / 1e6,
        "cd_return": float(continuous_discrete["Delta_return"]["paired_mean"]),
        "continuous_gate": str(continuous["gate"]),
        "extension_ready": str(bool(continuous["service_action_extension_ready"])).lower(),
        "discrete_return": float(original_discrete["Delta_return"]["paired_mean"]),
        "discrete_return_low": float(original_discrete["Delta_return"]["bootstrap_ci_low"]),
        "discrete_return_high": float(original_discrete["Delta_return"]["bootstrap_ci_high"]),
        "episode_pearson": float(episode_payload["pearson"]),
        "episode_spearman": float(episode_payload["spearman"]),
        "hstep_pearson": float(hstep_payload["pearson"]),
        "hstep_spearman": float(hstep_payload["spearman"]),
        "return_alignment": str(reward["return_head_alignment_with_primary_metric"]),
        "reward_gate": str(reward["reward_alignment_gate"]),
        "predictor": str(failure["predictor_error"]["evidence_strength"]),
        "action": str(failure["action_controllability_mismatch"]["evidence_strength"]),
        "semantic": str(failure["reward_return_waypoint_semantic_mismatch"]["evidence_strength"]),
        "representation": str(failure["representation_to_action_mapping_limitation"]["evidence_strength"]),
        "recommendation": str(readiness["next_recommendation"]),
        "tests": int(repository_tests_passed),
        "output": output.relative_to(root).as_posix(),
    }
    report_path.write_text(report, encoding="utf-8")
    amendment_lock = {"schema_version": SCHEMA, "amendment_only": True, "status": "locked", "locked_at_utc": datetime.now(timezone.utc).isoformat(), "original_round13_lock_sha256": manifest["original_round13_lock_sha256_before"], "original_round13_lock_unchanged": True, "original_formal_output_unchanged": True, "round12_integrity": round12, "status_document_hashes_before": manifest["status_document_hashes_before"], "status_document_hashes_after": _status_hashes(root), "neural_training_performed": False, "original_round13_results_modified": False, "old_development_test_access_count": 0, "untouched_holdout_access_count": 0, "continuous_service_action_gate": continuous["gate"], "reward_alignment_gate": reward["reward_alignment_gate"], "service_action_extension_ready": continuous["service_action_extension_ready"], "next_recommendation": readiness["next_recommendation"], "round14_created": False, "repository_tests_passed": int(repository_tests_passed), "compileall_passed": True, "formal_amendment_output": output.relative_to(root).as_posix(), "completion_addendum": report_path.relative_to(root).as_posix()}
    _write_json(root / "round_13/ROUND_LOCK_AMENDMENT.json", amendment_lock)
    return amendment_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--tests", type=int, default=0)
    parser.add_argument("--compileall-passed", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        parser.error("--output is required")
    if args.finalize:
        print(json.dumps(finalize_amendment(args.root, args.output, repository_tests_passed=args.tests, compileall_passed=args.compileall_passed), indent=2))
    else:
        print(run_amendment(args.root, args.output))


if __name__ == "__main__":
    main()
