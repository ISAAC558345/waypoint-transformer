"""Materialize and analyse the deterministic Round-15 planning audit.

This module intentionally creates a new diagnostic collection. It never opens
an older development split, a holdout split, or a learned checkpoint.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from wt.round12.contracts import PREDICTOR_SHA256, sha256

from .planner import (
    EPSILON,
    _causal_score,
    _oracle_score,
    _policy_action as _baseline_policy_action,
    finite_lattice_plan,
    myopic_one_step_greedy,
)
from .scenarios import PlanningScenario, materialize_scenario, planning_scenarios, scenario_manifest_rows


SCHEMA = "round15_planning_headroom_audit_v1"
PROTOCOL_PATH = Path("round_15/configs/planning_protocol.json")
POLICY_IDS = (
    "nearest_buoy", "max_queue", "queue_rate_energy", "myopic_one_step_greedy",
    "causal_H1", "causal_H5", "causal_H10", "causal_H20",
    "oracle_H1", "oracle_H5", "oracle_H10", "oracle_H20",
)
PRIMARY_METRICS = (
    "cumulative_actual_payload_bits", "final_residual_queue_bits", "episode_return",
    "cumulative_distance_m", "cumulative_energy_consumption", "completion_step",
)


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
            raise ValueError("Round-15 output may not contain NaN or Inf")
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Round-15 output may not contain NaN or Inf")
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty Round-15 CSV: {path.name}")
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(_plain(row[key]), ensure_ascii=False, sort_keys=True)
                if isinstance(row.get(key), (dict, list, tuple, np.ndarray)) else _plain(row.get(key, ""))
                for key in fields
            })


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _protocol(root: Path) -> dict[str, Any]:
    protocol = _read_json(root / PROTOCOL_PATH)
    if protocol.get("schema_version") != "round15_planning_protocol_v1":
        raise RuntimeError("Round-15 protocol schema mismatch")
    if protocol.get("horizons") != [1, 5, 10, 20]:
        raise RuntimeError("Round-15 horizons differ from frozen protocol")
    if protocol.get("training_performed") is not False:
        raise RuntimeError("Round-15 protocol cannot enable training")
    return protocol


def preflight(root: Path) -> dict[str, Any]:
    """Assert the fixed prior-round and disclosure preconditions once."""
    root = Path(root)
    current = (root / "CURRENT_ROUND.txt").read_text(encoding="utf-8")
    if not current.startswith("round_15\n") or "status=in_progress" not in current:
        raise RuntimeError("Round-15 is not active")
    if (root / "round_16").exists():
        raise RuntimeError("Round-16 must not exist")
    manifest = _read_json(root / "round_15/round_manifest.json")
    for key in ("training_performed", "planning_headroom_audit_performed", "hgnn_training_performed", "ppo_training_performed"):
        if manifest.get(key) is not False:
            raise RuntimeError(f"Round-15 initial flag must be false: {key}")
    if manifest.get("untouched_holdout_access_count") != 1 or manifest.get("untouched_holdout_accessed") is not True:
        raise RuntimeError("Round-15 must retain its user-authorized metadata-only-access disclosure")
    lock = _read_json(root / "round_14/ROUND_LOCK.json")
    if lock.get("status") != "locked" or not lock.get("acceptance_complete"):
        raise RuntimeError("Round-14 is not locked")
    output = root / str(lock["formal_output"])
    mismatches = [name for name, digest in lock["output_hashes"].items() if sha256(output / name) != digest]
    if mismatches:
        raise RuntimeError(f"Round-14 output changed: {mismatches}")
    return {
        "round14_lock_sha256": sha256(root / "round_14/ROUND_LOCK.json"),
        "round14_outputs_unchanged": True,
        "round14_test_count": int(lock["full_regression_tests_pass_count"]),
        "round14_compileall_passed": bool(lock["compileall_passed"]),
        "round11_predictor_sha256": PREDICTOR_SHA256,
        "old_development_test_accessed": False,
        "untouched_holdout_access_count": 1,
        "untouched_holdout_access_note": manifest["untouched_holdout_access_note"],
        "round16_created": False,
    }


def _policy_descriptor(policy_id: str) -> tuple[str, int | None]:
    if policy_id in ("nearest_buoy", "max_queue", "queue_rate_energy", "myopic_one_step_greedy"):
        return "myopic", None
    mode, horizon = policy_id.split("_H")
    return mode, int(horizon)


def dispatch_policy_action(env: Any, observation: np.ndarray, info: Mapping[str, Any], policy_id: str):
    """Return the real-time action and optional planner decision."""
    if policy_id in ("nearest_buoy", "max_queue", "queue_rate_energy"):
        return _baseline_policy_action(policy_id, observation, info), None
    if policy_id == "myopic_one_step_greedy":
        return myopic_one_step_greedy(env, observation, info), None
    mode, horizon = _policy_descriptor(policy_id)
    decision = finite_lattice_plan(env, observation, info, horizon=int(horizon), mode=mode)
    return decision.action, decision


def _json_vector(value: Any) -> str:
    return json.dumps(_plain(np.asarray(value, dtype=float)), separators=(",", ":"))


def _close_buoy(position: np.ndarray, buoy_positions: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(np.asarray(buoy_positions) - np.asarray(position), axis=1)))


def _queue_tie_proxy(raw_queue_bits: np.ndarray) -> bool:
    ordered = np.sort(np.asarray(raw_queue_bits, dtype=float))
    return bool(len(ordered) >= 2 and ordered[-1] > EPSILON and (ordered[-1] - ordered[-2]) / ordered[-1] <= 0.10)


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    if left_norm <= EPSILON or right_norm <= EPSILON:
        return 0.0 if left_norm <= EPSILON and right_norm <= EPSILON else 180.0
    cosine = float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _run_episode(scenario: PlanningScenario, policy_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run an independent real environment episode from the frozen new scenario."""
    env, observation, info = materialize_scenario(scenario)
    records: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    delayed: list[dict[str, Any]] = []
    payload_total = return_total = 0.0
    boundary_count = 0
    residual_error = 0.0
    completion_step: int | None = None
    for step in range(env.config.episode_horizon):
        action, decision = dispatch_policy_action(env, observation, info, policy_id)
        if decision is not None:
            runtimes.append({
                "scenario_id": scenario.scenario_id, "policy_id": policy_id, "mode": decision.mode,
                "horizon": decision.horizon, "step": step, "runtime_seconds": decision.runtime_seconds,
                "nodes_expanded": decision.nodes_expanded,
                "oracle_exogenous_knowledge": decision.oracle_exogenous_knowledge,
                "deployable_diagnostic": decision.deployable_diagnostic,
                "selected_score_payload_bits": decision.score_payload_bits,
                "candidate_scores": dict(decision.candidate_scores),
            })
        if policy_id == "causal_H20":
            h1_action, _ = dispatch_policy_action(env, observation, info, "causal_H1")
            remaining = min(20, env.config.episode_horizon - step)
            immediate_long, _ = _causal_score(env, info, action, 1)
            immediate_short, _ = _causal_score(env, info, h1_action, 1)
            future_long, _ = _oracle_score(env, observation, info, action, remaining)
            future_short, _ = _oracle_score(env, observation, info, h1_action, remaining)
            delayed.append({
                "scenario_id": scenario.scenario_id, "difficulty": scenario.difficulty,
                "num_buoys": scenario.num_buoys, "queue_pressure": scenario.queue_pressure,
                "geometry": scenario.geometry, "step": step, "remaining_horizon": remaining,
                "causal_h20_action": _json_vector(action), "causal_h1_action": _json_vector(h1_action),
                "actions_differ": bool(np.linalg.norm(action - h1_action) > EPSILON),
                "action_angle_degrees": _angle_degrees(action, h1_action),
                "immediate_h20_payload_bits": immediate_long,
                "immediate_h1_payload_bits": immediate_short,
                "oracle_future_h20_payload_bits": future_long,
                "oracle_future_h1_payload_bits": future_short,
                "immediate_sacrifice_bits": max(0.0, immediate_short - immediate_long),
                "future_gain_bits": future_long - future_short,
                "delayed_benefit": bool(immediate_long < immediate_short - EPSILON and future_long > future_short + EPSILON),
                "current_queue_tie_bridge_proxy": _queue_tie_proxy(np.asarray(info["raw_queue_bits"], dtype=float)),
                "oracle_exogenous_knowledge": True,
                "not_round7_bridge_evidence": True,
            })
        prior_position = np.asarray(info["uav_position_m"], dtype=float)
        observation, reward, terminated, truncated, after = env.step(np.asarray(action, dtype=float))
        transmitted = np.asarray(after["transmitted_payload_bits"], dtype=float)
        payload_total += float(transmitted.sum())
        return_total += float(reward)
        boundary_count += int(bool(after["boundary_violation"]))
        residual_vector = np.asarray(after["queue_conservation_residual_bits"], dtype=float)
        residual_error = max(residual_error, float(np.max(np.abs(residual_vector))))
        if bool(after["task_success"]) and completion_step is None:
            completion_step = step + 1
        current_position = np.asarray(after["uav_position_m"], dtype=float)
        buoy_positions = np.asarray(after["buoy_positions_m"], dtype=float)
        records.append({
            "scenario_id": scenario.scenario_id, "policy_id": policy_id, "mode": _policy_descriptor(policy_id)[0],
            "horizon": _policy_descriptor(policy_id)[1] or 1, "step": step,
            "action": _json_vector(action), "action_norm": float(np.linalg.norm(action)),
            "position_before_m": _json_vector(prior_position), "position_after_m": _json_vector(current_position),
            "nearest_buoy_index": _close_buoy(current_position, buoy_positions),
            "raw_queue_bits": _json_vector(after["raw_queue_bits"]),
            "payload_bits": float(transmitted.sum()), "transmitted_payload_bits": _json_vector(transmitted),
            "reward": float(reward), "boundary_violation": bool(after["boundary_violation"]),
            "queue_conservation_residual_bits": _json_vector(residual_vector),
            "cumulative_payload_bits": float(after["cumulative_payload_bits"]),
        })
        info = after
        if terminated or truncated:
            break
    metric = {
        "scenario_id": scenario.scenario_id, "policy_id": policy_id, "mode": _policy_descriptor(policy_id)[0],
        "horizon": _policy_descriptor(policy_id)[1] or 1, "difficulty": scenario.difficulty,
        "num_buoys": scenario.num_buoys, "queue_pressure": scenario.queue_pressure, "geometry": scenario.geometry,
        "steps": len(records), "cumulative_actual_payload_bits": payload_total,
        "reported_cumulative_payload_bits": float(info["cumulative_payload_bits"]),
        "final_residual_queue_bits": float(np.asarray(info["raw_queue_bits"], dtype=float).sum()),
        "episode_return": return_total, "cumulative_distance_m": float(info["cumulative_distance_m"]),
        "cumulative_energy_consumption": float(info["cumulative_energy_consumption"]),
        "completion_step": int(completion_step if completion_step is not None else env.config.episode_horizon + 1),
        "completed": completion_step is not None, "boundary_violation_count": boundary_count,
        "queue_conservation_max_abs_error_bits": residual_error,
        "arrival_mode": str(info["arrival_mode"]), "scenario_source": "new_round15_planning_diagnostic",
        "old_development_test_accessed": False, "untouched_holdout_used_for_analysis": False,
        "oracle_exogenous_knowledge": policy_id.startswith("oracle_"),
        "deployable_diagnostic": not policy_id.startswith("oracle_"),
    }
    if not math.isclose(metric["cumulative_actual_payload_bits"], metric["reported_cumulative_payload_bits"], rel_tol=0.0, abs_tol=1.0e-6):
        raise RuntimeError("actual payload accumulator disagrees with environment counter")
    return metric, records, runtimes, delayed


def _bootstrap(values: Sequence[float], *, seed_offset: int = 0, replicates: int = 4000) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        raise ValueError("empty bootstrap input")
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(15015 + seed_offset)
    means = np.empty(replicates, dtype=float)
    for index in range(replicates):
        means[index] = float(array[rng.integers(0, len(array), size=len(array))].mean())
    return mean, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _group_maps(rows: Sequence[Mapping[str, Any]]) -> Iterable[tuple[str, str, list[Mapping[str, Any]]]]:
    yield "overall", "all", list(rows)
    for key in ("difficulty", "num_buoys", "queue_pressure", "geometry"):
        for value in sorted({str(row[key]) for row in rows}):
            yield key, value, [row for row in rows if str(row[key]) == value]


def _comparison_rows(metrics: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {(str(row["scenario_id"]), str(row["policy_id"])): row for row in metrics}
    paired: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for horizon in (5, 10, 20):
        comparison = f"causal_H{horizon}-minus-causal_H1"
        raw: list[dict[str, Any]] = []
        for scenario in planning_scenarios():
            left, right = by_key[(scenario.scenario_id, f"causal_H{horizon}")], by_key[(scenario.scenario_id, "causal_H1")]
            for metric in PRIMARY_METRICS:
                row = {
                    "comparison": comparison, "metric": metric, "scenario_id": scenario.scenario_id,
                    "difficulty": scenario.difficulty, "num_buoys": scenario.num_buoys,
                    "queue_pressure": scenario.queue_pressure, "geometry": scenario.geometry,
                    "candidate_policy": left["policy_id"], "baseline_policy": right["policy_id"],
                    "candidate_value": float(left[metric]), "baseline_value": float(right[metric]),
                    "paired_delta": float(left[metric]) - float(right[metric]),
                }
                paired.append(row); raw.append(row)
        for metric in PRIMARY_METRICS:
            selected = [row for row in raw if row["metric"] == metric]
            for group_key, group_value, group in _group_maps(selected):
                seed = sum(ord(char) for char in f"{comparison}|{metric}|{group_key}|{group_value}")
                mean, low, high = _bootstrap([float(row["paired_delta"]) for row in group], seed_offset=seed)
                summary.append({
                    "comparison": comparison, "metric": metric, "group_key": group_key, "group_value": group_value,
                    "n_scenarios": len(group), "mean_paired_delta": mean, "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high, "positive_fraction": float(np.mean([float(row["paired_delta"]) > 0.0 for row in group])),
                })
    return paired, summary


def _horizon_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ("causal", "oracle"):
        previous: dict[str, float] | None = None
        for horizon in (1, 5, 10, 20):
            selected = [row for row in metrics if row["policy_id"] == f"{mode}_H{horizon}"]
            average = {metric: float(np.mean([float(row[metric]) for row in selected])) for metric in PRIMARY_METRICS}
            row = {"mode": mode, "horizon": horizon, "n_scenarios": len(selected), **{f"mean_{key}": value for key, value in average.items()}}
            for metric, value in average.items():
                row[f"increment_from_previous_{metric}"] = "" if previous is None else value - previous[metric]
            row["oracle_exogenous_knowledge"] = mode == "oracle"
            row["deployable_diagnostic"] = mode == "causal"
            rows.append(row); previous = average
    return rows


def _gap_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(str(row["scenario_id"]), str(row["policy_id"])): row for row in metrics}
    rows: list[dict[str, Any]] = []
    for scenario in planning_scenarios():
        heuristics = [by_key[(scenario.scenario_id, name)] for name in ("nearest_buoy", "max_queue", "queue_rate_energy")]
        best = max(heuristics, key=lambda item: float(item["cumulative_actual_payload_bits"]))
        causal, oracle = by_key[(scenario.scenario_id, "causal_H20")], by_key[(scenario.scenario_id, "oracle_H20")]
        rows.append({
            "scenario_id": scenario.scenario_id, "difficulty": scenario.difficulty, "num_buoys": scenario.num_buoys,
            "best_existing_heuristic": best["policy_id"], "best_existing_payload_bits": best["cumulative_actual_payload_bits"],
            "causal_h20_payload_bits": causal["cumulative_actual_payload_bits"], "oracle_h20_payload_bits": oracle["cumulative_actual_payload_bits"],
            "causal_h20_minus_best_heuristic_bits": float(causal["cumulative_actual_payload_bits"]) - float(best["cumulative_actual_payload_bits"]),
            "oracle_h20_minus_best_heuristic_bits": float(oracle["cumulative_actual_payload_bits"]) - float(best["cumulative_actual_payload_bits"]),
            "oracle_exogenous_knowledge": True,
        })
    return rows


def _decode_vector(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def _trajectory_divergence(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        indexed[(str(row["scenario_id"]), str(row["policy_id"]))].append(row)
    result: list[dict[str, Any]] = []
    for scenario in planning_scenarios():
        reference = sorted(indexed[(scenario.scenario_id, "causal_H1")], key=lambda row: int(row["step"]))
        for horizon in (5, 10, 20):
            candidate = sorted(indexed[(scenario.scenario_id, f"causal_H{horizon}")], key=lambda row: int(row["step"]))
            common = min(len(reference), len(candidate))
            angles = [_angle_degrees(_decode_vector(candidate[i]["action"]), _decode_vector(reference[i]["action"])) for i in range(common)]
            position_distances = [float(np.linalg.norm(_decode_vector(candidate[i]["position_after_m"]) - _decode_vector(reference[i]["position_after_m"]))) for i in range(common)]
            visits = [int(candidate[i]["nearest_buoy_index"]) != int(reference[i]["nearest_buoy_index"]) for i in range(common)]
            queue_l2 = [float(np.linalg.norm(_decode_vector(candidate[i]["raw_queue_bits"]) - _decode_vector(reference[i]["raw_queue_bits"]))) for i in range(common)]
            result.append({
                "scenario_id": scenario.scenario_id, "difficulty": scenario.difficulty, "num_buoys": scenario.num_buoys,
                "comparison": f"causal_H{horizon}-vs-causal_H1", "common_steps": common,
                "action_divergence_rate": float(np.mean([angle > 1.0e-8 for angle in angles])),
                "mean_action_angle_degrees": float(np.mean(angles)), "mean_position_distance_m": float(np.mean(position_distances)),
                "visitation_divergence_rate": float(np.mean(visits)), "visitation_sequences_identical": not any(visits),
                "mean_queue_l2_distance_bits": float(np.mean(queue_l2)),
            })
    return result


def _aggregate_delayed(rows: Sequence[Mapping[str, Any]], key: str, value: str) -> dict[str, Any]:
    group = [row for row in rows if key == "overall" or str(row[key]) == value]
    delayed = [row for row in group if bool(row["delayed_benefit"])]
    reversals = [row for row in group if bool(row["actions_differ"])]
    return {
        "group_key": key, "group_value": value, "state_count": len(group), "delayed_benefit_count": len(delayed),
        "delayed_benefit_rate": float(len(delayed) / len(group)),
        "mean_immediate_sacrifice_bits": float(np.mean([float(row["immediate_sacrifice_bits"]) for row in delayed])) if delayed else 0.0,
        "mean_future_gain_bits": float(np.mean([float(row["future_gain_bits"]) for row in delayed])) if delayed else 0.0,
        "action_reversal_count": len(reversals), "action_reversal_rate": float(len(reversals) / len(group)),
        "mean_reversal_angle_degrees": float(np.mean([float(row["action_angle_degrees"]) for row in reversals])) if reversals else 0.0,
    }


def _delayed_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups = [("overall", "all")]
    for key in ("difficulty", "num_buoys", "queue_pressure", "geometry"):
        groups.extend((key, str(value)) for value in sorted({row[key] for row in rows}))
    delayed = [_aggregate_delayed(rows, key, value) for key, value in groups]
    reversal = [{key: value for key, value in row.items() if key not in {"delayed_benefit_count", "delayed_benefit_rate", "mean_immediate_sacrifice_bits", "mean_future_gain_bits"}} for row in delayed]
    bridge_rows: list[dict[str, Any]] = []
    for is_proxy in (True, False):
        group = [row for row in rows if bool(row["current_queue_tie_bridge_proxy"]) is is_proxy]
        bridge_rows.append({
            "current_queue_tie_bridge_proxy": is_proxy, "state_count": len(group),
            "delayed_benefit_count": int(sum(bool(row["delayed_benefit"]) for row in group)),
            "delayed_benefit_rate": float(np.mean([bool(row["delayed_benefit"]) for row in group])) if group else 0.0,
            "interpretation": "Diagnostic proxy only; not Round-7 bridge evidence; protected Round-7 artifacts were not read",
            "round7_artifacts_read": False,
        })
    return delayed, reversal, bridge_rows


def _runtime_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for mode in ("causal", "oracle"):
        for horizon in (1, 5, 10, 20):
            selected = [row for row in rows if row["mode"] == mode and int(row["horizon"]) == horizon]
            runtime = np.asarray([float(row["runtime_seconds"]) for row in selected], dtype=float)
            nodes = np.asarray([int(row["nodes_expanded"]) for row in selected], dtype=float)
            result.append({
                "mode": mode, "horizon": horizon, "decision_count": len(selected),
                "mean_runtime_seconds": float(runtime.mean()), "p95_runtime_seconds": float(np.quantile(runtime, .95)),
                "mean_nodes_expanded": float(nodes.mean()), "max_nodes_expanded": int(nodes.max()),
                "oracle_exogenous_knowledge": mode == "oracle", "deployable_diagnostic": mode == "causal",
            })
    return result


def _gate(summary: Sequence[Mapping[str, Any]], delayed: Sequence[Mapping[str, Any]], divergence: Sequence[Mapping[str, Any]], horizon_values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def find(comparison: str, group_key: str, group_value: str) -> Mapping[str, Any]:
        return next(row for row in summary if row["comparison"] == comparison and row["metric"] == "cumulative_actual_payload_bits" and row["group_key"] == group_key and row["group_value"] == group_value)
    h10, h20 = find("causal_H10-minus-causal_H1", "overall", "all"), find("causal_H20-minus-causal_H1", "overall", "all")
    difficult = [row for row in summary if row["comparison"] == "causal_H20-minus-causal_H1" and row["metric"] == "cumulative_actual_payload_bits" and row["group_key"] in ("difficulty", "num_buoys")]
    delayed_overall = next(row for row in delayed if row["group_key"] == "overall")
    diversity = any(float(row["action_divergence_rate"]) > 0 for row in divergence if row["comparison"] == "causal_H20-vs-causal_H1")
    strong = float(h10["bootstrap_ci_low"]) > 0 and float(h20["bootstrap_ci_low"]) > 0 and all(float(row["mean_paired_delta"]) > 0 for row in difficult) and int(delayed_overall["delayed_benefit_count"]) > 0 and diversity
    conditional = float(h20["bootstrap_ci_low"]) <= 0 <= float(h20["bootstrap_ci_high"]) and any(row["group_key"] == "difficulty" and row["group_value"] == "hard" and float(row["bootstrap_ci_low"]) > 0 for row in difficult) and int(delayed_overall["delayed_benefit_count"]) > 0
    classification = "PN-A" if strong else "PN-B" if conditional else "PN-C"
    causal = {int(row["horizon"]): row for row in horizon_values if row["mode"] == "causal"}
    saturation = 20
    for previous, current in ((1, 5), (5, 10), (10, 20)):
        if float(causal[current]["increment_from_previous_cumulative_actual_payload_bits"]) <= EPSILON:
            saturation = current
            break
    return {
        "schema_version": SCHEMA, "gate": classification,
        "long_horizon_planning_required": True if classification == "PN-A" else "conditional" if classification == "PN-B" else False,
        "planning_saturation_horizon": saturation, "has_artificial_pass_threshold": False,
        "causal_h10_payload_ci": [h10["bootstrap_ci_low"], h10["bootstrap_ci_high"]],
        "causal_h20_payload_ci": [h20["bootstrap_ci_low"], h20["bootstrap_ci_high"]],
        "delayed_benefit_state_count": int(delayed_overall["delayed_benefit_count"]),
        "h20_action_divergence_observed": diversity,
        "criteria": {"pn_a": strong, "pn_b": conditional, "pn_c": not (strong or conditional)},
        "interpretation": "Descriptive audit gate; it is not a deployment recommendation or an authorization for another round.",
    }


def run_audit(root: Path, output_dir: Path) -> dict[str, Any]:
    """Write the raw, paired, and gate artifacts to a fresh Round-15 output."""
    root, output_dir = Path(root), Path(output_dir)
    protocol, controls = _protocol(root), preflight(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _write_json(output_dir / "planning_scenario_manifest.json", {"schema_version": SCHEMA, "scenarios": scenario_manifest_rows(), "controls": controls})
    metrics: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    delayed: list[dict[str, Any]] = []
    for scenario in planning_scenarios():
        for policy_id in POLICY_IDS:
            metric, trace, runtime, delayed_states = _run_episode(scenario, policy_id)
            metrics.append(metric); trajectories.extend(trace); runtimes.extend(runtime); delayed.extend(delayed_states)
    paired, paired_summary = _comparison_rows(metrics)
    horizon_values = _horizon_rows(metrics)
    gaps = _gap_rows(metrics)
    divergence = _trajectory_divergence(trajectories)
    delayed_summary, reversal, bridge = _delayed_rows(delayed)
    runtime_summary = _runtime_rows(runtimes)
    gate = _gate(paired_summary, delayed_summary, divergence, horizon_values)
    _write_csv(output_dir / "planning_episode_metrics.csv", metrics)
    _write_csv(output_dir / "planning_trajectory_steps.csv", trajectories)
    _write_csv(output_dir / "planning_runtime_events.csv", runtimes)
    _write_csv(output_dir / "delayed_benefit_states.csv", delayed)
    _write_csv(output_dir / "planning_paired_comparison.csv", paired)
    _write_csv(output_dir / "planning_headroom_metrics.csv", paired_summary)
    _write_csv(output_dir / "planning_horizon_value.csv", horizon_values)
    _write_csv(output_dir / "heuristic_planning_gap.csv", gaps)
    _write_csv(output_dir / "trajectory_divergence_audit.csv", divergence)
    _write_csv(output_dir / "delayed_benefit_audit.csv", delayed_summary)
    _write_csv(output_dir / "action_reversal_audit.csv", reversal)
    _write_csv(output_dir / "bridge_planning_relation.csv", bridge)
    _write_csv(output_dir / "planning_runtime_audit.csv", runtime_summary)
    _write_json(output_dir / "planning_necessity_gate.json", gate)
    _write_json(output_dir / "round15_readiness_gate.json", {
        "schema_version": SCHEMA, "planning_necessity_gate": gate["gate"], "round16_created": False,
        "next_round_authorized": False, "message": "Round-15 stops here; a new explicit instruction is required for any later round.",
    })
    run = {
        "schema_version": SCHEMA, "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "output_dir": output_dir.name, "protocol": protocol, "controls": controls,
        "scenario_count": len(planning_scenarios()), "policy_count": len(POLICY_IDS), "episode_count": len(metrics),
        "trajectory_row_count": len(trajectories), "runtime_event_count": len(runtimes), "delayed_state_count": len(delayed),
        "elapsed_seconds": time.perf_counter() - started, "gate": gate,
    }
    _write_json(output_dir / "round15_run.json", run)
    return run


__all__ = ["POLICY_IDS", "dispatch_policy_action", "preflight", "run_audit"]
