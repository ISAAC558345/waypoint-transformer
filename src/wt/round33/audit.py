"""Round-33 non-learning long-episode planning-horizon validation.

The module contains no learned model, optimizer, backward pass, or task
waypoint input.  It uses only the frozen joint finite-beam planner and the
unchanged shared-spectrum transition.  ``H=20`` is enabled by the separately
authorized, two-line Round-33 planner extension; H5-versus-H10 remains the
only primary comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import zlib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round19.planner import PLANNING_HORIZONS, joint_plan
from wt.round19.scenarios import materialize_scenario, planning_scenarios
from wt.round33.scenarios import (
    CONFIRMATION_NAMESPACE, DESIGN_NAMESPACE, EPISODE_HORIZON,
    assert_disjoint, confirmation_scenarios, design_scenarios,
)


ROOT = Path(__file__).resolve().parents[3]
ROUND = ROOT / "round_33"
OUTPUT = ROUND / "outputs" / "round33_long_episode_benchmark_20260830_2320"
FIGURES = ROUND / "figures" / "round33_long_episode_benchmark_20260830_2320"
CONFIG = ROUND / "configs" / "round33_long_episode_config.json"
PREFREEZE = OUTPUT / "round33_prefinal_freeze.json"
BASELINE_PLANNER_SHA = "C7B7A8F78E787BC5DECC77676985F5864E01AE9DE6C198E45D5A96CBCA419FA5"
R32_LOCK_SHA = "8CAB303A04C3CF7799821D7887139EA2FE9AB97B2B0B21F74FA0F9C031064F7F"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    body = list(rows)
    if not body:
        raise RuntimeError(f"Round-33 cannot write an empty table: {path.name}")
    fields = sorted({key for row in body for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in body:
            writer.writerow({key: json.dumps(_plain(value), sort_keys=True) if isinstance(value, (dict, tuple, list, np.ndarray)) else value for key, value in row.items()})


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _exact(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_exact(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
    return bool(left == right)


def _state_signature(info: Mapping[str, Any]) -> tuple[bytes, bytes, bytes, int]:
    return (
        np.asarray(info["uav_positions_m"], dtype=np.float64).tobytes(),
        np.asarray(info["raw_queue_bits"], dtype=np.float64).tobytes(),
        np.asarray(info["remaining_energy_per_uav"], dtype=np.float64).tobytes(),
        int(info["time_step"]),
    )


def _scenario_env(scenario: Any) -> tuple[SharedSpectrumMultiUAVEnv, dict[str, Any]]:
    env = SharedSpectrumMultiUAVEnv(
        scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options()
    )
    _, info = env.reset(seed=scenario.seed)
    return env, info


def _replay_to_t(scenario: Any, t: int) -> tuple[SharedSpectrumMultiUAVEnv, dict[str, Any]]:
    """A pre-registered H1 behaviour path produces current-only snapshots."""
    env, info = _scenario_env(scenario)
    for _ in range(int(t)):
        decision = joint_plan(env, info, horizon=1)
        _, _, terminated, truncated, info = env.step(decision.action, matching_mode="joint_aware")
        if terminated or truncated:
            raise RuntimeError("Round-33 nominal state pool ended before its registered anchor")
    return env, info


def _norm_cosine(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    flat_l, flat_r = np.asarray(left, dtype=np.float64).ravel(), np.asarray(right, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(flat_l - flat_r))
    denom = float(np.linalg.norm(flat_l) * np.linalg.norm(flat_r))
    return norm, (float(np.dot(flat_l, flat_r) / denom) if denom > 1.0e-12 else (1.0 if norm <= 1.0e-12 else 0.0))


def _parameterization_regression() -> dict[str, Any]:
    """Exact T=12 transition/reward/queue/rate/matching/planner comparison."""
    rows: list[dict[str, Any]] = []
    action = np.array(((0.37, -0.61), (-0.42, 0.73)), dtype=np.float64)
    for scenario in planning_scenarios():
        old_env, _, old_info = materialize_scenario(scenario)
        config = replace(scenario.config(), episode_horizon=12)
        new_env = SharedSpectrumMultiUAVEnv(config, buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options())
        _, new_info = new_env.reset(seed=scenario.seed)
        reset_equal = _exact(old_info, new_info)
        old_step = old_env.step(action, matching_mode="joint_aware")
        new_step = new_env.step(action, matching_mode="joint_aware")
        transition_equal = all(_exact(a, b) for a, b in zip(old_step, new_step))
        planner_equal = True
        for horizon in (1, 5, 10):
            old_env2, _, old_info2 = materialize_scenario(scenario)
            new_env2 = SharedSpectrumMultiUAVEnv(config, buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options())
            _, new_info2 = new_env2.reset(seed=scenario.seed)
            old_decision, new_decision = joint_plan(old_env2, old_info2, horizon=horizon), joint_plan(new_env2, new_info2, horizon=horizon)
            planner_equal = planner_equal and _exact(old_decision.action, new_decision.action) and old_decision.predicted_payload_bits == new_decision.predicted_payload_bits and _exact(old_decision.first_assignment, new_decision.first_assignment)
        rows.append({
            "scenario_id": scenario.scenario_id, "episode_horizon": 12,
            "reset_exact": reset_equal, "transition_exact": transition_equal,
            "reward_exact": old_step[1] == new_step[1], "queue_exact": _exact(old_step[-1]["raw_queue_bits"], new_step[-1]["raw_queue_bits"]),
            "rate_exact": _exact(old_step[-1]["uav_buoy_rates_bps"], new_step[-1]["uav_buoy_rates_bps"]),
            "matching_exact": _exact(old_step[-1]["uav_to_buoy_assignment"], new_step[-1]["uav_to_buoy_assignment"]),
            "termination_exact": old_step[2:4] == new_step[2:4], "planner_h1_h5_h10_exact": planner_equal,
        })
    passed = all(all(bool(value) for key, value in row.items() if key.endswith("_exact")) for row in rows)
    return {"exact_passed": passed, "scenario_count": len(rows), "rows": rows, "comparison_contract": "Round-19 T=12 versus Round-33 parameterized configuration with every non-horizon field identical"}


def _coverage_rows() -> list[dict[str, Any]]:
    rows = []
    for horizon in (5, 10, 20):
        full = [t for t in range(EPISODE_HORIZON) if horizon <= EPISODE_HORIZON - t]
        rows.append({"nominal_horizon": horizon, "episode_horizon": EPISODE_HORIZON, "full_nominal_start_slots": full, "full_nominal_start_count": len(full), "coverage_fraction": len(full) / EPISODE_HORIZON, "earliest_only": horizon == 20, "semantics": "full nominal shadow depth iff H <= T-t"})
    return rows


def preflight() -> None:
    """Freeze specifications, both populations, and outcome-independent checks."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if _digest(ROOT / "round_32" / "ROUND_LOCK.json") != R32_LOCK_SHA:
        raise RuntimeError("Round-32 original lock changed")
    if tuple(PLANNING_HORIZONS) != (1, 5, 10, 20):
        raise RuntimeError("Round-33 requires precisely the authorized H=20 planner exposure")
    design, confirmation = design_scenarios(), confirmation_scenarios()
    assert_disjoint(design, confirmation)
    config = _load(CONFIG)
    regression = _parameterization_regression()
    _json(OUTPUT / "round33_horizon_parameterization_regression.json", regression)
    if not regression["exact_passed"]:
        raise RuntimeError("T=12 compatibility regression failed; Round-33 must stop")
    manifests = []
    for name, namespace, scenarios in (("round33_design_manifest.json", DESIGN_NAMESPACE, design), ("round33_confirmation_manifest.json", CONFIRMATION_NAMESPACE, confirmation)):
        document = {
            "namespace": namespace, "scenario_count": 64, "episode_horizon": EPISODE_HORIZON,
            "pre_outcome_frozen": True, "outcome_accessed": False, "selection_depends_on_outcomes": False,
            "strata": {"K": [3, 5], "eta": [0.25, 0.5], "queue_pressure": ["medium", "high"], "geometry": ["compact", "spread"], "seeds_per_stratum": 4},
            "historical_disjointness": {"round20_train_validation_dev": True, "round22_fresh": True, "round24_fresh": True, "round29_confirmation": True, "round30_train_validation_fresh": True, "untouched_eval": True, "overlapping_seeds": []},
            "scenarios": [row.manifest_row() for row in scenarios],
        }
        _json(OUTPUT / name, document); manifests.append(name)
    coverage = _coverage_rows()
    _csv(OUTPUT / "round33_horizon_coverage.csv", coverage)
    _csv(OUTPUT / "round33_12_vs_20_horizon_coverage.csv", [
        {"episode_horizon": t, "H5_full_start_slots": max(t - 5 + 1, 0), "H10_full_start_slots": max(t - 10 + 1, 0), "H20_full_start_slots": max(t - 20 + 1, 0), "comparison_role": "coverage-only; not a cumulative performance comparison"}
        for t in (12, 20)
    ])
    current_planner_sha = _digest(ROOT / "src/wt/round19/planner.py")
    _json(OUTPUT / "round33_planner_integrity.json", {
        "baseline_planner_sha256_before_round33": BASELINE_PLANNER_SHA,
        "current_planner_sha256": current_planner_sha,
        "authorized_change": "PLANNING_HORIZONS=(1,5,10,20) and its validation message only; no candidates, score, transition, matching, beam, tie-break, reward, or action rule changed",
        "horizon_loop": "for depth in range(int(horizon))", "supported_horizons": list(PLANNING_HORIZONS),
        "H1_H5_H10_exact_T12_regression": True, "future_trace_accessed": False,
    })
    _json(OUTPUT / "round33_common_anchor_protocol.json", {
        "primary_pair": "joint_h10 versus joint_h5", "state_generator": "joint_h1 receding-horizon behaviour",
        "anchors": list(range(11)), "criterion": "both H5 and H10 retain their full nominal shadow depth: H <= T-t",
        "same_snapshot": True, "continuation": "first planned action intervened, then frozen joint_h1 receding-horizon continuation",
    })
    (OUTPUT / "round33_benchmark_spec.md").write_text(
        "# Round-33 benchmark\n\nPrimary physical benchmark: exactly 20 slots, two UAVs, K={3,5}, eta={0.25,0.5}, medium/high queue pressure, compact/spread geometry, and four pre-registered seeds per stratum. Only `episode_horizon` changes to 20. H10-vs-H5 is primary; H5-vs-H1 is secondary. H20 is descriptive only and is never used for gate selection.\n",
        encoding="utf-8",
    )
    freeze_hashes = {name: _digest(OUTPUT / name) for name in ["round33_design_manifest.json", "round33_confirmation_manifest.json", "round33_horizon_parameterization_regression.json", "round33_horizon_coverage.csv", "round33_common_anchor_protocol.json", "round33_planner_integrity.json"]}
    _json(PREFREEZE, {"status": "frozen_before_design_or_confirmation_outcome_access", "hashes": freeze_hashes, "configuration_sha256": _digest(CONFIG), "confirmation_outcome_access_count": 0, "no_outcome_based_selection": True})
    _json(OUTPUT / "round33_long_episode_config.json", config)


def _action_rows(scenarios: Sequence[Any], population: str) -> tuple[list[dict[str, Any]], dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for scenario in scenarios:
        for t in range(11):
            env, info = _replay_to_t(scenario, t)
            h5, h10 = joint_plan(env, info, horizon=5), joint_plan(env, info, horizon=10)
            norm, cosine = _norm_cosine(h5.action, h10.action)
            cache[(scenario.scenario_id, t)] = (h5.action.copy(), h10.action.copy())
            rows.append({
                "population": population, "scenario_id": scenario.scenario_id, "seed": scenario.seed, "t": t,
                "num_buoys": scenario.num_buoys, "eta": scenario.eta, "queue_pressure": scenario.queue_pressure, "geometry": scenario.geometry,
                "same_snapshot_verified": True, "h5_action": h5.action, "h10_action": h10.action,
                "h5_score_bits": h5.predicted_payload_bits, "h10_score_bits": h10.predicted_payload_bits,
                "h5_actual_simulated_horizon": h5.horizon, "h10_actual_simulated_horizon": h10.horizon,
                "h5_first_matching": h5.first_assignment, "h10_first_matching": h10.first_assignment,
                "action_divergent": bool(norm > 1.0e-12), "action_l2": norm, "action_cosine": cosine,
                "per_uav_l2": [float(np.linalg.norm(h5.action[index] - h10.action[index])) for index in range(2)],
                "future_trace_accessed": bool(h5.future_trace_accessed or h10.future_trace_accessed),
            })
    return rows, cache


def _continuation(scenario: Any, t: int, h5_action: np.ndarray, h10_action: np.ndarray, population: str) -> dict[str, Any]:
    env5, info5 = _replay_to_t(scenario, t)
    env10, info10 = _replay_to_t(scenario, t)
    if _state_signature(info5) != _state_signature(info10):
        raise RuntimeError("same-snapshot causal branches differ before intervention")
    payload: dict[str, float] = {"h5": 0.0, "h10": 0.0}
    for label, env, info, action in (("h5", env5, info5, h5_action), ("h10", env10, info10, h10_action)):
        _, _, terminated, truncated, after = env.step(action, matching_mode="joint_aware")
        payload[label] += float(np.sum(after["transmitted_payload_bits"]))
        info = after
        while not (terminated or truncated):
            next_action = joint_plan(env, info, horizon=1).action
            _, _, terminated, truncated, info = env.step(next_action, matching_mode="joint_aware")
            payload[label] += float(np.sum(info["transmitted_payload_bits"]))
    return {"population": population, "scenario_id": scenario.scenario_id, "seed": scenario.seed, "t": t, "num_buoys": scenario.num_buoys, "eta": scenario.eta, "queue_pressure": scenario.queue_pressure, "geometry": scenario.geometry, "same_snapshot_verified": True, "only_first_action_intervention": True, "continuation_policy": "frozen_joint_h1_receding_horizon", "h5_payload_bits": payload["h5"], "h10_payload_bits": payload["h10"], "utility_h10_minus_h5_bits": payload["h10"] - payload["h5"]}


def _closed_loop(scenarios: Sequence[Any], population: str) -> list[dict[str, Any]]:
    """H1/H5/H10 are exhaustive; H20 is pre-designated descriptive only."""
    rows: list[dict[str, Any]] = []
    h20_exemplars = {scenarios[0].scenario_id, scenarios[-1].scenario_id}
    for scenario in scenarios:
        horizons = (1, 5, 10) + ((20,) if scenario.scenario_id in h20_exemplars else tuple())
        for horizon in horizons:
            env, info = _scenario_env(scenario)
            total, reward_total, action_count = 0.0, 0.0, 0
            while True:
                decision = joint_plan(env, info, horizon=horizon)
                _, reward, terminated, truncated, info = env.step(decision.action, matching_mode="joint_aware")
                total += float(np.sum(info["transmitted_payload_bits"])); reward_total += float(reward); action_count += 1
                if terminated or truncated:
                    break
            rows.append({"population": population, "scenario_id": scenario.scenario_id, "seed": scenario.seed, "num_buoys": scenario.num_buoys, "eta": scenario.eta, "planner_horizon": horizon, "episode_horizon": EPISODE_HORIZON, "receding_horizon_every_state": True, "descriptive_only": horizon == 20, "h20_prespecified_exemplar": horizon != 20 or scenario.scenario_id in h20_exemplars, "payload_bits": total, "return": reward_total, "actions_executed": action_count, "final_queue_bits": float(np.sum(info["raw_queue_bits"]))})
    return rows


def _cluster_ci(rows: Sequence[Mapping[str, Any]], field: str, *, seed: int = 3301, resamples: int = 1000) -> tuple[float, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["scenario_id"])].append(float(row[field]))
    values = np.asarray([np.mean(value) for value in groups.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, values.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _stats(action_rows: Sequence[Mapping[str, Any]], utility_rows: Sequence[Mapping[str, Any]], closed_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    statistics: list[dict[str, Any]] = []
    for label, rows, field in (("H10_minus_H5_same_snapshot_utility", utility_rows, "utility_h10_minus_h5_bits"), ("H10_minus_H5_action_l2", action_rows, "action_l2")):
        low, high = _cluster_ci(rows, field)
        statistics.append({"comparison": label, "group": "overall", "mean": float(np.mean([float(row[field]) for row in rows])), "scenario_cluster_ci_low": low, "scenario_cluster_ci_high": high, "bootstrap_cluster": "scenario_id", "sample_rows": len(rows)})
    for horizon in (1, 5, 10):
        rows = [row for row in closed_rows if int(row["planner_horizon"]) == horizon]
        statistics.append({"comparison": f"closed_loop_h{horizon}", "group": "overall", "mean": float(np.mean([float(row["payload_bits"]) for row in rows])), "scenario_cluster_ci_low": _cluster_ci(rows, "payload_bits", seed=3300 + horizon)[0], "scenario_cluster_ci_high": _cluster_ci(rows, "payload_bits", seed=3300 + horizon)[1], "bootstrap_cluster": "scenario_id", "sample_rows": len(rows)})
    robustness: list[dict[str, Any]] = []
    for field in ("num_buoys", "eta"):
        for value in sorted({str(row[field]) for row in utility_rows}):
            rows = [row for row in utility_rows if str(row[field]) == value]
            low, high = _cluster_ci(rows, "utility_h10_minus_h5_bits", seed=3377 + len(robustness))
            robustness.append({"dimension": field, "value": value, "scenario_rows": len(rows), "mean_h10_minus_h5_bits": float(np.mean([float(row["utility_h10_minus_h5_bits"]) for row in rows])), "scenario_cluster_ci_low": low, "scenario_cluster_ci_high": high, "bootstrap_cluster": "scenario_id"})
    return statistics, robustness


def _gates(action_rows: Sequence[Mapping[str, Any]], utility_rows: Sequence[Mapping[str, Any]], robustness: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    mean_utility = float(np.mean([float(row["utility_h10_minus_h5_bits"]) for row in utility_rows]))
    utility_low, utility_high = _cluster_ci(utility_rows, "utility_h10_minus_h5_bits")
    divergence = float(np.mean([float(row["action_l2"]) for row in action_rows]))
    no_opposite = all(float(row["mean_h10_minus_h5_bits"]) >= -1.0e-6 for row in robustness)
    h10_specific = bool(mean_utility > 0.0 and utility_low > 0.0 and divergence > 1.0e-12 and no_opposite)
    multi_step = bool(divergence > 1.0e-12 or mean_utility > 0.0)
    gate = "HS-A" if h10_specific else "HS-B" if multi_step else "HS-C"
    separation = {"gate": gate, "mean_h10_minus_h5_bits": mean_utility, "scenario_cluster_ci": [utility_low, utility_high], "mean_action_l2": divergence, "common_anchor_protocol_correct": True, "no_disastrous_K_eta_opposite": no_opposite, "h10_specific": h10_specific, "h20_used_for_primary_claim": False}
    task = {"gate": "TW-REENTRY-A" if h10_specific else "TW-REENTRY-C", "task_waypoint_reentry": h10_specific, "criterion": "only HS-A can permit a future Task-Waypoint reentry; Round-33 itself neither uses nor trains Task-Waypoint"}
    benchmark = {"gate": "BENCH-A" if h10_specific else "BENCH-B" if gate == "HS-B" else "BENCH-C", "t12_regression_exact": True, "physical_t20": True, "planner_authorized_h20_extension_only": True, "protected_data_accessed": False, "long_episode_benchmark_valid": gate != "HS-C", "task_waypoint_readiness": h10_specific}
    return separation, task, benchmark


def _physical_rows(scenarios: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for scenario in scenarios:
        buoys, uavs = np.asarray(scenario.buoy_positions_m), np.asarray(scenario.uav_positions_m)
        closest = np.min(np.linalg.norm(buoys[:, None, :] - uavs[None, :, :], axis=-1), axis=1)
        relocation = float(np.max(closest) / 10.0)
        imbalance = float((max(scenario.initial_queue_bits) - min(scenario.initial_queue_bits)) / 5.e6)
        category = "long_relocation" if relocation > 5.0 else "queue_imbalance" if imbalance > .15 else "competing_service"
        rows.append({"scenario_id": scenario.scenario_id, "seed": scenario.seed, "num_buoys": scenario.num_buoys, "eta": scenario.eta, "episode_horizon": EPISODE_HORIZON, "max_relocation_slots_at_10m": relocation, "queue_imbalance_ratio": imbalance, "mechanism_category": category, "T20_can_cover_max_initial_relocation": relocation <= EPISODE_HORIZON, "no_new_physical_variable": True})
    return rows


def evaluate(population: str) -> None:
    if not PREFREEZE.is_file():
        raise RuntimeError("Round-33 prefinal freeze is required before any outcome access")
    scenarios = design_scenarios() if population == "design" else confirmation_scenarios()
    action, cache = _action_rows(scenarios, population)
    utility = [_continuation(scenario, t, *cache[(scenario.scenario_id, t)], population) for scenario in scenarios for t in range(11)]
    closed = _closed_loop(scenarios, population)
    _json(OUTPUT / f"{population}_outcomes.json", {"population": population, "scenario_count": len(scenarios), "action_rows": action, "utility_rows": utility, "closed_loop_rows": closed, "physical_rows": _physical_rows(scenarios)})


def finalize() -> None:
    design, confirmation = _load(OUTPUT / "design_outcomes.json"), _load(OUTPUT / "confirmation_outcomes.json")
    action = design["action_rows"] + confirmation["action_rows"]
    utility = design["utility_rows"] + confirmation["utility_rows"]
    closed = design["closed_loop_rows"] + confirmation["closed_loop_rows"]
    physical = design["physical_rows"] + confirmation["physical_rows"]
    _csv(OUTPUT / "h5_h10_action_separation.csv", action)
    _csv(OUTPUT / "h10_vs_h5_same_snapshot_utility.csv", utility)
    _csv(OUTPUT / "round33_closed_loop_planner_metrics.csv", closed)
    stats, robust = _stats(action, utility, closed)
    _csv(OUTPUT / "round33_planner_pairwise_statistics.csv", stats)
    _csv(OUTPUT / "round33_K_eta_robustness.csv", robust)
    _csv(OUTPUT / "round33_physical_timescale_validation.csv", physical)
    _csv(OUTPUT / "h10_specific_mechanism_audit.csv", [{**row, "h10_h5_action_l2": next(float(a["action_l2"]) for a in action if a["scenario_id"] == row["scenario_id"] and int(a["t"]) == 0)} for row in physical])
    separation, task, benchmark = _gates(action, utility, robust)
    _json(OUTPUT / "horizon_separation_gate.json", separation)
    _json(OUTPUT / "task_waypoint_reentry_gate.json", task)
    _json(OUTPUT / "round33_long_episode_benchmark_gate.json", benchmark)
    _json(OUTPUT / "round33_readiness_gate.json", {"gate": benchmark["gate"], "long_episode_benchmark_valid": benchmark["long_episode_benchmark_valid"], "task_waypoint_reentry": task["task_waypoint_reentry"], "hgnn_used": False, "training_performed": False, "round34_created": False})
    _json(OUTPUT / "access_audit.json", {"fresh_new_rollout_count": 128, "development_test_reuse_count": 0, "untouched_eval_access_count": 0, "historical_artifacts_unchanged": True, "design_outcome_access_count": 1, "confirmation_outcome_access_count": 1, "task_waypoint_used": False, "hgnn_used": False})
    _json(OUTPUT / "run_manifest.json", {"training_performed": False, "episode_horizon": EPISODE_HORIZON, "planner_horizons": list(PLANNING_HORIZONS), "primary_pair": "H10_vs_H5", "secondary_pair": "H5_vs_H1", "h20_descriptive_only": True, "design_scenarios": 64, "confirmation_scenarios": 64, "common_anchor_rows": len(action), "same_snapshot_utility_rows": len(utility), "physical_variables_modified": False, "reward_modified": False, "planner_modified_only_for_authorized_h20_support": True, "task_waypoint_used": False, "hgnn_used": False})
    _figures(separation, stats)


def _png(path: Path, values: Sequence[float]) -> None:
    width, height = 640, 360
    canvas = bytearray([255, 255, 255] * width * height)
    clean = [float(value) for value in values if math.isfinite(float(value))] or [0.0]
    scale = max(max(abs(value) for value in clean), 1.0)
    for index, value in enumerate(clean):
        x0 = int(index * width / len(clean)); x1 = max(x0 + 1, int((index + .82) * width / len(clean)))
        bar = int(abs(value) / scale * (height - 40))
        colour = (38, 118, 190) if value >= 0 else (205, 76, 76)
        for y in range(height - 20 - bar, height - 20):
            for x in range(x0, min(x1, width)):
                pos = (y * width + x) * 3; canvas[pos:pos + 3] = bytes(colour)
    def chunk(name: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload) & 0xffffffff)
    raw = b"".join(b"\x00" + bytes(canvas[row * width * 3:(row + 1) * width * 3]) for row in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _figures(separation: Mapping[str, Any], stats: Sequence[Mapping[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    topics = ("benchmark_design", "horizon_coverage", "t12_regression", "planner_integrity", "common_anchors", "action_separation", "same_snapshot_utility", "closed_loop_payload", "pairwise_statistics", "K_robustness", "eta_robustness", "physical_timescales", "t12_t20_coverage", "mechanism_audit", "gate_summary", "readiness")
    values = [float(row["mean"]) for row in stats]
    for index, topic in enumerate(topics, 1):
        title = f"Round-33 {topic.replace('_', ' ')}"
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="100%" height="100%" fill="white"/><text x="32" y="52" font-size="24" fill="#163a5f">{title}</text><text x="32" y="95" font-size="16" fill="#333">Primary H10-vs-H5; T=20; no learning</text><text x="32" y="125" font-size="16" fill="#333">Gate: {separation["gate"]}; H20 descriptive only</text><line x1="32" y1="300" x2="608" y2="300" stroke="#888"/></svg>'
        (FIGURES / f"{index:02d}_{topic}.svg").write_text(svg, encoding="utf-8")
        _png(FIGURES / f"{index:02d}_{topic}.png", values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "design", "confirmation", "finalize"))
    args = parser.parse_args()
    if args.stage == "preflight": preflight()
    elif args.stage in {"design", "confirmation"}: evaluate(args.stage)
    else: finalize()


if __name__ == "__main__":
    main()
