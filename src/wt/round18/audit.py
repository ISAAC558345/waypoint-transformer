"""Formal deterministic coordination and planning audit for Round-18."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.env import MaritimeEnv

from .multi_env import MultiUAVMaritimeEnv, rate_matrix
from .planner import independent_decision, joint_plan
from .scenarios import materialize_scenario, planning_scenarios, scenario_manifest_rows


POLICIES = ("independent_queue_rate", "joint_h1", "joint_h5", "joint_h10")
R17_LOCK_SHA256 = "A393148768835CB28296999D73490C77CAD51279B8060E9B99D10EC8581285FD"


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_plain(v) for v in value]
    return value


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_plain(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_plain(value), sort_keys=True) if isinstance(value, (list, tuple, dict, np.ndarray)) else value for key, value in row.items()})


def _bootstrap(values: Sequence[float], seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)): raise ValueError("paired values must be finite and nonempty")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(3000, array.size), replace=True).mean(axis=1)
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {"n": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)), "std": std, "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975)), "effect_size": float(array.mean() / std) if std > 1.0e-12 else 0.0, "wins": int(np.count_nonzero(array > 1.0e-6)), "ties": int(np.count_nonzero(np.abs(array) <= 1.0e-6)), "losses": int(np.count_nonzero(array < -1.0e-6))}


def preflight(root: Path) -> dict[str, Any]:
    if _sha(root / "round_17" / "ROUND_LOCK.json") != R17_LOCK_SHA256: raise RuntimeError("Round-17 lock changed")
    if (root / "round_19").exists(): raise RuntimeError("Round-19 exists before Round-18 completion")
    manifest = json.loads((root / "round_18" / "round_manifest.json").read_text(encoding="utf-8"))
    forbidden = ("training_performed", "task_waypoint_training_performed", "hgnn_training_performed", "ppo_training_performed")
    if any(bool(manifest[key]) for key in forbidden): raise RuntimeError("Round-18 manifest reports forbidden work")
    if any(int(manifest[key]) != 0 for key in ("old_development_test_access_count", "fresh_development_access_count", "untouched_holdout_new_access_count")): raise RuntimeError("protected-data access count is nonzero")
    return {"round17_lock_sha256": R17_LOCK_SHA256, "primary_num_uavs": 2, "primary_buoy_drift_mode": "static", "primary_channel_variation_mode": "static", "protected_data_access_counts": {"old_development": 0, "fresh_development": 0, "untouched_holdout_new": 0}}


def single_uav_regression() -> list[dict[str, Any]]:
    rows = []
    for index in range(12):
        count = 3 if index % 2 == 0 else 5
        queues = tuple(float(1.0e6 + 1.0e5 * (index + buoy)) for buoy in range(count))
        config = MaritimeEnvConfig(num_uavs=1, num_buoys=count, task_semantics_version="v1", buoy_drift_mode="static", channel_variation_mode="static", initial_queue_bits_by_buoy=queues, terminate_on_task_success=False, episode_horizon=8, seed=18200 + index)
        baseline, candidate = MaritimeEnv(config), MultiUAVMaritimeEnv(config)
        obs_a, info_a = baseline.reset(seed=18200 + index); obs_b, info_b = candidate.reset(seed=18200 + index)
        maximum = float(max(np.max(np.abs(obs_a - obs_b)), np.max(np.abs(info_a["uav_position_m"] - info_b["uav_position_m"])), np.max(np.abs(info_a["raw_queue_bits"] - info_b["raw_queue_bits"]))))
        for step in range(config.episode_horizon):
            action = np.array((math.sin(index + step), math.cos(index - step)), dtype=np.float64)
            obs_a, reward_a, term_a, trunc_a, info_a = baseline.step(action)
            obs_b, reward_b, term_b, trunc_b, info_b = candidate.step(action.reshape(1, 2))
            values = (obs_a, info_a["uav_position_m"], info_a["rates_bps"], info_a["raw_queue_bits"], info_a["transmitted_payload_bits"], np.array((info_a["remaining_energy"], reward_a, float(term_a), float(trunc_a))))
            mirrors = (obs_b, info_b["uav_position_m"], info_b["rates_bps"], info_b["raw_queue_bits"], info_b["transmitted_payload_bits"], np.array((info_b["remaining_energy"], reward_b, float(term_b), float(trunc_b))))
            maximum = max(maximum, *(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(values, mirrors)))
            if term_a or trunc_a: break
        rows.append({"case_id": f"m1_static_{index:02d}", "seed": 18200 + index, "num_buoys": count, "steps_compared": step + 1, "max_absolute_error": maximum, "passed": maximum == 0.0, "covered": "position,buoy_state,rate,queue,payload,energy,reward,termination"})
    if not all(row["passed"] for row in rows): raise RuntimeError("M=1 exact regression failed")
    return rows


def _independent_targets(info: Mapping[str, Any]) -> list[int]:
    queues = np.asarray(info["raw_queue_bits"], dtype=np.float64); rates = np.asarray(info["uav_buoy_rates_bps"], dtype=np.float64)
    eligible = queues > 1.0e-12
    return [-1 if not np.any(eligible) else int(np.argmax(np.where(eligible, queues / float(info["queue_capacity_bits"]) * rates[m], -np.inf))) for m in range(rates.shape[0])]


def _episode(scenario: Any, policy: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env, observation, info = materialize_scenario(scenario)
    steps: list[dict[str, Any]] = []
    total_reward = 0.0; runtime = nodes = actions = 0
    while True:
        if policy == "independent_queue_rate": decision = independent_decision(env, info); intended = _independent_targets(info)
        else:
            horizon = int(policy.rsplit("h", 1)[1]); decision = joint_plan(env, info, horizon=horizon); intended = [int(value) for value in decision.first_assignment]
        observation, reward, terminated, truncated, after = env.step(decision.action)
        total_reward += reward; runtime += decision.runtime_seconds; nodes += decision.nodes_expanded; actions += decision.actions_evaluated
        steps.append({"scenario_id": scenario.scenario_id, "policy": policy, "time_step": int(after["time_step"]), "assignment": after["uav_to_buoy_assignment"].copy(), "intended_targets": intended, "payload_bits": float(np.sum(after["transmitted_payload_bits"])), "payload_by_buoy_bits": after["transmitted_payload_bits"].copy(), "reward": reward, "uav_positions_m": after["uav_positions_m"].copy(), "min_uav_distance_m": float(after["min_uav_distance_m"]), "raw_collision_risk_pairs": int(after["raw_collision_risk_pairs"]), "safety_correction_m": after["safety_correction_m"].copy(), "decision_runtime_ms": 1000.0 * decision.runtime_seconds, "nodes_expanded": decision.nodes_expanded, "actions_evaluated": decision.actions_evaluated, "candidate_names": decision.first_candidate_names})
        info = after
        if terminated or truncated: break
    assignments = [np.asarray(row["assignment"], dtype=int) for row in steps]
    served = {int(value) for row in assignments for value in row if value >= 0}
    spread = [float(np.linalg.norm(np.asarray(row["uav_positions_m"])[1] - np.asarray(row["uav_positions_m"])[0])) for row in steps]
    return {"scenario_id": scenario.scenario_id, "policy": policy, "num_buoys": scenario.num_buoys, "queue_tier": scenario.queue_tier, "pattern": scenario.pattern, "geometry_group": scenario.geometry_group, "coordination_stress": scenario.coordination_stress, "payload_bits": float(after["cumulative_payload_bits"]), "payload_mbit": float(after["cumulative_payload_bits"] / 1.0e6), "completion": float(np.sum(after["cumulative_payload_bits_per_buoy"]) / max(np.sum(scenario.initial_queue_bits), 1.0)), "residual_queue_bits": float(np.sum(after["raw_queue_bits"])), "return": total_reward, "steps": len(steps), "mean_decision_runtime_ms": 1000.0 * runtime / len(steps), "nodes_expanded": nodes, "actions_evaluated": actions, "total_path_length_m": float(np.sum(after["cumulative_distance_m_per_uav"])), "movement_energy": float(np.sum(after["cumulative_energy_consumption_per_uav"])), "mean_spatial_spread_m": float(np.mean(spread)), "distinct_served_buoys": len(served), "min_uav_distance_m": float(min(row["min_uav_distance_m"] for row in steps)), "safety_correction_rate": float(np.mean([np.any(np.asarray(row["safety_correction_m"]) > 1.0e-9) for row in steps])), "raw_collision_risk_count": int(sum(row["raw_collision_risk_pairs"] for row in steps))}, steps


def _paired(metrics: Sequence[Mapping[str, Any]], left: str, right: str, comparison: str) -> list[dict[str, Any]]:
    by_key = {(row["scenario_id"], row["policy"]): row for row in metrics}
    records = []
    for scenario in planning_scenarios():
        delta = float(by_key[(scenario.scenario_id, left)]["payload_mbit"] - by_key[(scenario.scenario_id, right)]["payload_mbit"])
        records.append({"scenario_id": scenario.scenario_id, "comparison": comparison, "left_policy": left, "right_policy": right, "delta_payload_mbit": delta, "num_buoys": scenario.num_buoys, "coordination_stress": scenario.coordination_stress, "geometry_group": scenario.geometry_group})
    return records


def _summaries(records: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups = [("overall", records), *[(f"K={count}", [row for row in records if row["num_buoys"] == count]) for count in (3, 5)], ("coordination_stress", [row for row in records if row["coordination_stress"]]), ("non_stress", [row for row in records if not row["coordination_stress"]])]
    return [{"comparison": rows[0]["comparison"], "group": label, **_bootstrap([row["delta_payload_mbit"] for row in rows], seed + index)} for index, (label, rows) in enumerate(groups) if rows]


def _assignment_divergence(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in steps: indexed.setdefault((row["scenario_id"], row["policy"]), []).append(row)
    rows = []
    for scenario in planning_scenarios():
        h1 = indexed[(scenario.scenario_id, "joint_h1")]
        for policy in ("joint_h5", "joint_h10"):
            other = indexed[(scenario.scenario_id, policy)]
            disagree = [not np.array_equal(a["assignment"], b["assignment"]) for a, b in zip(h1, other)]
            roles = [np.asarray(item["assignment"], dtype=int) for item in other]
            switches = sum(not np.array_equal(roles[index], roles[index - 1]) for index in range(1, len(roles)))
            visitation = tuple(value for assignment in roles for value in assignment if value >= 0)
            base_visitation = tuple(value for assignment in [np.asarray(item["assignment"], dtype=int) for item in h1] for value in assignment if value >= 0)
            rows.append({"scenario_id": scenario.scenario_id, "comparison": f"{policy}-joint_h1", "assignment_disagreement_rate": float(np.mean(disagree)), "first_target_change": bool(disagree[0]), "role_switch_count": switches, "visitation_order_change": visitation != base_visitation})
    return rows


def _duplicate_audit(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in steps: indexed.setdefault((row["scenario_id"], row["policy"]), []).append(row)
    rows = []
    for scenario in planning_scenarios():
        for policy in ("independent_queue_rate", "joint_h1", "joint_h10"):
            episode = indexed[(scenario.scenario_id, policy)]
            duplicate = [len(row["intended_targets"]) == 2 and row["intended_targets"][0] >= 0 and row["intended_targets"][0] == row["intended_targets"][1] for row in episode]
            future = [sum(item["payload_bits"] for item in episode[index + 1:]) for index in range(len(episode))]
            yes = [future[index] for index, flag in enumerate(duplicate) if flag]; no = [future[index] for index, flag in enumerate(duplicate) if not flag]
            rows.append({"scenario_id": scenario.scenario_id, "policy": policy, "duplicate_target_intent_rate": float(np.mean(duplicate)), "duplicate_steps": int(sum(duplicate)), "future_payload_after_duplicate_bits": float(np.mean(yes)) if yes else 0.0, "future_payload_after_nonduplicate_bits": float(np.mean(no)) if no else 0.0, "future_payload_difference_bits": (float(np.mean(yes)) if yes else 0.0) - (float(np.mean(no)) if no else 0.0)})
    return rows


def _role_rows(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in steps: indexed.setdefault((row["scenario_id"], row["policy"]), []).append(row)
    rows = []
    for scenario in planning_scenarios():
        for policy in POLICIES:
            episode = indexed[(scenario.scenario_id, policy)]
            for uav in range(2):
                served = np.array([row["assignment"][uav] for row in episode], dtype=int); valid = served[served >= 0]
                counts = np.bincount(valid, minlength=scenario.num_buoys).astype(float) if valid.size else np.zeros(scenario.num_buoys)
                share = counts / counts.sum() if counts.sum() else counts
                entropy = -float(np.sum(share[share > 0.0] * np.log(share[share > 0.0]))) / math.log(scenario.num_buoys) if scenario.num_buoys > 1 else 0.0
                persistence = float(np.mean(served[1:] == served[:-1])) if served.size > 1 else 1.0
                rows.append({"scenario_id": scenario.scenario_id, "policy": policy, "uav_index": uav, "served_buoy_distribution": counts.tolist(), "specialization_entropy": entropy, "role_persistence": persistence, "role_switch_count": int(np.count_nonzero(served[1:] != served[:-1]))})
    return rows


def run_audit(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    preflight_data = preflight(root)
    regression = single_uav_regression(); _csv(output / "single_uav_regression_audit.csv", regression)
    scenarios = planning_scenarios(); _json(output / "round18_multi_uav_scenario_manifest.json", {"namespace": "round18_multi_uav_planning_diagnostic", "scenario_count": len(scenarios), "scenarios": scenario_manifest_rows()})
    metrics: list[dict[str, Any]] = []; steps: list[dict[str, Any]] = []
    for scenario in scenarios:
        for policy in POLICIES:
            metric, trajectory = _episode(scenario, policy); metrics.append(metric); steps.extend(trajectory)
    _csv(output / "coordination_headroom_metrics.csv", metrics)
    _csv(output / "multi_uav_trajectory_steps.csv", steps)
    cn_records = _paired(metrics, "joint_h1", "independent_queue_rate", "joint_h1_minus_independent")
    h5_records = _paired(metrics, "joint_h5", "joint_h1", "joint_h5_minus_joint_h1")
    h10_records = _paired(metrics, "joint_h10", "joint_h1", "joint_h10_minus_joint_h1")
    cn_summary = _summaries(cn_records, 18301); plan_summary = _summaries(h5_records, 18311) + _summaries(h10_records, 18321)
    _csv(output / "coordination_paired_comparison.csv", cn_records + cn_summary)
    _csv(output / "multi_uav_planning_paired.csv", h5_records + h10_records)
    _csv(output / "multi_uav_planning_headroom.csv", plan_summary)
    divergence = _assignment_divergence(steps); _csv(output / "multi_uav_assignment_divergence.csv", divergence)
    duplicates = _duplicate_audit(steps); _csv(output / "duplicate_intent_audit.csv", duplicates)
    roles = _role_rows(steps); _csv(output / "role_specialization_audit.csv", roles)
    coverage = [{"scenario_id": row["scenario_id"], "policy": row["policy"], "mean_uav_spatial_spread_m": row["mean_spatial_spread_m"], "distinct_buoy_regions_covered": row["distinct_served_buoys"], "distinct_served_buoy_count": row["distinct_served_buoys"], "clustering_inverse_spread": 1.0 / max(row["mean_spatial_spread_m"], 1.0e-9)} for row in metrics]; _csv(output / "team_spatial_coverage_audit.csv", coverage)
    safety = [{"scenario_id": row["scenario_id"], "policy": row["policy"], "raw_collision_risk_action_count": row["raw_collision_risk_count"], "safety_correction_rate": row["safety_correction_rate"], "minimum_uav_distance_m": row["min_uav_distance_m"]} for row in metrics]; _csv(output / "multi_uav_safety_audit.csv", safety)
    runtime = [{"scenario_id": row["scenario_id"], "policy": row["policy"], "ms_per_decision": row["mean_decision_runtime_ms"], "actions_evaluated": row["actions_evaluated"], "nodes_expanded": row["nodes_expanded"], "memory_estimate_kb": row["nodes_expanded"] * 0.25} for row in metrics]; _csv(output / "multi_uav_runtime_audit.csv", runtime)
    by = {(row["scenario_id"], row["policy"]): row for row in metrics}
    delayed = []
    for scenario in scenarios:
        h1 = [row for row in steps if row["scenario_id"] == scenario.scenario_id and row["policy"] == "joint_h1"][0]
        for policy in ("joint_h5", "joint_h10"):
            candidate = [row for row in steps if row["scenario_id"] == scenario.scenario_id and row["policy"] == policy][0]
            immediate_sacrifice = candidate["payload_bits"] <= h1["payload_bits"] + 1.0e-6
            future_gain = by[(scenario.scenario_id, policy)]["payload_bits"] > by[(scenario.scenario_id, "joint_h1")]["payload_bits"] + 1.0e-6
            delayed.append({"scenario_id": scenario.scenario_id, "comparison": f"{policy}-joint_h1", "immediate_sacrifice": immediate_sacrifice, "future_gain": future_gain, "delayed_cooperative_benefit": immediate_sacrifice and future_gain, "immediate_delta_payload_mbit": (candidate["payload_bits"] - h1["payload_bits"]) / 1.0e6, "future_delta_payload_mbit": (by[(scenario.scenario_id, policy)]["payload_bits"] - by[(scenario.scenario_id, "joint_h1")]["payload_bits"]) / 1.0e6})
    _csv(output / "delayed_cooperative_benefit.csv", delayed)
    cn_overall = next(row for row in cn_summary if row["group"] == "overall")
    h10_overall = next(row for row in plan_summary if row["comparison"] == "joint_h10_minus_joint_h1" and row["group"] == "overall")
    h10_stress = next(row for row in plan_summary if row["comparison"] == "joint_h10_minus_joint_h1" and row["group"] == "coordination_stress")
    h10_k = [row for row in plan_summary if row["comparison"] == "joint_h10_minus_joint_h1" and row["group"] in ("K=3", "K=5")]
    divergence_rate = float(np.mean([row["assignment_disagreement_rate"] for row in divergence if row["comparison"] == "joint_h10-joint_h1"]))
    delayed_ratio = float(np.mean([row["delayed_cooperative_benefit"] for row in delayed if row["comparison"] == "joint_h10-joint_h1"]))
    safety_h10 = float(np.mean([row["safety_correction_rate"] for row in safety if row["policy"] == "joint_h10"])); safety_h1 = float(np.mean([row["safety_correction_rate"] for row in safety if row["policy"] == "joint_h1"]))
    cn_gate = "CN-A" if cn_overall["ci95_low"] > 0.0 else "CN-B" if cn_overall["ci95_high"] > 0.0 else "CN-C"
    mpn_a = h10_overall["ci95_low"] > 0.0 and sum(row["ci95_low"] > 0.0 for row in h10_k) >= 2 and divergence_rate > 0.05 and delayed_ratio > 0.0 and safety_h10 <= safety_h1 + 1.0e-9
    mpn_gate = "MPN-A" if mpn_a else "MPN-B" if h10_stress["ci95_low"] > 0.0 else "MPN-C"
    coupling_gate = "MC-A" if cn_gate == "CN-A" and mpn_gate in ("MPN-A", "MPN-B") else "MC-B" if cn_gate == "CN-A" else "MC-C"
    necessity = {"multi_uav_planning_necessity_gate": mpn_gate, "multi_uav_long_horizon_planning_required": True if mpn_gate == "MPN-A" else "conditional" if mpn_gate == "MPN-B" else False, "joint_h10_minus_joint_h1": h10_overall, "coordination_stress_joint_h10_minus_joint_h1": h10_stress, "assignment_divergence_rate": divergence_rate, "delayed_cooperative_benefit_ratio": delayed_ratio, "safety_correction_rate_h1": safety_h1, "safety_correction_rate_h10": safety_h10}
    coordination_gate = {"multi_uav_coordination_gate": cn_gate, "joint_h1_minus_independent": cn_overall}
    coupling = {"multi_uav_coupling_gate": coupling_gate, "coordination_gate": cn_gate, "planning_gate": mpn_gate}
    _json(output / "multi_uav_coordination_gate.json", coordination_gate); _json(output / "multi_uav_planning_necessity_gate.json", necessity); _json(output / "multi_uav_coupling_gate.json", coupling)
    readiness = {"multi_uav_offline_dataset_ready": mpn_gate == "MPN-A", "recommended_next_step": "generate_multi_uav_mixed_quality_offline_dataset_then_consider_HGNN" if mpn_gate == "MPN-A" else "consider_HGNN_coordination_without_a_strong_waypoint_claim" if mpn_gate == "MPN-C" and cn_gate == "CN-A" else "audit_stronger_coupling_shared_spectrum_interference_or_backhaul" if mpn_gate == "MPN-C" and cn_gate == "CN-C" else "no_training_recommendation_from_this_round", "training_performed": False, "old_development_test_access_count": 0, "fresh_development_access_count": 0, "untouched_holdout_new_access_count": 0, "round19_created": False}
    _json(output / "round18_readiness_gate.json", readiness)
    _json(output / "run_manifest.json", {"scenario_count": len(scenarios), "episode_count": len(metrics), "policies": list(POLICIES), "primary_num_uavs": 2, "primary_buoy_drift_mode": "static", "primary_channel_variation_mode": "static", "preflight": preflight_data})
    (output / "implementation_log.md").write_text("Round-18 deterministic multi-UAV audit. M=1 exact control, M=2 shared queues, exclusive matching, hard separation, and causal H={1,5,10}; no training or protected-data access.\n", encoding="utf-8")
    return {"preflight": preflight_data, "metrics": metrics, "steps": steps,
            "coordination_gate": coordination_gate, "necessity_gate": necessity,
            "coupling_gate": coupling, "readiness": readiness,
            "single_regression": regression, "divergence": divergence,
            "duplicates": duplicates, "roles": roles, "safety": safety,
            "delayed": delayed, "planning_summary": plan_summary}


__all__ = ["POLICIES", "preflight", "run_audit", "single_uav_regression"]
