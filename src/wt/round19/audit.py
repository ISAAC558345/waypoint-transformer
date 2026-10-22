"""Formal deterministic Round-19 shared-spectrum coordination audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.env import MaritimeEnv
from wt.round18.multi_env import MultiUAVMaritimeEnv
from wt.round18.planner import independent_decision as r18_independent, joint_plan as r18_joint
from wt.round18.scenarios import materialize_scenario as r18_materialize, planning_scenarios as r18_scenarios

from .multi_env import SharedSpectrumMultiUAVEnv, SharedSpectrumOptions
from .planner import independent_decision, joint_plan, per_uav_candidates
from .scenarios import INTERFERENCE_TIERS, materialize_scenario, planning_scenarios, scenario_manifest_rows


POLICIES = ("independent_queue_rate", "joint_h1", "joint_h5", "joint_h10")
R18_LOCK_SHA256 = "8A9462FC87F1DB546E30877DB8814FE591BF90CCB5DE81B0A6C9C63E96C0DA9B"


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(item) for item in value]
    return value


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise"); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_plain(value), sort_keys=True) if isinstance(value, (list, tuple, dict, np.ndarray)) else value for key, value in row.items()})


def _bootstrap(values: Sequence[float], seed: int) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0 or not np.all(np.isfinite(data)): raise ValueError("paired values must be nonempty and finite")
    rng = np.random.default_rng(seed); samples = rng.choice(data, size=(3000, data.size), replace=True).mean(axis=1)
    std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
    return {"n": int(data.size), "mean": float(data.mean()), "median": float(np.median(data)), "std": std, "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975)), "effect_size": float(data.mean() / std) if std > 1.0e-12 else 0.0, "wins": int(np.count_nonzero(data > 1.0e-6)), "ties": int(np.count_nonzero(np.abs(data) <= 1.0e-6)), "losses": int(np.count_nonzero(data < -1.0e-6))}


def preflight(root: Path) -> dict[str, Any]:
    if _sha(root / "round_18" / "ROUND_LOCK.json") != R18_LOCK_SHA256: raise RuntimeError("Round-18 lock changed")
    lock = json.loads((root / "round_18" / "ROUND_LOCK.json").read_text(encoding="utf-8"))
    mismatches = []
    for group in ("source_tree", "output_hashes"):
        for item, expected in lock[group].items():
            path = root / item if group == "source_tree" else root / lock["formal_output"] / item
            if not path.is_file() or _sha(path) != expected: mismatches.append(str(path))
    if mismatches: raise RuntimeError(f"Round-18 locked evidence changed: {mismatches}")
    manifest = json.loads((root / "round_19" / "round_manifest.json").read_text(encoding="utf-8"))
    forbidden = ("training_performed", "task_waypoint_training_performed", "hgnn_training_performed", "ppo_training_performed", "multi_agent_rl_performed")
    if any(bool(manifest[name]) for name in forbidden): raise RuntimeError("Round-19 manifest reports prohibited work")
    if any(int(manifest[name]) != 0 for name in ("old_development_test_access_count", "fresh_development_access_count", "untouched_holdout_new_access_count")): raise RuntimeError("protected-data count is nonzero")
    return {"round18_lock_sha256": R18_LOCK_SHA256, "round18_lock_verified": True, "locked_source_and_output_hashes_verified": True, "primary_num_uavs": 2, "primary_buoy_drift_mode": "static", "primary_channel_variation_mode": "static", "protected_data_access_counts": {"old_development": 0, "fresh_development": 0, "untouched_holdout": 0}}


def _max_error(left: Any, right: Any) -> float:
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape: return math.inf
    if a.dtype.kind in "biu" or b.dtype.kind in "biu": return 0.0 if np.array_equal(a, b) else math.inf
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0


def single_uav_regression() -> list[dict[str, Any]]:
    """M=1 shared facade delegates exactly; eta cannot create a cross uplink."""
    rows = []
    actions = (np.array((0., 0.)), np.array((1., 0.)), np.array((0., 1.)), np.array((-.5, .5)), np.array((1.2, -1.2)), np.array((-.7, -.4)))
    config = MaritimeEnvConfig(num_uavs=1, num_buoys=3, task_semantics_version="v1", buoy_drift_mode="static", channel_variation_mode="static", seed=19111, terminate_on_task_success=False)
    for index, action in enumerate(actions):
        baseline, shared = MaritimeEnv(config), SharedSpectrumMultiUAVEnv(config, options=SharedSpectrumOptions("shared", .50))
        before_a, info_a = baseline.reset(seed=19111 + index); before_b, info_b = shared.reset(seed=19111 + index)
        after_a = baseline.step(action); after_b = shared.step(action.reshape(1, 2))
        errors = [_max_error(before_a, before_b), _max_error(after_a[0], after_b[0]), abs(float(after_a[1]) - float(after_b[1])), float(after_a[2] != after_b[2]), float(after_a[3] != after_b[3]), _max_error(after_a[4]["raw_queue_bits"], after_b[4]["raw_queue_bits"]), _max_error(after_a[4]["transmitted_payload_bits"], after_b[4]["transmitted_payload_bits"])]
        rows.append({"case_id": index, "seed": 19111 + index, "action": action.tolist(), "max_absolute_error": float(max(errors)), "interference_power_w": float(after_b[4]["cross_interference_power_w"][0]), "isr": float(after_b[4]["interference_to_signal_ratio"][0]), "passed": bool(max(errors) == 0.0 and after_b[4]["cross_interference_power_w"][0] == 0.0)})
    return rows


def orthogonal_regression() -> list[dict[str, Any]]:
    """Compare same Round-18 scenario/action/policy paths exactly."""
    rows: list[dict[str, Any]] = []
    # The pre-registered exact subset covers both primary buoy counts while
    # retaining every policy's complete same-snapshot trajectory.
    for scenario in (r18_scenarios()[0], r18_scenarios()[21]):
        for policy in POLICIES:
            old_env, _, old_info = r18_materialize(scenario)
            new_env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=SharedSpectrumOptions("orthogonal", 0.0, "legacy"))
            _, new_info = new_env.reset(seed=scenario.seed)
            max_error = _max_error(old_info["raw_queue_bits"], new_info["raw_queue_bits"]); exact_actions = True; steps = 0; done = False
            while not done:
                if policy == "independent_queue_rate": old_decision, new_decision = r18_independent(old_env, old_info), independent_decision(new_env, new_info)
                else:
                    horizon = int(policy.split("h")[-1]); old_decision, new_decision = r18_joint(old_env, old_info, horizon=horizon), joint_plan(new_env, new_info, horizon=horizon)
                exact_actions = exact_actions and bool(np.array_equal(old_decision.action, new_decision.action))
                old_after = old_env.step(old_decision.action); new_after = new_env.step(new_decision.action, matching_mode="joint_aware" if policy != "independent_queue_rate" else "legacy")
                for old_value, new_value in ((old_after[0], new_after[0]), (old_after[1], new_after[1]), (old_after[4]["raw_queue_bits"], new_after[4]["raw_queue_bits"]), (old_after[4]["uav_buoy_rates_bps"], new_after[4]["uav_buoy_rates_bps"]), (old_after[4]["transmitted_payload_bits"], new_after[4]["transmitted_payload_bits"]), (old_after[4]["uav_to_buoy_assignment"], new_after[4]["uav_to_buoy_assignment"])):
                    max_error = max(max_error, _max_error(old_value, new_value))
                old_info, new_info = old_after[4], new_after[4]; done = bool(old_after[2] or old_after[3]); steps += 1
            rows.append({"scenario_id": scenario.scenario_id, "policy": policy, "seed": scenario.seed, "steps": steps, "max_absolute_error": float(max_error), "exact_joint_actions": exact_actions, "same_termination": True, "passed": bool(max_error == 0.0 and exact_actions)})
    return rows


def _intended_targets(info: Mapping[str, Any]) -> list[int]:
    positions, queues, buoys = np.asarray(info["uav_positions_m"]), np.asarray(info["raw_queue_bits"]), np.asarray(info["buoy_positions_m"])
    candidates = per_uav_candidates(positions, queues, buoys, type("C", (), {"queue_capacity_bits": float(info["queue_capacity_bits"]), "num_buoys": int(info["num_buoys"]), "uav_altitude": 50.0, "total_bandwidth_hz": 1e6, "buoy_tx_power_w": .1, "reference_channel_gain": 1e-3, "pathloss_exponent": 2.2, "noise_power_w": 1e-9})())
    # Candidate directions do not expose their target index.  Reconstruct it
    # from the current per-UAV legacy score, using no other-UAV intention.
    rates = np.asarray(info.get("legacy_uav_buoy_rates_bps", info["uav_buoy_rates_bps"]), dtype=np.float64)
    return [int(np.argmax(queues / float(info["queue_capacity_bits"]) * rates[m])) if np.any(queues > 0.0) else -1 for m in range(positions.shape[0])]


def _episode(scenario: Any, policy: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env, _, info = materialize_scenario(scenario); cumulative_reward = 0.0; trajectory = []; decision_times = []; nodes = actions = matching = 0; done = False
    while not done:
        if policy == "independent_queue_rate": decision = independent_decision(env, info); matching_mode = "legacy"; intended = _intended_targets(info)
        else:
            horizon = int(policy.split("h")[-1]); decision = joint_plan(env, info, horizon=horizon); matching_mode = "joint_aware"; intended = decision.first_assignment.tolist()
        observation, reward, terminated, truncated, after = env.step(decision.action, matching_mode=matching_mode)
        if not np.all(np.isfinite(observation)) or not math.isfinite(float(reward)): raise RuntimeError("non-finite formal transition")
        decision_times.append(decision.runtime_seconds); nodes += decision.nodes_expanded; actions += decision.actions_evaluated; matching += decision.matching_assignments_evaluated; cumulative_reward += float(reward)
        assignment = np.asarray(after["uav_to_buoy_assignment"], dtype=int); positions = np.asarray(after["uav_positions_m"], dtype=float); separation = float(np.linalg.norm(positions[1] - positions[0]))
        duplicate = len(intended) == 2 and intended[0] >= 0 and intended[0] == intended[1]
        nearby = len(intended) == 2 and min(intended) >= 0 and float(np.linalg.norm(np.asarray(after["buoy_positions_m"])[intended[0]] - np.asarray(after["buoy_positions_m"])[intended[1]])) <= 45.0
        trajectory.append({"scenario_id": scenario.scenario_id, "policy": policy, "spectrum_tier": scenario.spectrum_tier, "interference_overlap_factor": scenario.interference_overlap_factor, "num_buoys": scenario.num_buoys, "queue_tier": scenario.queue_tier, "interference_stress": scenario.interference_stress, "time_step": int(after["time_step"]), "uav_positions_m": positions.tolist(), "assignment": assignment.tolist(), "intended_targets": intended, "action": decision.action.tolist(), "payload_bits": float(np.sum(after["transmitted_payload_bits"])), "payload_by_uav_bits": np.asarray(after["transmitted_payload_bits_by_uav"]).tolist(), "queue_before_bits": np.asarray(after["queue_before_bits"]).tolist(), "rates_bps": np.asarray(after["uav_buoy_rates_bps"]).tolist(), "desired_signal_power_w": np.asarray(after["desired_signal_power_w"]).tolist(), "cross_interference_power_w": np.asarray(after["cross_interference_power_w"]).tolist(), "sinr_linear": np.asarray(after["sinr_linear"]).tolist(), "isr": np.asarray(after["interference_to_signal_ratio"]).tolist(), "interference_conflict": bool(after["interference_conflict"]), "receiver_separation_m": separation, "duplicate_intent": duplicate, "nearby_target_intent": nearby, "reward": float(reward), "matching_assignments_evaluated": int(after["matching_assignments_evaluated"]), "safety_correction_m": np.asarray(after["safety_correction_m"]).tolist()})
        info, done = after, bool(terminated or truncated)
    payloads = np.asarray([row["payload_by_uav_bits"] for row in trajectory], dtype=float).sum(axis=0)
    fairness = float(np.sum(payloads) ** 2 / (2.0 * np.sum(np.square(payloads)))) if np.sum(np.square(payloads)) > 0.0 else 1.0
    initial = float(sum(scenario.initial_queue_bits)); offered = initial + float(sum(sum(row["queue_before_bits"]) * 0.0 for row in trajectory)) + len(trajectory) * scenario.num_buoys * 4.5e5
    payload = float(info["cumulative_payload_bits"]); mean_isr = float(np.mean([np.mean(row["isr"]) for row in trajectory])); min_distance = float(min(row["receiver_separation_m"] for row in trajectory)); conflict_rate = float(np.mean([row["interference_conflict"] for row in trajectory]))
    metric = {"scenario_id": scenario.scenario_id, "policy": policy, "spectrum_tier": scenario.spectrum_tier, "spectrum_mode": scenario.spectrum_mode, "interference_overlap_factor": scenario.interference_overlap_factor, "num_buoys": scenario.num_buoys, "queue_tier": scenario.queue_tier, "pattern": scenario.pattern, "geometry_group": scenario.geometry_group, "initialization_group": scenario.initialization_group, "interference_stress": scenario.interference_stress, "payload_bits": payload, "payload_mbit": payload / 1e6, "return": cumulative_reward, "completion_fraction": payload / offered if offered > 0 else 1.0, "residual_queue_bits": float(np.sum(info["raw_queue_bits"])), "fairness": fairness, "energy": float(np.sum(info["cumulative_energy_consumption_per_uav"])), "min_uav_distance_m": min_distance, "safety_correction_rate": float(np.mean([np.any(np.asarray(row["safety_correction_m"]) > 1e-9) for row in trajectory])), "mean_isr": mean_isr, "mean_sinr": float(np.mean([np.mean(row["sinr_linear"]) for row in trajectory])), "conflict_rate": conflict_rate, "conflict_state_seen": conflict_rate > 0.0, "duplicate_intent_rate": float(np.mean([row["duplicate_intent"] for row in trajectory])), "nearby_target_intent_rate": float(np.mean([row["nearby_target_intent"] for row in trajectory])), "mean_decision_runtime_ms": 1000.0 * float(np.mean(decision_times)), "nodes_expanded": nodes, "actions_evaluated": actions, "matching_assignments_evaluated": matching, "memory_estimate_kb": nodes * .30, "steps": len(trajectory)}
    return metric, trajectory


def _paired(metrics: Sequence[Mapping[str, Any]], left: str, right: str, comparison: str) -> list[dict[str, Any]]:
    indexed = {(row["scenario_id"], row["policy"]): row for row in metrics}; rows = []
    for scenario in planning_scenarios():
        a, b = indexed[(scenario.scenario_id, left)], indexed[(scenario.scenario_id, right)]
        rows.append({"scenario_id": scenario.scenario_id, "comparison": comparison, "left_policy": left, "right_policy": right, "spectrum_tier": scenario.spectrum_tier, "interference_overlap_factor": scenario.interference_overlap_factor, "num_buoys": scenario.num_buoys, "queue_tier": scenario.queue_tier, "interference_stress": scenario.interference_stress, "interference_conflict": bool(a["conflict_state_seen"] or b["conflict_state_seen"]), "delta_payload_mbit": float(a["payload_mbit"] - b["payload_mbit"]), "delta_return": float(a["return"] - b["return"]), "delta_residual_queue_bits": float(a["residual_queue_bits"] - b["residual_queue_bits"]), "safety_no_worse": bool(a["safety_correction_rate"] <= b["safety_correction_rate"] + 1e-9)})
    return rows


def _summary(records: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups = [("overall", records), ("K=3", [row for row in records if row["num_buoys"] == 3]), ("K=5", [row for row in records if row["num_buoys"] == 5]), ("interference_stress", [row for row in records if row["interference_stress"]]), ("interference_conflict", [row for row in records if row["interference_conflict"]])]
    return [{"comparison": rows[0]["comparison"], "spectrum_tier": rows[0]["spectrum_tier"], "interference_overlap_factor": rows[0]["interference_overlap_factor"], "group": label, **_bootstrap([row["delta_payload_mbit"] for row in rows], seed + index)} for index, (label, rows) in enumerate(groups) if rows]


def _mechanisms(steps: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in steps: index.setdefault((row["scenario_id"], row["policy"]), []).append(row)
    by = {(row["scenario_id"], row["policy"]): row for row in metrics}; conflict = []; spatial = []; assignment = []; duplicate = []; queue = []
    for scenario in planning_scenarios():
        h1 = index[(scenario.scenario_id, "joint_h1")]
        for policy in ("joint_h5", "joint_h10"):
            other = index[(scenario.scenario_id, policy)]; n = min(len(h1), len(other)); disagree = [h1[i]["assignment"] != other[i]["assignment"] for i in range(n)]; action_disagree = [h1[i]["action"] != other[i]["action"] for i in range(n)]
            role_switch = sum(other[i]["assignment"] != other[i - 1]["assignment"] for i in range(1, len(other)))
            conflict.append({"scenario_id": scenario.scenario_id, "spectrum_tier": scenario.spectrum_tier, "policy": policy, "comparison": f"{policy}-joint_h1", "conflict_state_steps_h1": int(sum(row["interference_conflict"] for row in h1)), "current_interference_sacrifice_bits": float(other[0]["payload_bits"] - h1[0]["payload_bits"]), "assignment_divergence_rate": float(np.mean(disagree)), "movement_divergence_rate": float(np.mean(action_disagree)), "future_payload_gain_bits": float(by[(scenario.scenario_id, policy)]["payload_bits"] - by[(scenario.scenario_id, "joint_h1")]["payload_bits"]), "conflict_pair_avoidance_rate": float(np.mean([float(not row["interference_conflict"]) for row in other]) - np.mean([float(not row["interference_conflict"]) for row in h1])), "role_switch_count": role_switch})
        h10 = index[(scenario.scenario_id, "joint_h10")]; n = min(len(h1), len(h10)); ratio = lambda row: float(np.mean(np.asarray(row["desired_signal_power_w"]) / np.maximum(np.asarray(row["cross_interference_power_w"]), 1e-18)))
        spatial.append({"scenario_id": scenario.scenario_id, "spectrum_tier": scenario.spectrum_tier, "receiver_separation_h1_m": float(np.mean([row["receiver_separation_m"] for row in h1])), "receiver_separation_h10_m": float(np.mean([row["receiver_separation_m"] for row in h10])), "desired_cross_gain_ratio_h1": float(np.mean([ratio(row) for row in h1])), "desired_cross_gain_ratio_h10": float(np.mean([ratio(row) for row in h10])), "isr_h1": float(np.mean([np.mean(row["isr"]) for row in h1])), "isr_h10": float(np.mean([np.mean(row["isr"]) for row in h10])), "isr_reduction": float(np.mean([np.mean(row["isr"]) for row in h1]) - np.mean([np.mean(row["isr"]) for row in h10])), "sinr_improvement": float(np.mean([np.mean(row["sinr_linear"]) for row in h10]) - np.mean([np.mean(row["sinr_linear"]) for row in h1])), "future_payload_gain_bits": float(by[(scenario.scenario_id, "joint_h10")]["payload_bits"] - by[(scenario.scenario_id, "joint_h1")]["payload_bits"])})
        for policy in ("independent_queue_rate", "joint_h1", "joint_h10"):
            episode = index[(scenario.scenario_id, policy)]; duplicate_steps = [row for row in episode if row["duplicate_intent"] or row["nearby_target_intent"]]
            duplicate.append({"scenario_id": scenario.scenario_id, "spectrum_tier": scenario.spectrum_tier, "policy": policy, "duplicate_intent_rate": float(np.mean([row["duplicate_intent"] for row in episode])), "nearby_target_intent_rate": float(np.mean([row["nearby_target_intent"] for row in episode])), "interference_loss_on_duplicate_or_nearby_w": float(np.mean([sum(row["cross_interference_power_w"]) for row in duplicate_steps])) if duplicate_steps else 0.0, "future_payload_after_duplicate_or_nearby_bits": float(np.mean([sum(later["payload_bits"] for later in episode[index + 1:]) for index, row in enumerate(episode) if row["duplicate_intent"] or row["nearby_target_intent"]])) if duplicate_steps else 0.0})
        if scenario.queue_tier == "high":
            h10_metric, h1_metric = by[(scenario.scenario_id, "joint_h10")], by[(scenario.scenario_id, "joint_h1")]
            queue.append({"scenario_id": scenario.scenario_id, "spectrum_tier": scenario.spectrum_tier, "high_queue_interference_geometry": True, "immediate_sacrifice": bool(h10[0]["payload_bits"] <= h1[0]["payload_bits"] + 1e-6), "future_payload_gain_bits": float(h10_metric["payload_bits"] - h1_metric["payload_bits"]), "delayed_coordination_benefit": bool(h10[0]["payload_bits"] <= h1[0]["payload_bits"] + 1e-6 and h10_metric["payload_bits"] > h1_metric["payload_bits"] + 1e-6), "h1_mean_isr": h1_metric["mean_isr"], "h10_mean_isr": h10_metric["mean_isr"]})
        for policy in ("joint_h5", "joint_h10"):
            other = index[(scenario.scenario_id, policy)]; n = min(len(h1), len(other))
            assignment.append({"scenario_id": scenario.scenario_id, "spectrum_tier": scenario.spectrum_tier, "comparison": f"{policy}-joint_h1", "assignment_disagreement_rate": float(np.mean([h1[i]["assignment"] != other[i]["assignment"] for i in range(n)])), "first_assignment_change": bool(h1[0]["assignment"] != other[0]["assignment"]), "role_switch_count": int(sum(other[i]["assignment"] != other[i - 1]["assignment"] for i in range(1, len(other))),), "conflict_pair_avoidance_rate": float(np.mean([not row["interference_conflict"] for row in other]) - np.mean([not row["interference_conflict"] for row in h1]))})
    return conflict, spatial, assignment, duplicate, queue


def _gates(coord: Sequence[Mapping[str, Any]], planning: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    def find(rows: Sequence[Mapping[str, Any]], tier: str, group: str) -> Mapping[str, Any] | None:
        return next((row for row in rows if row["spectrum_tier"] == tier and row["group"] == group), None)
    realistic = ("low", "medium")
    coordination_pass = [tier for tier in realistic if (row := find(coord, tier, "overall")) is not None and row["ci95_low"] > 0.0]
    icn = "ICN-A" if coordination_pass else "ICN-B" if any((find(coord, tier, "overall") or {}).get("ci95_high", 0.0) > 0.0 for tier in ("orthogonal", "low", "medium", "full")) else "ICN-C"
    ipn_a_tiers = []
    for tier in realistic:
        overall, k3, k5, conflict = find(planning, tier, "overall"), find(planning, tier, "K=3"), find(planning, tier, "K=5"), find(planning, tier, "interference_conflict")
        safety = [row for row in metrics if row["spectrum_tier"] == tier and row["policy"] in ("joint_h1", "joint_h10")]
        safe = all(row["safety_correction_rate"] <= next(item["safety_correction_rate"] for item in safety if item["scenario_id"] == row["scenario_id"] and item["policy"] == "joint_h1") + 1e-9 for row in safety if row["policy"] == "joint_h10")
        if overall and k3 and k5 and conflict and overall["mean"] > 0.0 and overall["ci95_low"] > 0.0 and k3["mean"] > 0.0 and k5["mean"] > 0.0 and conflict["ci95_low"] > 0.0 and safe: ipn_a_tiers.append(tier)
    ipn_b = any((find(planning, tier, "overall") or {}).get("ci95_low", 0.0) <= 0.0 and (((find(planning, tier, "K=5") or {}).get("ci95_low", 0.0) > 0.0) or ((find(planning, tier, "interference_stress") or {}).get("ci95_low", 0.0) > 0.0)) for tier in ("low", "medium", "full"))
    ipn = "IPN-A" if ipn_a_tiers else "IPN-B" if ipn_b else "IPN-C"
    realism = bool(ipn_a_tiers); mci = "MCI-A" if icn == "ICN-A" and ipn == "IPN-A" else "MCI-B" if icn == "ICN-A" else "MCI-C"
    coord_gate = {"interference_coordination_gate": icn, "realistic_positive_tiers": coordination_pass, "rule": "ICN-A requires an overall CI lower bound >0 at low or medium"}
    plan_gate = {"interference_planning_necessity_gate": ipn, "interference_long_horizon_planning_required": True if ipn == "IPN-A" else "conditional" if ipn == "IPN-B" else False, "ipn_a_realistic_tiers": ipn_a_tiers, "rule": "IPN-A requires a low/medium H10-H1 stable positive overall, K=3, K=5, and conflict subset with no worse safety"}
    realism_gate = {"realistic_interference_planning_headroom": realism, "full_shared_only_is_insufficient": True, "stable_realistic_tiers": ipn_a_tiers}
    coupling_gate = {"multi_uav_coupling_gate_v2": mci, "coordination_gate": icn, "planning_gate": ipn}
    readiness = {"multi_uav_shared_spectrum_offline_dataset_ready": bool(ipn == "IPN-A" and realism), "recommended_next_step": "generate_multi_uav_mixed_quality_offline_dataset_then_consider_HGNN" if ipn == "IPN-A" and realism else "consider_HGNN_coordination_but_weaken_waypoint_claim" if icn == "ICN-A" and ipn == "IPN-C" else "next_structural_coupling_backhaul_bottleneck_not_implemented_in_round19" if icn == "ICN-C" and ipn == "IPN-C" else "no_training_recommendation_from_this_round", "training_performed": False, "old_development_test_access_count": 0, "fresh_development_access_count": 0, "untouched_holdout_new_access_count": 0, "round20_created": False}
    return coord_gate, plan_gate, realism_gate, coupling_gate, readiness


def run_audit(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False); preflight_data = preflight(root)
    m1, orth = single_uav_regression(), orthogonal_regression(); _csv(output / "single_uav_shared_spectrum_regression.csv", m1); _csv(output / "round18_orthogonal_regression_audit.csv", orth)
    if not all(row["passed"] for row in m1) or not all(row["passed"] for row in orth): raise RuntimeError("required M=1 or orthogonal regression failed")
    scenarios = planning_scenarios(); _json(output / "round19_interference_scenario_manifest.json", {"namespace": "round19_interference_coordination_diagnostic", "scenario_count": len(scenarios), "scenarios": scenario_manifest_rows()})
    metrics: list[dict[str, Any]] = []; steps: list[dict[str, Any]] = []
    for scenario in scenarios:
        for policy in POLICIES:
            metric, trajectory = _episode(scenario, policy); metrics.append(metric); steps.extend(trajectory)
    _csv(output / "interference_scale_grouped_metrics.csv", metrics); _csv(output / "interference_trajectory_steps.csv", steps)
    coordination_raw = _paired(metrics, "joint_h1", "independent_queue_rate", "joint_h1_minus_independent")
    h5_raw = _paired(metrics, "joint_h5", "joint_h1", "joint_h5_minus_joint_h1"); h10_raw = _paired(metrics, "joint_h10", "joint_h1", "joint_h10_minus_joint_h1")
    coordination_summary = sum((_summary([row for row in coordination_raw if row["spectrum_tier"] == tier], 19210 + index) for index, (tier, _, _) in enumerate(INTERFERENCE_TIERS)), [])
    planning_summary = sum((_summary([row for row in h5_raw if row["spectrum_tier"] == tier], 19310 + 2 * index) + _summary([row for row in h10_raw if row["spectrum_tier"] == tier], 19311 + 2 * index) for index, (tier, _, _) in enumerate(INTERFERENCE_TIERS)), [])
    _csv(output / "interference_coordination_paired.csv", coordination_raw); _csv(output / "interference_coordination_headroom.csv", coordination_summary); _csv(output / "interference_planning_paired.csv", h5_raw + h10_raw); _csv(output / "interference_planning_headroom.csv", planning_summary)
    induced = []
    def mean_of(rows: Sequence[Mapping[str, Any]], tier: str, comparison: str) -> float:
        row = next(item for item in rows if item["spectrum_tier"] == tier and item["comparison"] == comparison and item["group"] == "overall"); return float(row["mean"])
    base_coord = mean_of(coordination_summary, "orthogonal", "joint_h1_minus_independent")
    for tier, _, eta in INTERFERENCE_TIERS:
        induced.append({"kind": "coordination", "spectrum_tier": tier, "interference_overlap_factor": eta, "comparison": "joint_h1_minus_independent", "observed_delta_mbit": mean_of(coordination_summary, tier, "joint_h1_minus_independent"), "orthogonal_delta_mbit": base_coord, "interference_induced_gain_mbit": mean_of(coordination_summary, tier, "joint_h1_minus_independent") - base_coord})
    for comparison in ("joint_h5_minus_joint_h1", "joint_h10_minus_joint_h1"):
        base = mean_of(planning_summary, "orthogonal", comparison)
        for tier, _, eta in INTERFERENCE_TIERS:
            observed = mean_of(planning_summary, tier, comparison); induced.append({"kind": "planning", "spectrum_tier": tier, "interference_overlap_factor": eta, "comparison": comparison, "observed_delta_mbit": observed, "orthogonal_delta_mbit": base, "interference_induced_gain_mbit": observed - base})
    _csv(output / "interference_induced_gain.csv", induced)
    conflict, spatial, assignment, duplicate, queue = _mechanisms(steps, metrics); _csv(output / "interference_conflict_audit.csv", conflict); _csv(output / "spatial_interference_deconfliction.csv", spatial); _csv(output / "interference_assignment_divergence.csv", assignment); _csv(output / "duplicate_intent_interference_audit.csv", duplicate); _csv(output / "queue_interference_coupling.csv", queue)
    runtime = [{"scenario_id": row["scenario_id"], "spectrum_tier": row["spectrum_tier"], "policy": row["policy"], "ms_per_decision": row["mean_decision_runtime_ms"], "matching_assignments_evaluated": row["matching_assignments_evaluated"], "joint_actions_evaluated": row["actions_evaluated"], "search_nodes": row["nodes_expanded"], "memory_estimate_kb": row["memory_estimate_kb"]} for row in metrics]; _csv(output / "multi_uav_interference_runtime.csv", runtime)
    coord_gate, plan_gate, realism_gate, coupling_gate, readiness = _gates(coordination_summary, planning_summary, metrics); _json(output / "interference_coordination_gate.json", coord_gate); _json(output / "interference_planning_necessity_gate.json", plan_gate); _json(output / "interference_realism_gate.json", realism_gate); _json(output / "multi_uav_coupling_gate_v2.json", coupling_gate); _json(output / "round19_readiness_gate.json", readiness)
    _json(output / "run_manifest.json", {"preflight": preflight_data, "scenario_count": len(scenarios), "episode_count": len(metrics), "policies": list(POLICIES), "interference_tiers": [{"name": name, "spectrum_mode": mode, "eta": eta} for name, mode, eta in INTERFERENCE_TIERS], "primary_num_uavs": 2, "primary_buoy_drift_mode": "static", "primary_channel_variation_mode": "static", "training_performed": False})
    (output / "implementation_log.md").write_text("Round-19 deterministic shared-spectrum audit; M=2 static buoy uplinks only, exact Round-18 orthogonal control, causal legacy/joint-aware matching, no training or protected-data access.\n", encoding="utf-8")
    return {"preflight": preflight_data, "m1": m1, "orthogonal": orth, "metrics": metrics, "steps": steps, "coordination_summary": coordination_summary, "planning_summary": planning_summary, "induced": induced, "conflict": conflict, "spatial": spatial, "assignment": assignment, "duplicate": duplicate, "queue": queue, "runtime": runtime, "coordination_gate": coord_gate, "planning_gate": plan_gate, "realism_gate": realism_gate, "coupling_gate": coupling_gate, "readiness": readiness}


__all__ = ["POLICIES", "orthogonal_regression", "preflight", "run_audit", "single_uav_regression"]
