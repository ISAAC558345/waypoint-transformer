"""Deterministic Round-17 predictable-channel planning audit; no training."""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from wt.maritime.env import MaritimeEnv
from wt.round15.planner import EPSILON, _policy_action, myopic_one_step_greedy
from wt.round7.round7_runner import radial_projection

from .planner import _causal_score, advance_channel_phase, finite_lattice_plan, predictable_channel_multiplier, predictable_rates
from .scenarios import ChannelPlanningScenario, materialize_scenario, planning_scenarios, scenario_manifest_rows


SCHEMA = "round17_predictable_channel_audit_v1"
POLICIES = (
    "nearest_buoy", "max_queue", "queue_rate_energy", "myopic_one_step_greedy",
    "causal_H1", "causal_H5", "causal_H10", "causal_H20",
    "oracle_H1", "oracle_H5", "oracle_H10", "oracle_H20",
)
HORIZONS = (1, 5, 10, 20)
PRIMARY_TIERS = ("static", "low", "medium", "high")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, np.ndarray): return _plain(value.tolist())
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating):
        if not math.isfinite(float(value)): raise ValueError("NaN/Inf forbidden")
        return float(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


def _json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_plain(data), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows: raise ValueError(f"empty required output: {path.name}")
    fields = sorted({str(k) for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(_plain(row[k]), ensure_ascii=False, separators=(",", ":")) if isinstance(row.get(k), (list, tuple, dict, np.ndarray)) else _plain(row.get(k, "")) for k in fields})


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def preflight(root: Path) -> dict[str, Any]:
    current = (root / "CURRENT_ROUND.txt").read_text(encoding="utf-8")
    if not current.startswith("round_17\n") or "status=in_progress" not in current: raise RuntimeError("Round-17 inactive")
    if (root / "round_18").exists(): raise RuntimeError("Round-18 must not exist")
    manifest = json.loads((root / "round_17/round_manifest.json").read_text(encoding="utf-8"))
    forbidden = ("training_performed", "task_waypoint_training_performed", "transformer_training_performed", "mlp_training_performed", "hgnn_training_performed", "ppo_training_performed", "multi_uav_enabled", "service_action_expanded", "power_control_enabled", "beamforming_enabled", "association_action_enabled", "reward_modified")
    if any(manifest[key] is not False for key in forbidden): raise RuntimeError("Round-17 prohibition mismatch")
    if manifest["primary_buoy_drift_mode"] != "static": raise RuntimeError("primary drift must be static")
    if any(manifest[key] != 0 for key in ("old_development_test_access_count", "fresh_development_access_count", "untouched_holdout_new_access_count")): raise RuntimeError("protected access is prohibited")
    lock = json.loads((root / "round_16/ROUND_LOCK.json").read_text(encoding="utf-8"))
    groups = (("output_hashes", root / lock["formal_output"]), ("figure_hashes", root / lock["figure_directory"]))
    changed = [f"{group}/{name}" for group, base in groups for name, digest in lock[group].items() if _sha(base / name) != digest]
    changed.extend(
        f"source_tree/{name}" for name, digest in lock["source_tree"].items()
        if name.startswith("src/wt/round16/") and _sha(root / name) != digest
    )
    if changed: raise RuntimeError(f"Round-16 protected artifact changed: {changed}")
    return {"round16_lock_sha256": _sha(root / "round_16/ROUND_LOCK.json"), "round16_round_owned_hashes_unchanged": True, "shared_environment_extension_authorized_in_round17": True, "old_development_test_access_count": 0, "fresh_development_access_count": 0, "untouched_holdout_new_access_count": 0, "round18_created": False}


def _max_error(a: Any, b: Any) -> float:
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _static_regression() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actions = (np.zeros(2), np.array((1., 0.)), np.array((0., -1.)), np.array((.6, .8)))
    for scenario in (s for s in planning_scenarios() if s.channel_tier == "static"):
        for action in actions:
            candidate = MaritimeEnv(scenario.config())
            baseline = MaritimeEnv(replace(scenario.config(), channel_initial_phase_by_buoy=None))
            obs_a, info_a = candidate.reset(seed=scenario.seed); obs_b, info_b = baseline.reset(seed=scenario.seed)
            maximum = max(_max_error(obs_a, obs_b), _max_error(info_a["rates_bps"], info_b["rates_bps"]), _max_error(info_a["channel_gains"], info_b["channel_gains"]))
            terminated = truncated = False
            for _ in range(8):
                obs_a, reward_a, term_a, trunc_a, info_a = candidate.step(action)
                obs_b, reward_b, term_b, trunc_b, info_b = baseline.step(action)
                values = (obs_a, info_a["buoy_positions_m"], info_a["rates_bps"], info_a["channel_gains"], info_a["raw_queue_bits"], info_a["transmitted_payload_bits"], np.array((reward_a, float(term_a), float(trunc_a))))
                other = (obs_b, info_b["buoy_positions_m"], info_b["rates_bps"], info_b["channel_gains"], info_b["raw_queue_bits"], info_b["transmitted_payload_bits"], np.array((reward_b, float(term_b), float(trunc_b))))
                maximum = max(maximum, *(_max_error(x, y) for x, y in zip(values, other)))
                terminated, truncated = term_a, trunc_a
                if terminated or truncated: break
            rows.append({"scenario_id": scenario.scenario_id, "action": np.asarray(action, dtype=float).tolist(), "steps": int(info_a["time_step"]), "exact_static_regression": maximum == 0.0, "max_abs_error": maximum, "passed": maximum == 0.0, "primary_buoy_drift_static": candidate.config.buoy_drift_mode == "static", "terminated": bool(terminated), "truncated": bool(truncated)})
    return rows


def _desc(policy: str) -> tuple[str, int]:
    if policy in ("nearest_buoy", "max_queue", "queue_rate_energy", "myopic_one_step_greedy"): return "myopic", 1
    mode, horizon = policy.split("_H"); return mode, int(horizon)


def _act(env: MaritimeEnv, observation: np.ndarray, info: Mapping[str, Any], policy: str):
    if policy in ("nearest_buoy", "max_queue", "queue_rate_energy"): return _policy_action(policy, observation, info), None
    if policy == "myopic_one_step_greedy": return myopic_one_step_greedy(env, observation, info), None
    mode, horizon = _desc(policy); decision = finite_lattice_plan(env, observation, info, horizon=horizon, mode=mode)
    return decision.action, decision


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= EPSILON or nb <= EPSILON: return 0.0 if na <= EPSILON and nb <= EPSILON else 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def _target_state(info: Mapping[str, Any], config: Any, horizon: int) -> tuple[np.ndarray, np.ndarray, int, int, bool, float]:
    position = np.asarray(info["uav_position_m"], dtype=float); buoys = np.asarray(info["buoy_positions_m"], dtype=float)
    phases = np.asarray(info.get("channel_phase_rad", np.zeros(config.num_buoys)), dtype=float)
    current_rates = predictable_rates(position, buoys, phases, config)
    future_phases = phases.copy()
    for _ in range(int(horizon)): future_phases = advance_channel_phase(future_phases, config)
    future_rates = predictable_rates(position, buoys, future_phases, config)
    current_idx, future_idx = int(np.argmax(current_rates)), int(np.argmax(future_rates))
    def direction(index: int) -> np.ndarray:
        delta = np.asarray(buoys[index] - position, dtype=float); norm = float(np.linalg.norm(delta))
        return np.zeros(2, dtype=float) if norm <= EPSILON else radial_projection(delta / norm)
    return direction(current_idx), direction(future_idx), current_idx, future_idx, current_idx != future_idx, float(future_rates[future_idx] - current_rates[future_idx])


def _episode(scenario: ChannelPlanningScenario, policy: str):
    env, obs, info = materialize_scenario(scenario)
    trace: list[dict[str, Any]] = []; runtime: list[dict[str, Any]] = []; delayed: list[dict[str, Any]] = []; anticipation: list[dict[str, Any]] = []; rankings: list[dict[str, Any]] = []
    total = episode_return = residual = 0.0
    for step in range(env.config.episode_horizon):
        action, decision = _act(env, obs, info, policy)
        if decision is not None:
            runtime.append({"scenario_id": scenario.scenario_id, "base_scenario_id": scenario.base_scenario_id, "channel_tier": scenario.channel_tier, "phase_relationship": scenario.phase_relationship, "policy_id": policy, "mode": decision.mode, "horizon": decision.horizon, "step": step, "runtime_seconds": decision.runtime_seconds, "nodes_expanded": decision.nodes_expanded, "oracle_exogenous_knowledge": decision.oracle_exogenous_knowledge, "oracle_future_channel": decision.oracle_future_channel, "deployable": decision.deployable_diagnostic})
        if policy in ("causal_H1", "causal_H10", "causal_H20"):
            _, horizon = _desc(policy); current, future, ci, fi, reversal, future_gain = _target_state(info, env.config, horizon)
            anticipation.append({"scenario_id": scenario.scenario_id, "base_scenario_id": scenario.base_scenario_id, "channel_tier": scenario.channel_tier, "phase_relationship": scenario.phase_relationship, "difficulty": scenario.difficulty, "queue_pressure": scenario.queue_pressure, "geometry": scenario.geometry, "policy_id": policy, "step": step, "horizon": horizon, "current_target_buoy": ci, "future_target_buoy": fi, "action_angle_to_current_degrees": _angle(action, current), "action_angle_to_future_degrees": _angle(action, future), "channel_window_anticipation": _angle(action, future) + 1.0e-9 < _angle(action, current), "lead_time_slots": horizon, "future_target_rate_gain_bps": future_gain})
            rankings.append({"scenario_id": scenario.scenario_id, "base_scenario_id": scenario.base_scenario_id, "channel_tier": scenario.channel_tier, "phase_relationship": scenario.phase_relationship, "policy_id": policy, "step": step, "horizon": horizon, "current_top_buoy": ci, "future_top_buoy": fi, "ranking_reversal": reversal, "current_future_target_action_difference": _angle(current, future) > 1.0e-9})
        if policy == "causal_H20":
            h1, _ = _act(env, obs, info, "causal_H1")
            immediate_long, _ = _causal_score(env, info, action, 1); immediate_h1, _ = _causal_score(env, info, h1, 1)
            long_score, _ = _causal_score(env, info, action, 20); h1_score, _ = _causal_score(env, info, h1, 20)
            delayed.append({"scenario_id": scenario.scenario_id, "base_scenario_id": scenario.base_scenario_id, "channel_tier": scenario.channel_tier, "phase_relationship": scenario.phase_relationship, "difficulty": scenario.difficulty, "queue_pressure": scenario.queue_pressure, "geometry": scenario.geometry, "step": step, "channel_delayed_benefit": bool(immediate_long < immediate_h1 - EPSILON and long_score > h1_score + EPSILON), "immediate_sacrifice_bits": max(0.0, immediate_h1 - immediate_long), "future_gain_bits": long_score - h1_score, "actions_differ": bool(np.linalg.norm(action - h1) > EPSILON)})
        before = np.asarray(info["uav_position_m"], dtype=float); obs, reward, terminated, truncated, after = env.step(action)
        sent = np.asarray(after["transmitted_payload_bits"], dtype=float); total += float(sent.sum()); episode_return += float(reward)
        residual = max(residual, float(np.max(np.abs(np.asarray(after["queue_conservation_residual_bits"], dtype=float)))))
        trace.append({"scenario_id": scenario.scenario_id, "base_scenario_id": scenario.base_scenario_id, "channel_tier": scenario.channel_tier, "phase_relationship": scenario.phase_relationship, "policy_id": policy, "mode": _desc(policy)[0], "horizon": _desc(policy)[1], "step": step, "action": np.asarray(action, dtype=float).tolist(), "position_before_m": before.tolist(), "position_after_m": np.asarray(after["uav_position_m"], dtype=float).tolist(), "channel_phase_rad": np.asarray(after.get("channel_phase_rad", np.zeros(env.num_buoys)), dtype=float).tolist(), "channel_multipliers": np.asarray(after.get("channel_multipliers", np.ones(env.num_buoys)), dtype=float).tolist(), "rates_bps": np.asarray(after["rates_bps"], dtype=float).tolist(), "payload_bits": float(sent.sum()), "raw_queue_bits": np.asarray(after["raw_queue_bits"], dtype=float).tolist(), "queue_conservation_residual_bits": np.asarray(after["queue_conservation_residual_bits"], dtype=float).tolist()})
        info = after
        if terminated or truncated: break
    mode, horizon = _desc(policy)
    return {"scenario_id": scenario.scenario_id, "base_scenario_id": scenario.base_scenario_id, "channel_tier": scenario.channel_tier, "channel_variation_mode": scenario.config().channel_variation_mode, "channel_log_amplitude": scenario.config().channel_log_amplitude, "channel_period_slots": scenario.config().channel_period_slots, "phase_relationship": scenario.phase_relationship, "difficulty": scenario.difficulty, "num_buoys": scenario.num_buoys, "queue_pressure": scenario.queue_pressure, "geometry": scenario.geometry, "primary_buoy_drift_mode": scenario.config().buoy_drift_mode, "policy_id": policy, "mode": mode, "horizon": horizon, "cumulative_actual_payload_bits": total, "reported_cumulative_payload_bits": float(info["cumulative_payload_bits"]), "final_residual_queue_bits": float(np.asarray(info["raw_queue_bits"], dtype=float).sum()), "episode_return": episode_return, "cumulative_distance_m": float(info["cumulative_distance_m"]), "queue_conservation_max_abs_error_bits": residual, "oracle_future_channel": policy.startswith("oracle_"), "deployable": not policy.startswith("oracle_"), "old_development_test_access_count": 0, "fresh_development_access_count": 0, "untouched_holdout_new_access_count": 0}, trace, runtime, delayed, anticipation, rankings


def _bootstrap(values: Sequence[float], seed: int) -> tuple[float, float, float, float]:
    x = np.asarray(values, dtype=float); mean, median = float(x.mean()), float(np.median(x))
    if len(x) == 1: return mean, median, mean, mean
    rng = np.random.default_rng(17017 + seed)
    samples = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)])
    return mean, median, float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def _summary(paired: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for horizon in (5, 10, 20):
        for tier in PRIMARY_TIERS:
            base = [r for r in paired if r["channel_tier"] == tier and int(r["horizon"]) == horizon]
            for key in ("overall", "difficulty", "queue_pressure", "geometry", "num_buoys", "phase_relationship"):
                groups = ["all"] if key == "overall" else sorted({str(r[key]) for r in base})
                for group in groups:
                    rows = base if key == "overall" else [r for r in base if str(r[key]) == group]
                    values = [float(r["delta_payload_bits"]) for r in rows]; mean, median, lo, hi = _bootstrap(values, sum(map(ord, f"{tier}{horizon}{key}{group}")))
                    result.append({"channel_tier": tier, "horizon": horizon, "group_key": key, "group_value": group, "n": len(rows), "mean_delta_payload_bits": mean, "median_delta_payload_bits": median, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "win_count": sum(v > 0 for v in values), "tie_count": sum(abs(v) <= EPSILON for v in values), "loss_count": sum(v < 0 for v in values), "cohen_d": float(mean / (np.std(values, ddof=1) + EPSILON)) if len(values) > 1 else 0.0})
    return result


def _aggregate_binary(rows: Sequence[Mapping[str, Any]], key: str, group: str = "channel_tier") -> list[dict[str, Any]]:
    result=[]
    for value in sorted({str(row[group]) for row in rows}):
        subset=[row for row in rows if str(row[group]) == value]
        result.append({group:value,"state_count":len(subset),"positive_count":sum(bool(r[key]) for r in subset),"ratio":float(np.mean([bool(r[key]) for r in subset])) if subset else 0.0})
    return result


def _analyses(metrics, delayed, anticipation, rankings):
    lookup = {(r["scenario_id"], r["policy_id"]): r for r in metrics}; paired=[]; horizon_rows=[]; dpg=[]
    scenarios = planning_scenarios()
    for tier in PRIMARY_TIERS:
        for horizon in HORIZONS:
            for mode in ("causal", "oracle"):
                rows = [r for r in metrics if r["channel_tier"] == tier and r["policy_id"] == f"{mode}_H{horizon}"]
                horizon_rows.append({"channel_tier":tier,"mode":mode,"horizon":horizon,"n":len(rows),"mean_actual_payload_bits":float(np.mean([r["cumulative_actual_payload_bits"] for r in rows])),"median_actual_payload_bits":float(np.median([r["cumulative_actual_payload_bits"] for r in rows])),"oracle_future_channel":mode=="oracle","deployable":mode=="causal"})
        for scenario in [s for s in scenarios if s.channel_tier == tier]:
            h1=lookup[(scenario.scenario_id,"causal_H1")]
            for horizon in (5,10,20):
                item=lookup[(scenario.scenario_id,f"causal_H{horizon}")]
                paired.append({"scenario_id":scenario.scenario_id,"base_scenario_id":scenario.base_scenario_id,"channel_tier":tier,"phase_relationship":scenario.phase_relationship,"difficulty":scenario.difficulty,"num_buoys":scenario.num_buoys,"queue_pressure":scenario.queue_pressure,"geometry":scenario.geometry,"horizon":horizon,"horizon_payload_bits":item["cumulative_actual_payload_bits"],"h1_payload_bits":h1["cumulative_actual_payload_bits"],"delta_payload_bits":float(item["cumulative_actual_payload_bits"])-float(h1["cumulative_actual_payload_bits"])})
    pmap={(r["base_scenario_id"],r["channel_tier"],int(r["horizon"])):r for r in paired}
    for tier in ("low","medium","high"):
        for horizon in (5,10,20):
            values=[float(pmap[(s.base_scenario_id,tier,horizon)]["delta_payload_bits"])-float(pmap[(s.base_scenario_id,"static",horizon)]["delta_payload_bits"]) for s in scenarios if s.channel_tier==tier]
            mean,median,lo,hi=_bootstrap(values,sum(map(ord,tier))+horizon)
            dpg.append({"channel_tier":tier,"horizon":horizon,"n":len(values),"mean_channel_induced_planning_gain_bits":mean,"median_channel_induced_planning_gain_bits":median,"bootstrap_ci_low":lo,"bootstrap_ci_high":hi})
    headroom=_summary(paired)
    delayed_rows=[]
    for tier in PRIMARY_TIERS:
        rows=[r for r in delayed if r["channel_tier"]==tier]; yes=[r for r in rows if r["channel_delayed_benefit"]]
        delayed_rows.append({"channel_tier":tier,"state_count":len(rows),"delayed_benefit_count":len(yes),"delayed_benefit_ratio":len(yes)/len(rows) if rows else 0.,"mean_immediate_sacrifice_bits":float(np.mean([r["immediate_sacrifice_bits"] for r in yes])) if yes else 0.,"mean_future_gain_bits":float(np.mean([r["future_gain_bits"] for r in yes])) if yes else 0.,"action_difference_ratio":float(np.mean([r["actions_differ"] for r in rows])) if rows else 0.})
    anticipation_rows=[]
    for tier in PRIMARY_TIERS:
        for policy in ("causal_H1","causal_H10","causal_H20"):
            rows=[r for r in anticipation if r["channel_tier"]==tier and r["policy_id"]==policy]
            anticipation_rows.append({"channel_tier":tier,"policy_id":policy,"state_count":len(rows),"anticipation_ratio":float(np.mean([r["channel_window_anticipation"] for r in rows])) if rows else 0.,"mean_lead_time_slots":float(np.mean([r["lead_time_slots"] for r in rows])) if rows else 0.,"mean_future_target_rate_gain_bps":float(np.mean([r["future_target_rate_gain_bps"] for r in rows])) if rows else 0.})
    reversal_rows=[]
    for tier in PRIMARY_TIERS:
        for policy in ("causal_H1","causal_H10","causal_H20"):
            rows=[r for r in rankings if r["channel_tier"]==tier and r["policy_id"]==policy]
            reversal_rows.append({"channel_tier":tier,"policy_id":policy,"state_count":len(rows),"ranking_reversal_rate":float(np.mean([r["ranking_reversal"] for r in rows])) if rows else 0.,"target_switch_rate":float(np.mean([r["current_future_target_action_difference"] for r in rows])) if rows else 0.})
    coupling=[]
    for key in ("queue_pressure","geometry","phase_relationship"):
        for tier in ("low","medium","high"):
            for horizon in (10,20):
                for value in sorted({str(r[key]) for r in paired if r["channel_tier"]==tier}):
                    values=[float(r["delta_payload_bits"]) for r in paired if r["channel_tier"]==tier and int(r["horizon"])==horizon and str(r[key])==value]
                    mean,median,lo,hi=_bootstrap(values,sum(map(ord,f"{key}{tier}{horizon}{value}")))
                    coupling.append({"coupling":key,"channel_tier":tier,"horizon":horizon,"group_value":value,"n":len(values),"mean_delta_payload_bits":mean,"median_delta_payload_bits":median,"bootstrap_ci_low":lo,"bootstrap_ci_high":hi})
    oracle=[]
    for tier in PRIMARY_TIERS:
        for horizon in HORIZONS:
            values=[float(lookup[(s.scenario_id,f"oracle_H{horizon}")]["cumulative_actual_payload_bits"])-float(lookup[(s.scenario_id,f"causal_H{horizon}")]["cumulative_actual_payload_bits"]) for s in scenarios if s.channel_tier==tier]
            mean,median,lo,hi=_bootstrap(values,sum(map(ord,tier))+horizon+99)
            oracle.append({"channel_tier":tier,"horizon":horizon,"n":len(values),"mean_oracle_minus_causal_payload_bits":mean,"median_oracle_minus_causal_payload_bits":median,"bootstrap_ci_low":lo,"bootstrap_ci_high":hi,"oracle_future_channel":True,"deployable":False})
    return paired,horizon_rows,dpg,headroom,delayed_rows,anticipation_rows,reversal_rows,coupling,oracle


def _stable(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(float(r["mean_delta_payload_bits"]) > 0.0 and float(r["bootstrap_ci_low"]) > 0.0 and int(r["win_count"]) > int(r["loss_count"]) for r in rows)


def run_audit(root: Path, output: Path) -> dict[str, Any]:
    started=time.perf_counter(); controls=preflight(root); output.mkdir(parents=True,exist_ok=True)
    static_rows=_static_regression()
    if not all(row["passed"] for row in static_rows): raise RuntimeError("static channel regression failed")
    metrics=[]; traces=[]; runtime=[]; delayed=[]; anticipation=[]; rankings=[]
    for scenario in planning_scenarios():
        for policy in POLICIES:
            episode, trace, timing, delay, anticipate, reversal = _episode(scenario,policy)
            metrics.append(episode); traces.extend(trace); runtime.extend(timing); delayed.extend(delay); anticipation.extend(anticipate); rankings.extend(reversal)
    paired,horizon,dpg,headroom,delayed_rows,anticipation_rows,reversal_rows,coupling,oracle=_analyses(metrics,delayed,anticipation,rankings)
    realistic_rows=[r for r in headroom if r["channel_tier"] in ("low","medium") and r["group_key"]=="overall" and int(r["horizon"]) in (10,20)]
    conflict_rows=[r for r in headroom if r["channel_tier"] in ("low","medium","high") and r["group_key"]=="phase_relationship" and r["group_value"]=="channel_phase_conflict" and int(r["horizon"]) in (10,20)]
    realistic=_stable(realistic_rows); conditional=_stable(conflict_rows)
    gate="CPN-A" if realistic else "CPN-B" if conditional else "CPN-C"
    gate_data={"schema_version":SCHEMA,"channel_planning_necessity_gate":gate,"predictable_channel_long_horizon_planning_required":True if gate=="CPN-A" else "conditional" if gate=="CPN-B" else False,"criteria":{"realistic_low_or_medium_stable_positive":realistic,"phase_conflict_stable_positive":conditional,"no_artificial_pass_threshold":True}}
    realism={"schema_version":SCHEMA,"realistic_channel_planning_headroom":realistic,"only_high_or_unrealistic_positive":bool(_stable([r for r in headroom if r["channel_tier"]=="high" and r["group_key"]=="overall"]) and not realistic),"interpretation":"Only a low or medium pre-registered tier can establish realistic planning headroom."}
    readiness={"schema_version":SCHEMA,"channel_gate":gate,"next_round_authorized":False,"round18_created":False}
    _json(output/"round17_channel_scenario_manifest.json",{"schema_version":SCHEMA,"scenario_count":len(planning_scenarios()),"scenarios":scenario_manifest_rows(),"controls":controls})
    for name, rows in (("static_channel_regression_audit.csv",static_rows),("channel_episode_metrics.csv",metrics),("channel_trajectory_steps.csv",traces),("channel_runtime_events.csv",runtime),("channel_planning_paired.csv",paired),("channel_planning_horizon_value.csv",horizon),("channel_planning_headroom.csv",headroom),("channel_induced_planning_gain.csv",dpg),("channel_window_anticipation_audit.csv",anticipation_rows),("channel_ranking_reversal_audit.csv",reversal_rows),("channel_delayed_benefit.csv",delayed_rows),("queue_channel_coupling_audit.csv",[r for r in coupling if r["coupling"]=="queue_pressure"]),("geometry_channel_coupling_audit.csv",[r for r in coupling if r["coupling"]=="geometry"]),("phase_conflict_coupling_audit.csv",[r for r in coupling if r["coupling"]=="phase_relationship"]),("causal_oracle_channel_gap.csv",oracle)):
        _csv(output/name,rows)
    _json(output/"channel_planning_necessity_gate.json",gate_data); _json(output/"channel_realism_gate.json",realism); _json(output/"round17_readiness_gate.json",readiness)
    manifest={"schema_version":SCHEMA,"created_at_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.perf_counter()-started,"scenario_count":len(planning_scenarios()),"episode_count":len(metrics),"static_regression_passed":True,"controls":controls,"gate":gate_data,"realism":realism,"figures":[f"{index:02d}_{name}.png" for index,name in enumerate(("channel_multiplier_examples","asynchronous_channel_windows","static_regression","payload_vs_horizon_by_tier","delta_payload_by_horizon","channel_induced_planning_gain","ranking_reversal_examples","channel_window_anticipation","channel_delayed_benefit","current_good_future_good_trajectory","queue_channel_coupling","geometry_channel_coupling","three_vs_five_buoys","causal_vs_oracle_channel","runtime_vs_horizon"),1)]}
    _json(output/"run_manifest.json",manifest)
    return {"output":str(output),"metrics":metrics,"paired":paired,"horizon":horizon,"dpg":dpg,"headroom":headroom,"anticipation":anticipation_rows,"reversals":reversal_rows,"delayed":delayed_rows,"coupling":coupling,"oracle":oracle,"gate":gate_data,"realism":realism,"static_rows":static_rows,"run_manifest":manifest}


__all__=["HORIZONS","POLICIES","_static_regression","preflight","run_audit"]
