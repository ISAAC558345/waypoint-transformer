"""Deterministic Round-16 drift planning audit; no training inputs or models."""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from wt.maritime.env import MaritimeEnv, reflect_buoy_motion
from wt.round12.contracts import sha256
from wt.round15.planner import EPSILON, _policy_action as _baseline_action, _qre_action_from_arrays, myopic_one_step_greedy

from .planner import _causal_score, _oracle_score, finite_lattice_plan
from .scenarios import DynamicPlanningScenario, materialize_scenario, planning_scenarios, scenario_manifest_rows


SCHEMA = "round16_dynamic_planning_audit_v1"
POLICIES = ("nearest_buoy", "max_queue", "queue_rate_energy", "myopic_one_step_greedy", "causal_H1", "causal_H5", "causal_H10", "causal_H20", "oracle_H1", "oracle_H5", "oracle_H10", "oracle_H20")
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


def _protocol(root: Path) -> dict[str, Any]:
    value = json.loads((root / "round_16/configs/dynamic_planning_protocol.json").read_text(encoding="utf-8"))
    if value.get("horizons") != [1, 5, 10, 20] or value.get("training_performed") is not False: raise RuntimeError("Round-16 protocol mismatch")
    return value


def preflight(root: Path) -> dict[str, Any]:
    current = (root / "CURRENT_ROUND.txt").read_text(encoding="utf-8")
    if not current.startswith("round_16\n") or "status=in_progress" not in current: raise RuntimeError("Round-16 inactive")
    if (root / "round_17").exists(): raise RuntimeError("Round-17 must not exist")
    manifest = json.loads((root / "round_16/round_manifest.json").read_text(encoding="utf-8"))
    if manifest["untouched_holdout_new_access_count"] != 0 or manifest["untouched_holdout_new_accessed"]: raise RuntimeError("new holdout access prohibited")
    for flag in ("training_performed", "planning_revalidation_performed", "hgnn_training_performed", "ppo_training_performed", "multi_uav_enabled"):
        if manifest[flag] is not False: raise RuntimeError(f"invalid initial control: {flag}")
    lock = json.loads((root / "round_15/ROUND_LOCK.json").read_text(encoding="utf-8"))
    output = root / str(lock["formal_output"])
    changed = [p.name for p in (output.iterdir()) if p.is_file() and p.name in lock["output_hashes"] and sha256(p) != lock["output_hashes"][p.name]]
    if changed: raise RuntimeError(f"Round-15 output changed: {changed}")
    return {"round15_lock_sha256": sha256(root / "round_15/ROUND_LOCK.json"), "round15_outputs_unchanged": True, "prior_metadata_only_access_count": 1, "untouched_holdout_new_access_count": 0, "old_development_test_access_count": 0, "round17_created": False}


def _desc(policy: str) -> tuple[str, int]:
    if policy in ("nearest_buoy", "max_queue", "queue_rate_energy", "myopic_one_step_greedy"): return "myopic", 1
    mode, horizon = policy.split("_H"); return mode, int(horizon)


def _act(env: MaritimeEnv, obs: np.ndarray, info: Mapping[str, Any], policy: str):
    if policy in ("nearest_buoy", "max_queue", "queue_rate_energy"): return _baseline_action(policy, obs, info), None
    if policy == "myopic_one_step_greedy": return myopic_one_step_greedy(env, obs, info), None
    mode, horizon = _desc(policy)
    decision = finite_lattice_plan(env, obs, info, horizon=horizon, mode=mode)
    return decision.action, decision


def _vector(value: Any) -> str: return json.dumps(_plain(np.asarray(value, dtype=float)), separators=(",", ":"))

def _angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= EPSILON or nb <= EPSILON: return 0.0 if na <= EPSILON and nb <= EPSILON else 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0))))

def _future_direction(info: Mapping[str, Any], config: Any, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(info["uav_position_m"], dtype=float); queues = np.asarray(info["raw_queue_bits"], dtype=float)
    buoys = np.asarray(info["buoy_positions_m"], dtype=float); velocities = np.asarray(info["buoy_drift_velocity_mps"], dtype=float)
    current = _qre_action_from_arrays(position, queues, buoys, config)
    for _ in range(horizon): buoys, velocities = reflect_buoy_motion(buoys, velocities, config.delta_t, config)
    return current, _qre_action_from_arrays(position, queues, buoys, config)


def _episode(scenario: DynamicPlanningScenario, policy: str):
    env, obs, info = materialize_scenario(scenario); trace=[]; runtime=[]; delayed=[]; anticipation=[]; total=ret=0.0; residual=0.0
    for step in range(env.config.episode_horizon):
        action, decision = _act(env, obs, info, policy)
        if decision is not None:
            runtime.append({"scenario_id":scenario.scenario_id,"base_scenario_id":scenario.base_scenario_id,"drift_tier":scenario.drift_tier,"policy_id":policy,"mode":decision.mode,"horizon":decision.horizon,"step":step,"runtime_seconds":decision.runtime_seconds,"nodes_expanded":decision.nodes_expanded,"oracle_exogenous_knowledge":decision.oracle_exogenous_knowledge,"oracle_future_drift":decision.oracle_future_drift,"deployable":decision.deployable_diagnostic})
        if policy in ("causal_H1","causal_H10","causal_H20"):
            current, future = _future_direction(info, env.config, min(_desc(policy)[1], 20))
            anticipation.append({"scenario_id":scenario.scenario_id,"base_scenario_id":scenario.base_scenario_id,"drift_tier":scenario.drift_tier,"difficulty":scenario.difficulty,"queue_pressure":scenario.queue_pressure,"geometry":scenario.geometry,"policy_id":policy,"step":step,"action_angle_to_current_degrees":_angle(action,current),"action_angle_to_future_degrees":_angle(action,future),"anticipatory_action":_angle(action,future)+1e-9<_angle(action,current),"current_target_direction":_vector(current),"future_target_direction":_vector(future),"action":_vector(action)})
        if policy == "causal_H20":
            h1, _ = _act(env, obs, info, "causal_H1"); rem=min(20,env.config.episode_horizon-step)
            il,_=_causal_score(env,info,action,1); is_,_=_causal_score(env,info,h1,1); fl,_=_oracle_score(env,obs,info,action,rem); fs,_=_oracle_score(env,obs,info,h1,rem)
            delayed.append({"scenario_id":scenario.scenario_id,"base_scenario_id":scenario.base_scenario_id,"drift_tier":scenario.drift_tier,"difficulty":scenario.difficulty,"queue_pressure":scenario.queue_pressure,"geometry":scenario.geometry,"step":step,"delayed_benefit":bool(il<is_-EPSILON and fl>fs+EPSILON),"immediate_sacrifice_bits":max(0.,is_-il),"future_gain_bits":fl-fs,"actions_differ":bool(np.linalg.norm(action-h1)>EPSILON),"oracle_future_drift":True})
        before=np.asarray(info["uav_position_m"],dtype=float); obs,reward,terminated,truncated,after=env.step(action); sent=np.asarray(after["transmitted_payload_bits"],dtype=float); total+=float(sent.sum()); ret+=float(reward); residual=max(residual,float(np.max(np.abs(np.asarray(after["queue_conservation_residual_bits"],dtype=float))))); pos=np.asarray(after["uav_position_m"],dtype=float); buoys=np.asarray(after["buoy_positions_m"],dtype=float)
        trace.append({"scenario_id":scenario.scenario_id,"base_scenario_id":scenario.base_scenario_id,"drift_tier":scenario.drift_tier,"policy_id":policy,"mode":_desc(policy)[0],"horizon":_desc(policy)[1],"step":step,"action":_vector(action),"position_before_m":_vector(before),"position_after_m":_vector(pos),"buoy_positions_m":_vector(buoys),"buoy_drift_velocity_mps":_vector(after["buoy_drift_velocity_mps"]),"nearest_buoy_index":int(np.argmin(np.linalg.norm(buoys-pos,axis=1))),"payload_bits":float(sent.sum()),"raw_queue_bits":_vector(after["raw_queue_bits"]),"queue_conservation_residual_bits":_vector(after["queue_conservation_residual_bits"])})
        info=after
        if terminated or truncated: break
    mode,horizon=_desc(policy)
    return {"scenario_id":scenario.scenario_id,"base_scenario_id":scenario.base_scenario_id,"drift_tier":scenario.drift_tier,"drift_mode":scenario.config().buoy_drift_mode,"drift_speed_mps":float(scenario.config().buoy_drift_speed_fraction*scenario.config().max_speed),"drift_speed_fraction_of_uav_max":scenario.config().buoy_drift_speed_fraction,"difficulty":scenario.difficulty,"num_buoys":scenario.num_buoys,"queue_pressure":scenario.queue_pressure,"geometry":scenario.geometry,"policy_id":policy,"mode":mode,"horizon":horizon,"cumulative_actual_payload_bits":total,"reported_cumulative_payload_bits":float(info["cumulative_payload_bits"]),"final_residual_queue_bits":float(np.asarray(info["raw_queue_bits"],dtype=float).sum()),"episode_return":ret,"cumulative_distance_m":float(info["cumulative_distance_m"]),"cumulative_energy_consumption":float(info["cumulative_energy_consumption"]),"queue_conservation_max_abs_error_bits":residual,"oracle_future_drift":policy.startswith("oracle_"),"deployable":not policy.startswith("oracle_"),"old_development_test_access_count":0,"untouched_holdout_new_access_count":0}, trace,runtime,delayed,anticipation


def _bootstrap(values: Sequence[float], seed: int) -> tuple[float,float,float,float]:
    x=np.asarray(values,dtype=float); mean=float(x.mean()); median=float(np.median(x))
    if len(x)==1:return mean,median,mean,mean
    rng=np.random.default_rng(16016+seed); samples=np.array([x[rng.integers(0,len(x),len(x))].mean() for _ in range(4000)])
    return mean,median,float(np.quantile(samples,.025)),float(np.quantile(samples,.975))


def _summary(paired: Sequence[Mapping[str,Any]]) -> list[dict[str,Any]]:
    result=[]
    for horizon in (5,10,20):
      for tier in sorted({r["drift_tier"] for r in paired}):
       base=[r for r in paired if r["drift_tier"]==tier and int(r["horizon"])==horizon]
       for key in ("overall","difficulty","queue_pressure","geometry","num_buoys"):
        values=["all"] if key=="overall" else sorted({str(r[key]) for r in base})
        for value in values:
         rows=base if key=="overall" else [r for r in base if str(r[key])==value]
         mean,median,lo,hi=_bootstrap([float(r["delta_payload_bits"]) for r in rows],sum(map(ord,f"{tier}{horizon}{key}{value}")))
         result.append({"drift_tier":tier,"horizon":horizon,"group_key":key,"group_value":value,"n":len(rows),"mean_delta_payload_bits":mean,"median_delta_payload_bits":median,"bootstrap_ci_low":lo,"bootstrap_ci_high":hi,"win_count":sum(float(r["delta_payload_bits"])>0 for r in rows),"tie_count":sum(abs(float(r["delta_payload_bits"]))<=EPSILON for r in rows),"loss_count":sum(float(r["delta_payload_bits"])<0 for r in rows),"cohen_d":float(mean/(np.std([float(r["delta_payload_bits"]) for r in rows],ddof=1)+EPSILON)) if len(rows)>1 else 0.0})
    return result


def _analyses(metrics, traces, delayed, anticipation):
    lookup={(r["scenario_id"],r["policy_id"]):r for r in metrics}; paired=[]; horizon_rows=[]; dpg=[]
    for tier in sorted({r["drift_tier"] for r in metrics}):
      for h in HORIZONS:
       for mode in ("causal","oracle"):
        rows=[r for r in metrics if r["drift_tier"]==tier and r["policy_id"]==f"{mode}_H{h}"]
        horizon_rows.append({"drift_tier":tier,"mode":mode,"horizon":h,"n":len(rows),"mean_actual_payload_bits":float(np.mean([r["cumulative_actual_payload_bits"] for r in rows])),"median_actual_payload_bits":float(np.median([r["cumulative_actual_payload_bits"] for r in rows])),"mean_residual_queue_bits":float(np.mean([r["final_residual_queue_bits"] for r in rows])),"oracle_future_drift":mode=="oracle","deployable":mode=="causal"})
      for scenario in [s for s in planning_scenarios() if s.drift_tier==tier]:
       h1=lookup[(scenario.scenario_id,"causal_H1")]
       for h in (5,10,20):
        item=lookup[(scenario.scenario_id,f"causal_H{h}")]
        paired.append({"scenario_id":scenario.scenario_id,"base_scenario_id":scenario.base_scenario_id,"drift_tier":tier,"difficulty":scenario.difficulty,"num_buoys":scenario.num_buoys,"queue_pressure":scenario.queue_pressure,"geometry":scenario.geometry,"horizon":h,"horizon_payload_bits":item["cumulative_actual_payload_bits"],"h1_payload_bits":h1["cumulative_actual_payload_bits"],"delta_payload_bits":float(item["cumulative_actual_payload_bits"])-float(h1["cumulative_actual_payload_bits"])})
    pmap={(r["base_scenario_id"],r["drift_tier"],int(r["horizon"])):r for r in paired}
    for tier in ("low","medium","high","slowly_varying_secondary"):
      for h in (5,10,20):
       rows=[]
       for scenario in [s for s in planning_scenarios() if s.drift_tier==tier]:
        rows.append(float(pmap[(scenario.base_scenario_id,tier,h)]["delta_payload_bits"])-float(pmap[(scenario.base_scenario_id,"static",h)]["delta_payload_bits"]))
       mean,median,lo,hi=_bootstrap(rows,sum(map(ord,tier))+h)
       dpg.append({"drift_tier":tier,"horizon":h,"n":len(rows),"mean_drift_induced_planning_gain_bits":mean,"median_drift_induced_planning_gain_bits":median,"bootstrap_ci_low":lo,"bootstrap_ci_high":hi})
    headroom=_summary(paired)
    delayed_rows=[]
    for tier in sorted({r["drift_tier"] for r in delayed}):
      rows=[r for r in delayed if r["drift_tier"]==tier]; yes=[r for r in rows if r["delayed_benefit"]]
      delayed_rows.append({"drift_tier":tier,"state_count":len(rows),"delayed_benefit_count":len(yes),"delayed_benefit_ratio":len(yes)/len(rows),"mean_immediate_sacrifice_bits":float(np.mean([r["immediate_sacrifice_bits"] for r in yes])) if yes else 0.,"mean_future_gain_bits":float(np.mean([r["future_gain_bits"] for r in yes])) if yes else 0.,"action_difference_ratio":float(np.mean([r["actions_differ"] for r in rows]))})
    ant=[]
    for tier in sorted({r["drift_tier"] for r in anticipation}):
      for policy in ("causal_H1","causal_H10","causal_H20"):
       rows=[r for r in anticipation if r["drift_tier"]==tier and r["policy_id"]==policy]
       ant.append({"drift_tier":tier,"policy_id":policy,"state_count":len(rows),"anticipatory_action_ratio":float(np.mean([r["anticipatory_action"] for r in rows])),"mean_angle_to_current_degrees":float(np.mean([r["action_angle_to_current_degrees"] for r in rows])),"mean_angle_to_future_degrees":float(np.mean([r["action_angle_to_future_degrees"] for r in rows]))})
    indexed=defaultdict(list)
    for r in traces:indexed[(r["scenario_id"],r["policy_id"])].append(r)
    divergence=[]; visit=[]
    for tier in sorted({r["drift_tier"] for r in metrics}):
      for scenario in [s for s in planning_scenarios() if s.drift_tier==tier]:
       ref=sorted(indexed[(scenario.scenario_id,"causal_H1")],key=lambda x:x["step"])
       for h in (10,20):
        cur=sorted(indexed[(scenario.scenario_id,f"causal_H{h}")],key=lambda x:x["step"]); n=min(len(ref),len(cur)); angles=[_angle(np.asarray(json.loads(cur[i]["action"])),np.asarray(json.loads(ref[i]["action"]))) for i in range(n)]; pd=[float(np.linalg.norm(np.asarray(json.loads(cur[i]["position_after_m"]))-np.asarray(json.loads(ref[i]["position_after_m"])))) for i in range(n)]; different=[cur[i]["nearest_buoy_index"]!=ref[i]["nearest_buoy_index"] for i in range(n)]
        divergence.append({"scenario_id":scenario.scenario_id,"drift_tier":tier,"horizon":h,"action_divergence_rate":float(np.mean([x>1e-8 for x in angles])),"mean_spatial_divergence_m":float(np.mean(pd)),"visitation_difference_rate":float(np.mean(different))})
        visit.append({"scenario_id":scenario.scenario_id,"drift_tier":tier,"horizon":h,"first_target_changed":cur[0]["nearest_buoy_index"]!=ref[0]["nearest_buoy_index"],"visitation_order_disagreement_rate":float(np.mean(different))})
    coupling=[]
    for key,name in (("queue_pressure","queue_drift_coupling"),("geometry","geometry_drift_coupling")):
      for tier in PRIMARY_TIERS:
       rows=[r for r in paired if r["drift_tier"]==tier and r["horizon"]==20]
       for val in sorted({r[key] for r in rows}):
        x=[float(r["delta_payload_bits"]) for r in rows if r[key]==val]; mean,med,lo,hi=_bootstrap(x,sum(map(ord,str(val)))+20)
        coupling.append({"audit":name,"drift_tier":tier,"group":val,"n":len(x),"mean_delta_payload_bits":mean,"bootstrap_ci_low":lo,"bootstrap_ci_high":hi})
    oracle=[]
    for tier in PRIMARY_TIERS:
     for h in HORIZONS:
      vals=[float(lookup[(s.scenario_id,f"oracle_H{h}")]["cumulative_actual_payload_bits"])-float(lookup[(s.scenario_id,f"causal_H{h}")]["cumulative_actual_payload_bits"]) for s in planning_scenarios() if s.drift_tier==tier]; mean,med,lo,hi=_bootstrap(vals,h+sum(map(ord,tier))); oracle.append({"drift_tier":tier,"horizon":h,"n":len(vals),"oracle_minus_causal_payload_bits":mean,"bootstrap_ci_low":lo,"bootstrap_ci_high":hi,"oracle_future_drift":True,"deployable":False})
    return paired,headroom,horizon_rows,dpg,delayed_rows,ant,divergence,visit,coupling,oracle


def _static_regression() -> list[dict[str,Any]]:
    rows=[]; actions=(np.zeros(2),np.array([1.,0.]),np.array([0.,-1.]),np.array([.6,.8]))
    for scenario in [s for s in planning_scenarios() if s.drift_tier=="static"]:
      for action in actions:
       base=scenario.config(); reference=MaritimeEnv(replace(base,buoy_drift_mode="static",buoy_drift_speed_fraction=0.0)); candidate=MaritimeEnv(base); oa,ia=reference.reset(seed=scenario.seed); ob,ib=candidate.reset(seed=scenario.seed); exact=np.array_equal(oa,ob) and np.array_equal(ia["buoy_positions_m"],ib["buoy_positions_m"])
       max_error=0.
       for step in range(8):
        oa,ra,ta,xa,ia=reference.step(action); ob,rb,tb,xb,ib=candidate.step(action); max_error=max(max_error,float(np.max(np.abs(oa-ob))),abs(ra-rb),float(np.max(np.abs(ia["buoy_positions_m"]-ib["buoy_positions_m"]))),float(np.max(np.abs(ia["rates_bps"]-ib["rates_bps"]))),float(np.max(np.abs(ia["raw_queue_bits"]-ib["raw_queue_bits"])))); exact=exact and ta==tb and xa==xb and bool(np.allclose(ia["buoy_positions_m"],ib["buoy_positions_m"],atol=1e-12))
       rows.append({"scenario_id":scenario.scenario_id,"action":_vector(action),"steps":8,"exact_static_regression":exact,"max_abs_error":max_error,"static_drift_velocity_is_zero":bool(np.array_equal(ib["buoy_drift_velocity_mps"],np.zeros_like(ib["buoy_drift_velocity_mps"]))),"passed":exact and max_error<=1e-12})
    return rows


def run_audit(root: Path, output: Path) -> dict[str,Any]:
    root,output=Path(root),Path(output); protocol=_protocol(root); controls=preflight(root); output.mkdir(parents=True,exist_ok=False); static=_static_regression()
    if not all(row["passed"] for row in static):raise RuntimeError("static drift regression failed")
    _json(output/"round16_dynamic_scenario_manifest.json",{"schema_version":SCHEMA,"scenarios":scenario_manifest_rows(),"controls":controls,"protocol":protocol})
    metrics=[]; traces=[]; runtimes=[]; delayed=[]; anticipation=[]; started=time.perf_counter()
    for scenario in planning_scenarios():
      for policy in POLICIES:
       m,t,r,d,a=_episode(scenario,policy); metrics.append(m);traces+=t;runtimes+=r;delayed+=d;anticipation+=a
    paired,headroom,horizons,dpg,delayed_summary,ant,div,visit,coupling,oracle=_analyses(metrics,traces,delayed,anticipation)
    _csv(output/"static_drift_regression_audit.csv",static); _csv(output/"dynamic_episode_metrics.csv",metrics); _csv(output/"dynamic_trajectory_steps.csv",traces); _csv(output/"dynamic_runtime_events.csv",runtimes); _csv(output/"dynamic_planning_paired.csv",paired); _csv(output/"dynamic_planning_headroom.csv",headroom); _csv(output/"dynamic_planning_horizon_value.csv",horizons); _csv(output/"drift_induced_planning_gain.csv",dpg); _csv(output/"dynamic_delayed_benefit.csv",delayed_summary); _csv(output/"anticipatory_motion_audit.csv",ant); _csv(output/"target_direction_audit.csv",anticipation); _csv(output/"dynamic_visitation_order.csv",visit); _csv(output/"dynamic_trajectory_divergence.csv",div); _csv(output/"queue_drift_coupling_audit.csv",[r for r in coupling if r["audit"]=="queue_drift_coupling"]); _csv(output/"geometry_drift_coupling_audit.csv",[r for r in coupling if r["audit"]=="geometry_drift_coupling"]); _csv(output/"causal_oracle_drift_gap.csv",oracle)
    overall={(r["drift_tier"],r["horizon"]):r for r in headroom if r["group_key"]=="overall"}; realistic=any(float(overall[(tier,h)]["bootstrap_ci_low"])>0 for tier in ("low","medium") for h in (10,20)); medium_positive=any(float(overall[("medium",h)]["bootstrap_ci_low"])>0 for h in (10,20)); high_positive=any(float(overall[("high",h)]["bootstrap_ci_low"])>0 for h in (10,20)); ant_lift=next(r for r in ant if r["drift_tier"]=="medium" and r["policy_id"]=="causal_H20")["anticipatory_action_ratio"]>next(r for r in ant if r["drift_tier"]=="medium" and r["policy_id"]=="causal_H1")["anticipatory_action_ratio"]
    gate="DPN-A" if medium_positive and realistic and ant_lift else "DPN-B" if high_positive else "DPN-C"; complexity={"realistic_dynamic_planning_headroom":bool(realistic and gate in ("DPN-A","DPN-B")),"only_high_tier_positive":bool(high_positive and not realistic),"low_or_medium_positive":realistic,"interpretation":"A high-only result is not sufficient for realistic dynamic planning headroom."}; gate_body={"schema_version":SCHEMA,"dynamic_planning_necessity_gate":gate,"dynamic_long_horizon_planning_required":True if gate=="DPN-A" else "conditional" if gate=="DPN-B" else False,"criteria":{"medium_positive":medium_positive,"high_positive":high_positive,"anticipatory_lift":ant_lift,"realistic_low_or_medium_positive":realistic},"no_artificial_pass_threshold":True}
    _json(output/"dynamic_planning_necessity_gate.json",gate_body); _json(output/"drift_complexity_gate.json",complexity); _json(output/"round16_readiness_gate.json",{"schema_version":SCHEMA,"dynamic_gate":gate,"round17_created":False,"next_round_authorized":False})
    run={"schema_version":SCHEMA,"created_at_utc":datetime.now(timezone.utc).isoformat(),"scenario_count":len(planning_scenarios()),"episode_count":len(metrics),"static_regression_passed":True,"elapsed_seconds":time.perf_counter()-started,"gate":gate_body,"complexity":complexity,"controls":controls}; _json(output/"run_manifest.json",run); return run


__all__=["run_audit","preflight"]
