"""Deterministic frozen-model paired closed-loop evaluation for Round-6.

This module deliberately contains no optimizer, loss, backward pass, or model
checkpoint writer.  It evaluates the Round-5 saved models on new Round-6
scenario snapshots only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

from wt.maritime.baselines import create_baseline_policy
from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.env import MaritimeEnv
from wt.maritime.round3_runner import _build_case_config, _current_policy_info
from wt.maritime.waypoint_samples import (
    OBSERVATION_BASE_DIM, OBSERVATION_ENERGY_INDEX, OBSERVATION_NODE_DIM,
    NODE_COMPLETION_INDEX, NODE_QUEUE_INDEX, padded_observation_dim,
    waypoint_dim,
)
from wt.round4.round4_training import load_spatial_waypoint_checkpoint
from wt.round5.causal_policy import build_policy
from wt.round5.round5_evaluation import assert_finite_tree, radial_action_projection, write_finite_json, write_records_csv

ROOT = Path(__file__).resolve().parents[3]
R5 = ROOT / "round_05" / "outputs" / "round05_formal_canonical_20260803_1500"
R4_CHECKPOINT = ROOT / "round_04" / "outputs" / "round04_formal_canonical_20260802_173717" / "horizon_030" / "goal_conditioned" / "seed_001" / "checkpoints" / "best.pt"
WAYPOINT_SHA = "F92133AACA1FF6F0839EE7C817F40B696F7FD8FB15973A6118D222F83F01368D"
MAX_NODES = 5


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()[:8], "little")


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray): return [_json(x) for x in value.tolist()]
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, Mapping): return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json(v) for v in value]
    return value


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("Round-6 model freeze contract failed")
    return model


def _load_models(device: torch.device) -> tuple[torch.nn.Module, dict[str, dict[int, torch.nn.Module]], list[dict[str, Any]]]:
    if _sha(R4_CHECKPOINT) != WAYPOINT_SHA:
        raise RuntimeError("locked Round-4 waypoint checkpoint hash mismatch")
    waypoint, _ = load_spatial_waypoint_checkpoint(R4_CHECKPOINT, device=device, expected_horizon=30, expected_model_kind="goal_conditioned", expected_input_dim=80)
    waypoint = _freeze(waypoint)
    wanted = {
        ("wt_transformer", 10): "wt_transformer_hc10",
        ("wt_transformer", 1): "wt_transformer_hc1",
        ("direct_goal_transformer", 10): "direct_goal_transformer_hc10",
        ("state_only_transformer", 10): "state_only_transformer_hc10",
        ("mlp_waypoint", 1): "mlp_waypoint",
        ("mlp_state_only", 1): "mlp_state_only",
    }
    rows = list(csv.DictReader((R5 / "transformer_training_summary.csv").open(encoding="utf-8")))
    models: dict[str, dict[int, torch.nn.Module]] = {name: {} for name in wanted.values()}
    manifest: list[dict[str, Any]] = []
    for row in rows:
        key = (row["model_type"], int(row["context_length"]))
        if key not in wanted: continue
        seed = int(row["seed"])
        checkpoint = Path(row["best_checkpoint"])
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = build_policy(payload["model_config"]).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models[wanted[key]][seed] = _freeze(model)
        manifest.append({"model": wanted[key], "model_seed": seed, "checkpoint": str(checkpoint), "checkpoint_sha256": _sha(checkpoint), "model_config": payload["model_config"], "optimizer_created": False})
    for name, by_seed in models.items():
        if sorted(by_seed) != [0, 1, 2]: raise RuntimeError(f"missing frozen Round-5 seeds for {name}: {sorted(by_seed)}")
    return waypoint, models, manifest


def _pad_nodes(values: np.ndarray) -> np.ndarray:
    result = np.zeros(MAX_NODES, dtype=np.float32); result[:len(values)] = values.astype(np.float32); return result


def _state_from_observation(observation: np.ndarray, info: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(info["num_buoys"]); nodes = np.asarray(info["valid_node_mask"], dtype=bool)
    if n > MAX_NODES or observation.shape != (OBSERVATION_BASE_DIM + OBSERVATION_NODE_DIM * n,): raise ValueError("Round-6 observed schema differs from locked schema")
    node_values = observation[OBSERVATION_BASE_DIM:].reshape(n, OBSERVATION_NODE_DIM)
    current = np.zeros(waypoint_dim(MAX_NODES), dtype=np.float32)
    current[:2] = observation[:2]; current[2:2+n] = node_values[:, NODE_QUEUE_INDEX]
    current[2+MAX_NODES:2+MAX_NODES+n] = node_values[:, NODE_COMPLETION_INDEX]; current[-1] = observation[OBSERVATION_ENERGY_INDEX]
    padded_obs = np.zeros(padded_observation_dim(MAX_NODES), dtype=np.float32)
    padded_obs[:OBSERVATION_BASE_DIM] = observation[:OBSERVATION_BASE_DIM]
    padded_obs[OBSERVATION_BASE_DIM:].reshape(MAX_NODES, OBSERVATION_NODE_DIM)[:n] = node_values
    padded = _pad_nodes(nodes)
    return np.concatenate((current, padded_obs, padded)).astype(np.float32), current, padded


def _achieved_goal(info: Mapping[str, Any]) -> np.ndarray:
    n = int(info["num_buoys"]); goal = np.zeros(waypoint_dim(MAX_NODES), dtype=np.float32)
    goal[:2] = np.asarray(info["normalized_position"], dtype=np.float32)
    goal[2:2+n] = np.asarray(info["raw_queue_bits"], dtype=np.float32) / float(info["queue_capacity_bits"])
    goal[2+MAX_NODES:2+MAX_NODES+n] = np.asarray(info["service_completion_ratio"], dtype=np.float32)
    goal[-1] = float(info["remaining_energy_ratio"])
    return goal


def _snapshot(config: MaritimeEnvConfig, scenario_seed: int, trace: np.ndarray, trace_id: str, scenario_id: str) -> dict[str, Any]:
    raw = asdict(config); raw["arrival_trace_bits"] = [[float(x) for x in row] for row in trace]
    raw["initial_queue_bits_by_buoy"] = list(config.initial_queue_bits_by_buoy or ())
    body = {"scenario_id": scenario_id, "scenario_seed": int(scenario_seed), "environment_config": raw, "arrival_trace_id": trace_id}
    body["snapshot_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
    body["trace_sha256"] = hashlib.sha256(np.ascontiguousarray(trace).tobytes()).hexdigest().upper()
    return body


def _reset(snapshot: Mapping[str, Any]) -> tuple[MaritimeEnv, np.ndarray, dict[str, Any]]:
    raw = dict(snapshot["environment_config"]); raw["arrival_trace_bits"] = tuple(tuple(float(x) for x in row) for row in raw["arrival_trace_bits"]); raw["initial_queue_bits_by_buoy"] = tuple(float(x) for x in raw["initial_queue_bits_by_buoy"])
    env = MaritimeEnv(MaritimeEnvConfig(**raw)); obs, info = env.reset(seed=int(snapshot["scenario_seed"]))
    if str(info["arrival_trace_id"]) != str(snapshot["arrival_trace_id"]): raise RuntimeError("snapshot did not restore its trace")
    return env, obs, dict(info)


def _baseline_rollout(snapshot: Mapping[str, Any], name: str) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    env, obs, info = _reset(snapshot); policy_seed = _stable_seed("round06-reference", snapshot["snapshot_sha256"], name)
    policy = create_baseline_policy(name, seed=policy_seed); policy.reset(seed=policy_seed); rows=[]; total=0.0
    while True:
        action = np.asarray(policy.act(obs.copy(), _current_policy_info(info, env)), dtype=np.float64)
        next_obs, reward, terminated, truncated, next_info = env.step(action); total += reward
        rows.append(_step_record(info, next_info, action, action, False, 0.0, 0.0, total, name)); obs, info = next_obs, dict(next_info)
        if terminated or truncated: break
    return _achieved_goal(info), {"return": total, "final_info": info, "steps": rows}, rows


def _step_record(before: Mapping[str, Any], after: Mapping[str, Any], raw: np.ndarray, executed: np.ndarray, projection: bool, waypoint_error: float, action_error: float, cumulative_return: float, model: str) -> dict[str, Any]:
    return {"model": model, "timestep": int(after["time_step"]), "reward": float(cumulative_return), "step_reward": float(cumulative_return if int(after["time_step"]) == 1 else 0.0), "cumulative_payload_bits": float(after["cumulative_payload_bits"]), "residual_queue_ratio": float(after["residual_queue_ratio"]), "raw_action_norm": float(np.linalg.norm(raw)), "executed_action_norm": float(np.linalg.norm(executed)), "projection_activated": bool(projection), "boundary_violation": bool(after.get("boundary_violation", False)), "waypoint_error_m": float(waypoint_error), "action_error_m": float(action_error), "remaining_energy": float(after["remaining_energy"])}


def _policy_action(model: torch.nn.Module, history: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]], condition: np.ndarray, device: torch.device) -> np.ndarray:
    context = int(getattr(model.config, "context_length", 1)); slice_ = history[-context:]; pad = context-len(slice_); state_dim=int(model.config.state_dim); node_dim=int(model.config.node_mask_dim); cond_dim=int(model.config.condition_dim)
    states=np.zeros((1, context, state_dim),np.float32); nodes=np.zeros((1,context,node_dim),np.float32); cond=np.zeros((1,context,cond_dim),np.float32); rem=np.zeros((1,context,1),np.float32); attention=np.zeros((1,context),bool)
    for i,(state,node,_,remain) in enumerate(slice_, start=pad): states[0,i]=state; nodes[0,i]=node; rem[0,i,0]=remain; attention[0,i]=True; cond[0,i]=condition
    with torch.no_grad():
        result=model(states=torch.from_numpy(states).to(device), node_masks=torch.from_numpy(nodes).to(device), conditions=torch.from_numpy(cond).to(device), remaining_time=torch.from_numpy(rem).to(device), attention_mask=torch.from_numpy(attention).to(device))
    raw=result[0,-1].detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(raw)): raise FloatingPointError("policy emitted NaN/Inf")
    return raw


def _neural_rollout(snapshot: Mapping[str, Any], reference_goal: np.ndarray, name: str, model_seed: int, model: torch.nn.Module, waypoint: torch.nn.Module, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env, obs, info = _reset(snapshot); history=[]; rows=[]; total=0.0
    while True:
        state,current,node = _state_from_observation(obs, info); remain=float(info["remaining_horizon_fraction"]); history.append((state,node,current,remain))
        if name in ("wt_transformer_hc10", "wt_transformer_hc1", "mlp_waypoint"):
            with torch.no_grad(): wp=waypoint(torch.from_numpy(np.concatenate((state, reference_goal, node))[None]).to(device))[0].detach().cpu().numpy().astype(np.float64)
            condition=wp
        elif name == "direct_goal_transformer_hc10": condition=reference_goal
        else: condition=np.zeros((0,),dtype=np.float64)
        raw=_policy_action(model, history, condition.astype(np.float32), device); projected=radial_action_projection(raw[None], 10.0); executed=np.asarray(projected.projected_normalized[0],dtype=np.float64)
        next_obs,reward,terminated,truncated,next_info=env.step(executed); total+=float(reward); next_info=dict(next_info)
        # The currently available executed displacement is the only legal
        # one-step action target; the H=30 waypoint error is filled after the
        # episode once the actual future endpoint exists.
        actual_delta=np.asarray(next_info["applied_displacement_m"],dtype=np.float64)
        waypoint_error=0.0
        rows.append(_step_record(info,next_info,raw,executed,bool(projected.activation_mask[0]),waypoint_error,float(np.linalg.norm(executed*10.0-actual_delta)),total,name)); rows[-1]["state_features_json"] = json.dumps(state.tolist()); rows[-1]["predicted_waypoint_normalized_x"] = float(wp[0]) if name in ("wt_transformer_hc10","wt_transformer_hc1","mlp_waypoint") else 0.0; rows[-1]["predicted_waypoint_normalized_y"] = float(wp[1]) if name in ("wt_transformer_hc10","wt_transformer_hc1","mlp_waypoint") else 0.0; rows[-1]["uav_position_before_x_m"] = float(info["uav_position_m"][0]); rows[-1]["uav_position_before_y_m"] = float(info["uav_position_m"][1]); obs,info=next_obs,next_info
        if terminated or truncated: break
    positions=np.asarray([[r["uav_position_before_x_m"],r["uav_position_before_y_m"]] for r in rows],dtype=np.float64)
    next_positions=np.asarray([r["uav_position_before_x_m"] for r in rows],dtype=np.float64)
    for index,row in enumerate(rows):
        if name in ("wt_transformer_hc10","wt_transformer_hc1","mlp_waypoint"):
            endpoint_index=min(index+29,len(rows)-1)
            endpoint=np.asarray(info["uav_position_m"],dtype=np.float64) if endpoint_index==len(rows)-1 else positions[endpoint_index]
            predicted=np.asarray([row["predicted_waypoint_normalized_x"],row["predicted_waypoint_normalized_y"]],dtype=np.float64)*300.0
            row["waypoint_error_m"]=float(np.linalg.norm(predicted-(endpoint-positions[index])))
    metric=_episode_metrics(info,total,rows,reference_goal,name,model_seed,snapshot)
    return metric,rows


def _episode_metrics(info: Mapping[str, Any], total: float, rows: Sequence[Mapping[str,Any]], goal: np.ndarray, name: str, model_seed: int, snapshot: Mapping[str,Any]) -> dict[str,Any]:
    achieved=_achieved_goal(info); n=int(info["num_buoys"]); offered=np.asarray(info["initial_queue_bits"],float)+np.asarray(info["cumulative_arrival_bits_per_buoy"],float); payload=np.asarray(info["cumulative_payload_bits_per_buoy"],float)
    fairness=float((payload.sum()**2)/(n*np.square(payload).sum())) if float(np.square(payload).sum())>0 else 0.0
    return {"scenario_id":snapshot["scenario_id"],"scenario_seed":int(snapshot["scenario_seed"]),"snapshot_sha256":snapshot["snapshot_sha256"],"trace_sha256":snapshot["trace_sha256"],"reference_policy":"","model":name,"model_seed":model_seed,"num_buoys":n,"episode_return":float(total),"cumulative_payload_bits":float(info["cumulative_payload_bits"]),"task_completion_ratio":float(info["task_completion_ratio"]),"final_residual_queue_ratio":float(info["residual_queue_ratio"]),"success":bool(info["task_success"]),"path_length_m":float(info["cumulative_distance_m"]),"movement_energy":float(info["cumulative_energy_consumption"]),"remaining_energy":float(info["remaining_energy"]),"projection_rate":float(np.mean([r["projection_activated"] for r in rows])),"boundary_violation_rate":float(np.mean([r["boundary_violation"] for r in rows])),"invalid_raw_action_rate":float(np.mean([r["raw_action_norm"]>1.0+1e-9 for r in rows])),"episode_length":len(rows),"goal_error_final":float(np.linalg.norm(achieved-goal)),'goal_uav_error':float(np.linalg.norm(achieved[:2]-goal[:2])),'goal_queue_error':float(np.mean(np.abs(achieved[2:2+n]-goal[2:2+n]))),'goal_completion_error':float(np.mean(np.abs(achieved[2+MAX_NODES:2+MAX_NODES+n]-goal[2+MAX_NODES:2+MAX_NODES+n]))),'goal_energy_error':float(abs(achieved[-1]-goal[-1])),'jain_payload_fairness':fairness,"per_buoy_payload_bits":json.dumps(payload.tolist()),"nan_inf":False,"environment_exception":False}


def _bootstrap(values: np.ndarray, seed: int, replicates: int) -> tuple[float,float]:
    rng=np.random.default_rng(seed); means=np.mean(values[rng.integers(0,len(values),(replicates,len(values)))],axis=1); return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def _paired(rows: Sequence[Mapping[str,Any]], bootstrap_seed:int, reps:int) -> list[dict[str,Any]]:
    by={};
    for row in rows:
        if row["model"] not in ("wt_transformer_hc10","direct_goal_transformer_hc10"): continue
        by.setdefault((row["scenario_seed"],row["num_buoys"],row["reference_policy"],row["model_seed"]),{})[row["model"]]=row
    out=[]
    for metric,sign in (("cumulative_payload_bits",1.0),("episode_return",1.0),("goal_error_final",-1.0)):
        values=np.asarray([sign*(float(v["wt_transformer_hc10"][metric])-float(v["direct_goal_transformer_hc10"][metric])) for v in by.values() if len(v)==2],float)
        low,high=_bootstrap(values,_stable_seed(bootstrap_seed,metric),reps); out.append({"comparison":"WT_Hc10_vs_DirectGoal_Hc10","metric":metric,"n_pairs":len(values),"paired_mean":float(values.mean()),"paired_median":float(np.median(values)),"paired_std":float(values.std()),"bootstrap_ci_low":low,"bootstrap_ci_high":high,"win_rate":float(np.mean(values>0)),"tie_rate":float(np.mean(values==0)),"loss_rate":float(np.mean(values<0)),"effect_size_mean_over_std":float(values.mean()/values.std()) if values.std()>0 else 0.0})
    return out


def _summary(rows:Sequence[Mapping[str,Any]])->list[dict[str,Any]]:
    groups={}
    for r in rows: groups.setdefault((r['model'],r['model_seed'],r['reference_policy'],r['num_buoys']),[]).append(r)
    return [{"model":k[0],"model_seed":k[1],"reference_policy":k[2],"num_buoys":k[3],"episodes":len(v),"mean_return":float(np.mean([x['episode_return'] for x in v])),"mean_payload_bits":float(np.mean([x['cumulative_payload_bits'] for x in v])),"mean_goal_error":float(np.mean([x['goal_error_final'] for x in v])),"success_rate":float(np.mean([x['success'] for x in v]))} for k,v in sorted(groups.items())]


def _postprocess(episodes: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Write diagnostics from frozen rollouts; these never feed a policy."""
    goals=[{k:r[k] for k in ("scenario_id","scenario_seed","reference_policy","model","model_seed","num_buoys","goal_error_final","goal_uav_error","goal_queue_error","goal_completion_error","goal_energy_error")} for r in episodes]
    write_records_csv(output/'goal_achievement_metrics.csv',goals)
    by={}
    for row in steps:
        key=(row['model'],row['model_seed'],row['reference_policy'],row['num_buoys']); by.setdefault(key,[]).append(row)
    propagation=[]; ood=[]
    # The reference state statistics are computed from the locked Round-3
    # training split only; no Round-6 scenario is ever inserted into it.
    from wt.round5.round5_data import load_round5_transition_dataset
    train=load_round5_transition_dataset(ROOT/'round_03'/'outputs'/'round03_pilot_canonical_20260730_230000').train
    mean=np.mean(train.states,axis=0); std=np.maximum(np.std(train.states,axis=0),1e-6)
    for key,values in by.items():
        wp=np.asarray([float(x['waypoint_error_m']) for x in values]); act=np.asarray([float(x['action_error_m']) for x in values]); corr=float(np.corrcoef(wp,act)[0,1]) if wp.std()>1e-12 and act.std()>1e-12 else 0.0
        propagation.append({'model':key[0],'model_seed':key[1],'reference_policy':key[2],'num_buoys':key[3],'transition_count':len(values),'waypoint_action_error_correlation':corr,'mean_waypoint_error_m':float(wp.mean()),'mean_action_error_m':float(act.mean())})
        for value in values:
            state=np.asarray(json.loads(value['state_features_json']),dtype=float); score=float(np.sqrt(np.mean(np.square((state-mean)/std))))
            ood.append({'model':key[0],'model_seed':key[1],'reference_policy':key[2],'num_buoys':key[3],'scenario_seed':value['scenario_seed'],'timestep':value['timestep'],'ood_score_normalized_feature_distance':score})
    write_records_csv(output/'waypoint_error_propagation.csv',propagation); write_records_csv(output/'ood_diagnostics.csv',ood)
    pairs={}
    for row in episodes:
        if row['model'] not in ('wt_transformer_hc10','wt_transformer_hc1'): continue
        key=(row['scenario_seed'],row['num_buoys'],row['reference_policy'],row['model_seed']); pairs.setdefault(key,{})[row['model']]=row
    context=[]
    for key,value in pairs.items():
        if len(value)==2: context.append({'scenario_seed':key[0],'num_buoys':key[1],'reference_policy':key[2],'model_seed':key[3],'delta_return_h10_minus_h1':float(value['wt_transformer_hc10']['episode_return']-value['wt_transformer_hc1']['episode_return']),'delta_payload_h10_minus_h1':float(value['wt_transformer_hc10']['cumulative_payload_bits']-value['wt_transformer_hc1']['cumulative_payload_bits']),'delta_goal_error_h1_minus_h10':float(value['wt_transformer_hc1']['goal_error_final']-value['wt_transformer_hc10']['goal_error_final'])})
    write_records_csv(output/'context_history_comparison.csv',context)
    safety=[]
    for key,values in by.items(): safety.append({'model':key[0],'model_seed':key[1],'reference_policy':key[2],'num_buoys':key[3],'transitions':len(values),'projection_activation_rate':float(np.mean([x['projection_activated'] for x in values])),'boundary_clipping_rate':float(np.mean([x['boundary_violation'] for x in values])),'invalid_raw_action_rate':float(np.mean([x['raw_action_norm']>1.0+1e-9 for x in values])),'nan_inf_count':0,'environment_exception_count':0})
    write_records_csv(output/'safety_projection_metrics.csv',safety)


def run_round6_evaluation(*, mode:str, output_dir:Path, config_path:Path) -> Path:
    config=json.loads(config_path.read_text(encoding='utf-8')); seeds=config['smoke_scenario_seeds'] if mode=='smoke' else config['scenario_seeds']; device=_device(); output_dir.mkdir(parents=True,exist_ok=False)
    waypoint,models,model_manifest=_load_models(device); base=MaritimeEnvConfig(); snapshots=[]; goals=[]; episodes=[]; step_rows=[]
    for buoy_count in config['buoy_counts']:
      for seed in seeds:
        case,trace,trace_id=_build_case_config(base,scenario_id=config['scenario_id'],scenario_seed=int(seed),num_buoys=int(buoy_count),horizon=int(config['episode_horizon'])); snap=_snapshot(case,int(seed),trace,trace_id,config['scenario_id']); snapshots.append(snap)
        for ref in config['reference_policies']:
          goal,ref_result,_=_baseline_rollout(snap,ref); goals.append({"scenario_id":snap['scenario_id'],"scenario_seed":seed,"num_buoys":buoy_count,"reference_policy":ref,"reference_goal":goal.tolist(),"reference_return":ref_result['return'],"reference_payload_bits":float(ref_result['final_info']['cumulative_payload_bits']),"snapshot_sha256":snap['snapshot_sha256'],"trace_sha256":snap['trace_sha256']})
          for name,by_seed in models.items():
            for model_seed,model in by_seed.items():
              metric,steps=_neural_rollout(snap,goal,name,model_seed,model,waypoint,device); metric['reference_policy']=ref; episodes.append(metric)
              for step in steps: step.update({"scenario_seed":seed,"num_buoys":buoy_count,"reference_policy":ref,"model_seed":model_seed,"snapshot_sha256":snap['snapshot_sha256']}); step_rows.append(step)
    paired=_paired(episodes,int(config['bootstrap_seed']),int(config['bootstrap_replicates'])); summary=_summary(episodes)
    write_finite_json(output_dir/'closed_loop_scenarios.json',{"schema_version":"round06_scenarios_v1","scenarios":snapshots,"train_validation_test_seeds":[0,1,2,3,4,5,6,7,8,9]})
    write_finite_json(output_dir/'evaluation_goal_manifest.json',{"schema_version":"round06_reference_goal_v1","goals":goals,"frozen_before_tested_rollout":True})
    write_records_csv(output_dir/'closed_loop_episode_metrics.csv',episodes); write_records_csv(output_dir/'closed_loop_summary.csv',summary); write_records_csv(output_dir/'closed_loop_grouped_metrics.csv',summary); write_records_csv(output_dir/'closed_loop_paired_comparison.csv',paired); write_records_csv(output_dir/'closed_loop_step_metrics.csv',step_rows); _postprocess(episodes,step_rows,output_dir); write_finite_json(output_dir/'model_checkpoint_manifest.json',{"waypoint_checkpoint_sha256":WAYPOINT_SHA,"models":model_manifest,"all_eval":True,"all_requires_grad_false":True,"optimizers_created":False})
    write_finite_json(output_dir/'run_config.json',config); write_finite_json(output_dir/'run_manifest.json',{"schema_version":"round06_run_manifest_v1","mode":mode,"training_performed":False,"waypoint_training_performed":False,"transformer_training_performed":False,"closed_loop_evaluation_performed":True,"formal_closed_loop_evaluation":mode=='formal',"scenario_count":len(snapshots),"reference_goal_count":len(goals),"episode_count":len(episodes),"step_count":len(step_rows),"device":str(device),"round5_lock_sha256":_sha(ROOT/'round_05'/'ROUND_LOCK.json')})
    return output_dir


def run_reproducibility_audit(output_dir: Path, *, seeds: Sequence[int] = (1000,1001,1002,1003,1004)) -> Path:
    """Repeat five fixed WT rollouts twice and record exact deterministic checks."""
    device=_device(); waypoint,models,manifest=_load_models(device); base=MaritimeEnvConfig(); records=[]
    before=[m['checkpoint_sha256'] for m in manifest]
    for seed in seeds:
        case,trace,trace_id=_build_case_config(base,scenario_id='closed_loop_eval_v1',scenario_seed=int(seed),num_buoys=3,horizon=128); snap=_snapshot(case,int(seed),trace,trace_id,'closed_loop_eval_v1'); goal,_,_=_baseline_rollout(snap,'max_queue')
        first,first_steps=_neural_rollout(snap,goal,'wt_transformer_hc10',0,models['wt_transformer_hc10'][0],waypoint,device); second,second_steps=_neural_rollout(snap,goal,'wt_transformer_hc10',0,models['wt_transformer_hc10'][0],waypoint,device)
        records.append({'scenario_seed':seed,'num_buoys':3,'reference_policy':'max_queue','model':'wt_transformer_hc10','model_seed':0,'episode_metrics_exact':first==second,'step_count_equal':len(first_steps)==len(second_steps),'step_projection_exact':all(a['projection_activated']==b['projection_activated'] for a,b in zip(first_steps,second_steps)),'return_equal':float(first['episode_return'])==float(second['episode_return']),'payload_equal':float(first['cumulative_payload_bits'])==float(second['cumulative_payload_bits']),'snapshot_sha256':snap['snapshot_sha256']})
    after=[_sha(Path(m['checkpoint'])) for m in manifest]
    for row in records: row['checkpoint_hashes_unchanged']=before==after
    write_records_csv(output_dir/'reproducibility_audit.csv',records); return output_dir/'reproducibility_audit.csv'
