"""No-learning Round-7 Gate-B audit and lawful maritime dataset-v2 builder."""
from __future__ import annotations
import csv, hashlib, json, math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np
from wt.maritime.baselines import create_baseline_policy
from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.dataset_v1 import save_dataset_v1, validate_dataset_v1
from wt.maritime.env import MaritimeEnv
from wt.maritime.round3_runner import _build_case_config, _current_policy_info
from wt.maritime.trajectory_v1 import MaritimeEpisodeBuilderV1, MaritimeEpisodeV1

SCHEMA="round07_data_audit_v1"; DATASET="maritime_offline_dataset_v2"; GOAL="goal_schema_v2_candidate"; WPSCHEMA="waypoint_sample_schema_v2"
HORIZONS=(5,10,20,30); NOISE=(.10,.25,.50)
def _seed(*x:Any)->int:return int.from_bytes(hashlib.sha256("\x1f".join(map(str,x)).encode()).digest()[:8],"little")
def _plain(x:Any)->Any:
    if isinstance(x,Mapping):return {str(k):_plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_plain(v) for v in x]
    if isinstance(x,np.ndarray):return _plain(x.tolist())
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,)):return float(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    return x
def _json(path:Path,x:Mapping[str,Any])->None:
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(_plain(x),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def _csv(path:Path,rows:Sequence[Mapping[str,Any]])->None:
    if path.exists():raise FileExistsError(path)
    keys=sorted({str(k) for r in rows for k in r});path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(_plain(r[k]),ensure_ascii=False,sort_keys=True) if isinstance(r.get(k),(list,tuple,dict,np.ndarray)) else r.get(k,"") for k in keys})
def radial_projection(action:np.ndarray)->np.ndarray:
    x=np.clip(np.asarray(action,dtype=float).reshape(2),-1,1)
    if not np.all(np.isfinite(x)):raise ValueError("nonfinite action")
    return x/max(1.,float(np.linalg.norm(x)))
def _specs()->List[Dict[str,Any]]:
    out=[{"behavior_generator":x,"base_policy":x,"noise_level":0.,"switch_schedule":"none"} for x in ("nearest_buoy","max_queue","queue_rate_energy","random_valid")]
    for n in NOISE:out += [{"behavior_generator":"noisy_nearest","base_policy":"nearest_buoy","noise_level":n,"switch_schedule":"none"},{"behavior_generator":"noisy_queue_rate_energy","base_policy":"queue_rate_energy","noise_level":n,"switch_schedule":"none"}]
    return out+[{"behavior_generator":"policy_switch","base_policy":"mixed","noise_level":0.,"switch_schedule":"cyclic_8_steps"},{"behavior_generator":"delayed_commitment","base_policy":"nearest_buoy","noise_level":0.,"switch_schedule":"hold_previous_3_steps"}]
def _case(seed:int,n:int,horizon:int)->Tuple[str,MaritimeEnvConfig,np.ndarray,str,Dict[str,float]]:
    rng=np.random.default_rng(_seed("r7case",seed,n));mode=seed%3
    start=(100.,100.) if mode==0 else (tuple(rng.uniform(30,170,2)) if mode==1 else ((25.,25.) if (seed//3)%2==0 else (175.,175.)))
    low,high=((.42,.67),(.58,.82),(.72,.94))[mode];sid=f"{DATASET}__seed-{seed:05d}__buoys-{n}"
    base=MaritimeEnvConfig(initial_uav_x=float(start[0]),initial_uav_y=float(start[1]),seed=seed);cfg,trace,tid=_build_case_config(base,scenario_id=sid,scenario_seed=seed,num_buoys=n,horizon=horizon)
    q=np.clip(rng.uniform(low,high,n)+np.linspace(0,.03,n),0,.96);cfg=replace(cfg,initial_queue_bits_by_buoy=tuple(float(v*cfg.queue_capacity_bits) for v in q))
    return sid,cfg,trace,tid,{"scenario_mode":mode,"initial_uav_x_m":float(start[0]),"initial_uav_y_m":float(start[1]),"initial_queue_fraction_mean":float(q.mean())}
def _rollout(cfg:MaritimeEnvConfig,sid:str,seed:int,tid:str,spec:Mapping[str,Any],ordinal:int)->Tuple[MaritimeEpisodeV1,Dict[str,Any]]:
    bseed=_seed("r7behavior",sid,seed,spec["behavior_generator"],spec["noise_level"],ordinal);env=MaritimeEnv(cfg);obs,info=env.reset(seed=seed)
    if info["arrival_trace_id"]!=tid:raise RuntimeError("trace mismatch")
    policies={k:create_baseline_policy(k,seed=_seed(bseed,k)) for k in ("nearest_buoy","max_queue","queue_rate_energy","random_valid")}
    for p in policies.values():p.reset(seed=p.seed)
    eid=f"{sid}__{spec['behavior_generator']}__noise-{float(spec['noise_level']):.2f}__rep-{ordinal}";b=MaritimeEpisodeBuilderV1(eid,str(spec["behavior_generator"]),seed,sid,env.num_buoys,info,observation_schema=env.observation_schema,arrival_trace_id=tid,desired_goal=info["desired_goal"],area_width=cfg.area_width,area_height=cfg.area_height,initial_energy=cfg.initial_energy,queue_capacity_bits=cfg.queue_capacity_bits);rng=np.random.default_rng(bseed);last=np.zeros(2)
    while True:
        t=env.time_step;g=str(spec["behavior_generator"]);name=("nearest_buoy","max_queue","queue_rate_energy","random_valid")[(t//8)%4] if g=="policy_switch" else str(spec["base_policy"]);p=policies[name];act=np.asarray(p.act(obs.copy(),_current_policy_info(info,env)),float)
        if g.startswith("noisy_"):act += rng.normal(0,float(spec["noise_level"]),2)
        if g=="delayed_commitment" and t%4:act=.45*last
        act=radial_projection(act);selected=p.metadata["last_decision"].get("selected_buoy_index");nextobs,reward,term,trunc,nextinfo=env.step(act);b.append(obs,act,reward,nextobs,term,trunc,info_before=info,info_after=nextinfo,policy_selected_buoy_index=-1 if selected is None else int(selected));obs,info,last=nextobs,nextinfo,act
        if term or trunc:break
    ep=b.finalize(info["terminal_reason"]);return ep,{**dict(spec),"behavior_seed":bseed,"episode_id":eid,"scenario_id":sid,"scenario_seed":seed,"num_buoys":cfg.num_buoys,"arrival_trace_id":tid,"uses_future_information":False,"action_projection":"common_radial_projection_before_env_step"}

def _row(ep:MaritimeEpisodeV1,m:Mapping[str,Any])->Dict[str,Any]:
    return {**dict(m),"episode_return":float(ep.episode_return),"payload_bits":float(ep.next_cumulative_payload_bits[-1]),"completion_ratio":float(ep.next_task_completion_ratio[-1]),"success":bool(ep.terminal_reason.startswith("task_success")),"constraint_violations":int(ep.boundary_violation.sum()),"action_adjustments":int(ep.action_was_adjusted.sum()),"episode_length":int(ep.episode_length),"terminal_reason":ep.terminal_reason}
def _geo(ep:MaritimeEpisodeV1)->Dict[str,float]:
    u,b=ep.uav_positions_m[0],ep.buoy_positions_m[0];d=np.linalg.norm(b-u,axis=1);p=np.linalg.norm(b[:,None]-b[None,:],axis=2);upper=p[np.triu_indices(ep.num_buoys,1)];cap=max(float(ep.rates_bps[0].sum()*ep.episode_length),1.)
    return {"initial_distance_min_m":float(d.min()),"initial_distance_mean_m":float(d.mean()),"buoy_spread_mean_m":float(upper.mean()) if upper.size else 0.,"route_competition":float(np.sort(d)[1]/max(np.sort(d)[0],1e-9)) if len(d)>1 else 1.,"horizon_over_min_travel_time":float(ep.episode_length/max(d.min()/10.,1e-9)),"initial_queue_over_achievable_capacity":float(ep.raw_queue_bits[0].sum()/cap)}
def _rank(v:np.ndarray)->np.ndarray:return np.argsort(np.argsort(v,kind="mergesort"),kind="mergesort")/max(len(v)-1,1)
def _difficulty(eps:Sequence[MaritimeEpisodeV1],rows:List[Dict[str,Any]])->List[Dict[str,Any]]:
    by={};epmap={e.episode_id:e for e in eps}
    for r in rows:by.setdefault(r["scenario_id"],[]).append(r)
    out=[]
    for sid,members in sorted(by.items()):
        e=epmap[members[0]["episode_id"]];pay=np.asarray([m["payload_bits"] for m in members]);ret=np.asarray([m["episode_return"] for m in members]);comp=np.asarray([m["completion_ratio"] for m in members]);best,worst=int(pay.argmax()),int(pay.argmin());g=_geo(e);random=next(m["payload_bits"] for m in members if m["behavior_generator"]=="random_valid")
        out.append({"scenario_id":sid,"scenario_seed":members[0]["scenario_seed"],"num_buoys":e.num_buoys,"initial_geometry_metrics":g,"horizon":e.episode_length,"best_policy":members[best]["behavior_generator"],"worst_policy":members[worst]["behavior_generator"],"best_return":float(ret[best]),"worst_return":float(ret[worst]),"best_payload":float(pay[best]),"worst_payload":float(pay[worst]),"performance_gap":float(pay[best]-pay[worst]),"random_payload":float(random),"best_completion":float(comp.max()),"feasible_flag":bool(comp.max()>=.35 and pay.max()>0),**g})
    gap=np.asarray([x["performance_gap"]/max(x["best_payload"],1.) for x in out]);loss=np.asarray([1-x["random_payload"]/max(x["best_payload"],1.) for x in out]);comp=np.asarray([x["best_completion"] for x in out]);score=.45*_rank(gap)+.35*_rank(loss)+.20*_rank(1-comp);lo,hi=np.quantile(score,(1/3,2/3))
    for x,s in zip(out,score):x["difficulty_score"]=float(s);x["difficulty_tier"]="easy" if s<=lo else ("medium" if s<=hi else "hard")
    return out
def _quality(rows:List[Dict[str,Any]],cases:Sequence[Mapping[str,Any]])->None:
    tier={x["scenario_id"]:x["difficulty_tier"] for x in cases}
    for r in rows:r["difficulty_tier"]=tier[r["scenario_id"]]
    for t in ("easy","medium","hard"):
        xs=[r for r in rows if r["difficulty_tier"]==t];v=np.asarray([r["episode_return"] for r in xs]);lo,hi=np.quantile(v,(.2,.6))
        for r in xs:r["quality_tier"]="low" if r["episode_return"]<=lo else ("high" if r["episode_return"]>=hi else "medium")
def _splits(cases:Sequence[Mapping[str,Any]])->Dict[str,str]:
    seeds=sorted({int(x["scenario_seed"]) for x in cases});hold={s for i,s in enumerate(seeds) if i%5==0};out={};by_seed={}
    for x in cases:by_seed.setdefault(int(x["scenario_seed"]),[]).append(x)
    rank={"easy":0,"medium":1,"hard":2};groups={}
    for seed,xs in by_seed.items():
        if seed in hold:
            for x in xs:out[x["scenario_id"]]="holdout"
        else:groups.setdefault(max(xs,key=lambda x:rank[x["difficulty_tier"]])["difficulty_tier"],[]).append((seed,xs))
    for _,items in groups.items():
        items=sorted(items,key=lambda x:_seed("split",x[0]));a=round(.6*len(items));b=round(.2*len(items))
        for i,(_,xs) in enumerate(items):
            label="train" if i<a else ("validation" if i<a+b else "test")
            for x in xs:out[x["scenario_id"]]=label
    return out
def _padded(ep:MaritimeEpisodeV1,nmax:int)->np.ndarray:
    x=np.zeros((ep.episode_length,9+7*nmax));x[:,:ep.observations.shape[1]]=ep.observations;return x
def _records(eps:Sequence[MaritimeEpisodeV1],rmap:Mapping[str,Mapping[str,Any]],stride:int=16)->List[Dict[str,Any]]:
    nmax=max(e.num_buoys for e in eps);out=[]
    for e in eps:
        s=_padded(e,nmax)
        for t in range(0,e.episode_length,stride):
            u=min(t+10,e.episode_length-1);out.append({"episode_id":e.episode_id,"behavior_generator":rmap[e.episode_id]["behavior_generator"],"state":s[t],"future_delta":e.next_uav_positions_m[u]-e.uav_positions_m[t],"downstream_return":float(e.episode_return-e.rewards[:t].sum())})
    return out
def _stitch(records:Sequence[Mapping[str,Any]])->Tuple[List[Dict[str,Any]],List[Dict[str,Any]],List[Dict[str,Any]],Dict[str,Any]]:
    x=np.asarray([r["state"] for r in records]);x=(x-x.mean(0))/np.maximum(x.std(0),1e-6);local=np.linalg.norm(x[1:]-x[:-1],axis=1);thr=np.quantile(local,(.1,.25,.5));d=np.linalg.norm(x[:,None]-x[None,:],axis=2);np.fill_diagonal(d,np.inf);near=[]
    for i,r in enumerate(records):
        ok=np.asarray([z["episode_id"]!=r["episode_id"] for z in records]);j=int(np.argmin(np.where(ok,d[i],np.inf)));a,b=np.asarray(r["future_delta"]),np.asarray(records[j]["future_delta"]);cos=float(a.dot(b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9));near.append({"episode_id":r["episode_id"],"neighbor_episode_id":records[j]["episode_id"],"behavior_generator":r["behavior_generator"],"neighbor_behavior_generator":records[j]["behavior_generator"],"cross_policy":r["behavior_generator"]!=records[j]["behavior_generator"],"cross_episode_neighbor_distance":float(d[i,j]),"cross_policy_neighbor_distance":float(d[i,j]) if r["behavior_generator"]!=records[j]["behavior_generator"] else "","downstream_behavior_diversity":abs(float(r["downstream_return"]-records[j]["downstream_return"])),"future_direction_cosine":cos})
    med=float(np.median([z["downstream_behavior_diversity"] for z in near]));coverage=[]
    for name,t in zip(("p10","p25","p50"),thr):
        bridges=[z for z in near if z["cross_episode_neighbor_distance"]<=t and (z["future_direction_cosine"]<.7 or z["downstream_behavior_diversity"]>=med)];coverage.append({"threshold_name":name,"threshold":float(t),"state_count":len(records),"cross_episode_neighbor_count":len(near),"cross_policy_overlap_rate":float(np.mean([z["cross_policy"] for z in near])),"bridge_candidate_count":len(bridges),"bridge_candidate_ratio":len(bridges)/max(len(near),1),"downstream_behavior_diversity_mean":float(np.mean([z["downstream_behavior_diversity"] for z in bridges])) if bridges else 0.})
    edges=[(z["episode_id"],z["neighbor_episode_id"]) for z in near if z["cross_episode_neighbor_distance"]<=thr[1] and z["future_direction_cosine"]<.7];nodes=sorted({r["episode_id"] for r in records});parent={n:n for n in nodes}
    def find(a:str)->str:
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for a,b in edges:
        a,b=find(a),find(b)
        if a!=b:parent[a]=b
    comps={}
    for n in nodes:comps[find(n)]=comps.get(find(n),0)+1
    graph=[{"graph_class":"all","graph_node_count":len(nodes),"edge_count":len(edges),"connected_components":len(comps),"largest_connected_component_ratio":max(comps.values())/max(len(nodes),1),"cross_policy_edge_ratio":float(np.mean(["__" not in a or a.split("__")[-3]!=b.split("__")[-3] for a,b in edges])) if edges else 0.,"start_to_high_quality_terminal_connectivity":"diagnostic_not_claimed","bridge_bottleneck_count":sum(v==1 for v in comps.values())}]
    stress={"stitching_stress_constructible":bool(edges),"selection_rule":"existing cross-episode state-overlap pairs with divergent valid futures; no transition fabricated","candidate_edges":[{"source_episode_id":a,"target_episode_id":b} for a,b in edges[:50]]};return near,coverage,graph,stress
def _goal_audit(eps:Sequence[MaritimeEpisodeV1])->Tuple[List[Dict[str,Any]],Dict[str,Any],Dict[str,Any]]:
    nmax=max(e.num_buoys for e in eps);names=["final_task_completion_ratio","final_residual_queue_ratio","final_energy_ratio","final_uav_x_normalized","final_uav_y_normalized"]+[f"final_queue_ratio_buoy_{i}" for i in range(nmax)]+[f"final_completion_buoy_{i}" for i in range(nmax)];g=[];terminal=[];first=[];meanact=[];ret=[];payload=[];disp=[]
    for e in eps:
        z=np.zeros(len(names));z[:5]=e.achieved_goal_global;z[5:5+e.num_buoys]=e.achieved_goal_per_buoy_queue_ratio;z[5+nmax:5+nmax+e.num_buoys]=e.achieved_goal_per_buoy_completion;g.append(z);terminal.append(e.next_uav_positions_m[-1]);first.append(e.actions[0]);meanact.append(e.actions.mean(0));ret.append(e.episode_return);payload.append(e.next_cumulative_payload_bits[-1]);disp.append(e.next_uav_positions_m[-1]-e.uav_positions_m[0])
    g=np.asarray(g);terminal=np.asarray(terminal);first=np.asarray(first);meanact=np.asarray(meanact);disp=np.asarray(disp);corr=lambda a,b:float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-10 and np.std(b)>1e-10 else 0.;out=[]
    for i,n in enumerate(names):
        a=g[:,i];out.append({"name":n,"semantic_meaning":"hindsight achieved terminal field","known_before_rollout":False,"hindsight_only":True,"mean":float(a.mean()),"std":float(a.std()),"min":float(a.min()),"max":float(a.max()),"unique_effective_variation":int(np.unique(np.round(a,8)).size),"correlation_terminal_uav_x":corr(a,terminal[:,0]),"correlation_terminal_uav_y":corr(a,terminal[:,1]),"correlation_initial_to_final_displacement_x":corr(a,disp[:,0]),"correlation_initial_to_final_displacement_y":corr(a,disp[:,1]),"correlation_episode_return":corr(a,np.asarray(ret)),"correlation_payload":corr(a,np.asarray(payload)),"correlation_first_action_x":corr(a,first[:,0]),"correlation_first_action_y":corr(a,first[:,1]),"correlation_mean_trajectory_direction_x":corr(a,meanact[:,0]),"correlation_mean_trajectory_direction_y":corr(a,meanact[:,1]),"padding_driven":bool("buoy_" in n)})
    s=np.linalg.svd(g-g.mean(0),compute_uv=False);v=s*s/max(len(g)-1,1);p=v/max(v.sum(),1e-12);audit={"goal_dimension":len(names),"pca_explained_variance":p.tolist(),"effective_goal_dimension_95pct":int(np.searchsorted(np.cumsum(p),.95)+1),"near_constant_dimensions":[names[i] for i in range(len(names)) if g[:,i].std()<1e-6],"finding":"achieved-terminal goal leaks final UAV position and is hindsight-only"};candidate={"schema_version":GOAL,"deployable_goal_available":True,"fields":[{"name":"target_task_completion_ratio","source":"existing desired_goal","known_before_rollout":True},{"name":"maximum_allowed_residual_queue_ratio","source":"existing desired_goal","known_before_rollout":True},{"name":"minimum_terminal_energy_ratio","source":"existing desired_goal","known_before_rollout":True}],"excluded_fields":["final_uav_position","future_reward","future_payload","future_queue","reference_future_trajectory"],"note":"deployable task goal candidate, but no supplied spatial destination"};return out,audit,candidate
def _horizon(eps:Sequence[MaritimeEpisodeV1])->Tuple[List[Dict[str,Any]],List[Dict[str,Any]],List[Dict[str,Any]]]:
    result=[];multi=[];long=[];nmax=max(e.num_buoys for e in eps)
    for h in HORIZONS:
        ds=[];angles=[];acts=[];future=[];pay=[];rep=[];ids=[]
        for e in eps:
            p=_padded(e,nmax);goal=np.zeros(5+2*nmax); goal[:5]=e.achieved_goal_global; goal[5:5+e.num_buoys]=e.achieved_goal_per_buoy_queue_ratio; goal[5+nmax:5+nmax+e.num_buoys]=e.achieved_goal_per_buoy_completion
            for t in range(e.episode_length-h+1):
                d=e.next_uav_positions_m[t+h-1]-e.uav_positions_m[t];ds.append(d);angles.append(math.atan2(d[1],d[0]) if np.linalg.norm(d)>1e-9 else 0.);acts.append(e.actions[t]);future.append(e.actions[t:min(t+5,e.episode_length)].mean(0));pay.append(e.next_cumulative_payload_bits[-1]-e.cumulative_payload_bits[t]);rep.append(np.r_[p[t],goal]);ids.append(e.episode_id)
        d=np.asarray(ds);a=np.asarray(angles);act=np.asarray(acts);fut=np.asarray(future);mag=np.linalg.norm(d,axis=1);co=lambda x,y:float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-10 and np.std(y)>1e-10 else 0.;hist=np.histogram(a,bins=16,range=(-math.pi,math.pi))[0];pr=hist/max(hist.sum(),1);entropy=float(-np.sum(pr[pr>0]*np.log(pr[pr>0])))
        result.append({"H_w":h,"sample_count":len(d),"displacement_magnitude_mean_m":float(mag.mean()),"normalized_displacement_mean":float(mag.mean()/(h*10.)),"direction_circular_variance":float(1-np.hypot(np.cos(a).mean(),np.sin(a).mean())),"zero_or_near_zero_ratio":float((mag<1e-6).mean()),"target_variance":float(d.var()),"target_entropy_proxy":entropy,"downstream_payload_relation":co(mag,np.asarray(pay)),"policy_separation_proxy":float(np.std(mag))})
        idx=np.linspace(0,len(d)-1,min(len(d),1200),dtype=int);z=np.asarray(rep)[idx];z=(z-z.mean(0))/np.maximum(z.std(0),1e-6);dist=np.linalg.norm(z[:,None]-z[None,:],axis=2);np.fill_diagonal(dist,np.inf);local=[];angular=[];op=[]
        for i in range(len(idx)):
            js=[j for j in np.argsort(dist[i]) if ids[idx[j]]!=ids[idx[i]]][:5]
            if js:
                y=d[idx[js]];local.append(float(np.var(y,axis=0).mean()));q=np.arctan2(y[:,1],y[:,0]);angular.append(float(1-np.hypot(np.cos(q).mean(),np.sin(q).mean())));op.append(float(np.mean(np.cos(q-q[0])<-.5)))
        lm=float(np.mean(local)) if local else 0.;multi.append({"H_w":h,"knn_state_goal_samples":len(idx),"local_target_variance_mean":lm,"local_angular_variance_mean":float(np.mean(angular)) if angular else 0.,"opposite_direction_rate":float(np.mean(op)) if op else 0.,"target_magnitude_variance":float(mag.var()),"local_multimodality_score":lm+(float(np.mean(angular)) if angular else 0.),"mse_averaging_risk":"high" if lm>20 else "moderate"})
        long.append({"H_w":h,"corr_waypoint_current_action_x":co(d[:,0],act[:,0]),"corr_waypoint_current_action_y":co(d[:,1],act[:,1]),"corr_waypoint_future_mean_action_x":co(d[:,0],fut[:,0]),"corr_waypoint_future_mean_action_y":co(d[:,1],fut[:,1]),"direction_consistency_current_action":float(np.mean((d*act).sum(1)/np.maximum(np.linalg.norm(d,axis=1)*np.linalg.norm(act,axis=1),1e-9))),"direction_consistency_future_mean_action":float(np.mean((d*fut).sum(1)/np.maximum(np.linalg.norm(d,axis=1)*np.linalg.norm(fut,axis=1),1e-9)))})
    return result,multi,long
def _waypoint(path:Path,eps:Sequence[MaritimeEpisodeV1],rmap:Mapping[str,Mapping[str,Any]],splits:Mapping[str,str])->Tuple[Dict[str,Any],List[Dict[str,Any]],List[Dict[str,Any]]]:
    nmax=max(e.num_buoys for e in eps);rows=[];state=[];mask=[];goal=[];raw=[];norm=[];point=[]
    for e in eps:
        p=_padded(e,nmax);g=np.zeros(5+2*nmax);g[:5]=e.achieved_goal_global;g[5:5+e.num_buoys]=e.achieved_goal_per_buoy_queue_ratio;g[5+nmax:5+nmax+e.num_buoys]=e.achieved_goal_per_buoy_completion
        for h in HORIZONS:
            for t in range(e.episode_length-h+1):
                q=e.next_uav_positions_m[t+h-1];delta=q-e.uav_positions_m[t];m=rmap[e.episode_id];rows.append({"episode_id":e.episode_id,"scenario_id":e.scenario_id,"split":splits[e.scenario_id],"difficulty_tier":m["difficulty_tier"],"quality_tier":m["quality_tier"],"behavior_generator":m["behavior_generator"],"base_policy":m["base_policy"],"noise_level":m["noise_level"],"t":t,"H_w":h,"valid_flag":True,"terminated":bool(e.terminated[t+h-1]),"truncated":bool(e.truncated[t+h-1])});state.append(p[t]);mask.append(np.r_[np.ones(9+7*e.num_buoys,bool),np.zeros(7*(nmax-e.num_buoys),bool)]);goal.append(np.r_[g,e.desired_goal]);raw.append(delta);norm.append(delta/(h*10.));point.append(q)
    np.savez_compressed(path,current_state=np.asarray(state,np.float32),current_mask=np.asarray(mask,bool),hindsight_goal_v1=np.asarray(goal,np.float32),raw_future_delta=np.asarray(raw,np.float32),normalized_future_delta=np.asarray(norm,np.float32),future_point_m=np.asarray(point,np.float32));stats=[{"H_w":h,"sample_count":sum(r["H_w"]==h for r in rows),"valid_sample_count":sum(r["H_w"]==h for r in rows),"cross_episode":False,"cross_termination":False} for h in HORIZONS];return {"schema_version":WPSCHEMA,"sample_count":len(rows),"max_num_buoys":nmax,"horizons":list(HORIZONS),"storage":"waypoint_samples_v2.npz + waypoint_v2_metadata.csv","required_fields":["episode_id","scenario_id","split","difficulty_tier","quality_tier","behavior_generator","base_policy","noise_level","t","H_w","current_state","current_mask","hindsight_goal_v1","deployable_goal_candidate","raw_future_delta","normalized_future_delta","future_point","valid_flag"],"training_performed":False},stats,rows
def _figures(root:Path,cases:Sequence[Mapping[str,Any]],rows:Sequence[Mapping[str,Any]],coverage:Sequence[Mapping[str,Any]],goals:Sequence[Mapping[str,Any]],horizon:Sequence[Mapping[str,Any]],multi:Sequence[Mapping[str,Any]])->int:
    import matplotlib.pyplot as plt
    root.mkdir(parents=True,exist_ok=True);count=0
    def save(name:str,x:Sequence[Any],y:Sequence[float],line:bool=False)->None:
        nonlocal count
        fig,ax=plt.subplots(figsize=(6,4),facecolor="white");ax.set_facecolor("white")
        if line:ax.plot(x,y,"o-",color="#1f4e79")
        else:ax.bar(range(len(y)),y,color="#9fc5e8");ax.set_xticks(range(len(y)));ax.set_xticklabels([str(v) for v in x],rotation=25,ha="right")
        ax.set_title(name.replace("_"," "));ax.set_ylabel("value");fig.tight_layout();fig.savefig(root/(name+".png"),dpi=150);fig.savefig(root/(name+".svg"));plt.close(fig);count+=2
    q=lambda k,v:sum(r.get(k)==v for r in rows);beh=sorted({r["behavior_generator"] for r in rows});noise=sorted({r["noise_level"] for r in rows})
    save("01_v1_quality_distribution",["high","medium","low"],[0,0,0]);save("02_v2_quality_distribution",["high","medium","low"],[q("quality_tier",x) for x in ("high","medium","low")]);save("03_scenario_difficulty_distribution",["easy","medium","hard"],[sum(c["difficulty_tier"]==x for c in cases) for x in ("easy","medium","hard")]);save("04_behavior_return",beh,[np.mean([r["episode_return"] for r in rows if r["behavior_generator"]==b]) for b in beh]);save("05_behavior_payload",beh,[np.mean([r["payload_bits"] for r in rows if r["behavior_generator"]==b]) for b in beh]);save("06_noise_level_return",noise,[np.mean([r["episode_return"] for r in rows if r["noise_level"]==n]) for n in noise],True);save("07_quality_payload",["high","medium","low"],[np.mean([r["payload_bits"] for r in rows if r["quality_tier"]==x]) for x in ("high","medium","low")]);save("08_cross_policy_state_overlap",[x["threshold_name"] for x in coverage],[x["cross_policy_overlap_rate"] for x in coverage],True);save("09_stitching_graph_example",["states","bridges"],[len(rows),sum(x["bridge_candidate_count"] for x in coverage)]);save("10_bridge_candidate_count",[x["threshold_name"] for x in coverage],[x["bridge_candidate_count"] for x in coverage]);save("11_component_size_proxy",[x["threshold_name"] for x in coverage],[x["bridge_candidate_ratio"] for x in coverage],True);save("12_goal_feature_variance",[x["name"] for x in goals],[x["std"] for x in goals]);save("13_goal_first_action_correlation",[x["name"] for x in goals],[abs(x["correlation_first_action_x"]) for x in goals]);save("14_goal_payload_correlation",[x["name"] for x in goals],[abs(x["correlation_payload"]) for x in goals]);save("15_horizon_displacement",[x["H_w"] for x in horizon],[x["displacement_magnitude_mean_m"] for x in horizon],True);save("16_horizon_angular_diversity",[x["H_w"] for x in horizon],[x["direction_circular_variance"] for x in horizon],True);save("17_horizon_multimodality",[x["H_w"] for x in multi],[x["local_multimodality_score"] for x in multi],True);save("18_state_similarity_target_difference",[x["H_w"] for x in multi],[x["local_target_variance_mean"] for x in multi],True);save("19_v1_v2_coverage",["v1 episodes","v2 episodes"],[80,len(rows)]);save("20_split_scenario_distribution",["train","validation","test","holdout"],[sum(r.get("split")==x for r in rows) for x in ("train","validation","test","holdout")]);return count
def run_round7(*,output_dir:Path,figures_dir:Path,run_id:str,smoke:bool=False)->Dict[str,Any]:
    seeds=range(2000,2004) if smoke else range(2000,2030);horizon=48 if smoke else 128;eps=[];metas=[];snapshots=[];traces={}
    for seed in seeds:
        for n in (3,5):
            sid,cfg,trace,tid,profile=_case(seed,n,horizon);snapshots.append({"scenario_id":sid,"scenario_seed":seed,"num_buoys":n,"config":asdict(cfg),"arrival_trace_id":tid,"profile":profile});traces[sid]=trace.tolist()
            for i,spec in enumerate(_specs()):e,m=_rollout(cfg,sid,seed,tid,spec,i);eps.append(e);metas.append(m)
    rows=[_row(e,next(m for m in metas if m["episode_id"]==e.episode_id)) for e in eps];cases=_difficulty(eps,rows);_quality(rows,cases);rmap={r["episode_id"]:r for r in rows};splits=_splits(cases)
    for r in rows:r["split"]=splits[r["scenario_id"]]
    output_dir.mkdir(parents=True,exist_ok=False);figures_dir.mkdir(parents=True,exist_ok=False);_json(output_dir/"scenario_pool_v2.json",{"schema_version":SCHEMA,"scenarios":snapshots,"arrival_traces":traces,"selection":"complete mixed feasible pool; no favorable-result exclusion"});_csv(output_dir/"scenario_difficulty_audit.csv",cases);_csv(output_dir/"trajectory_quality_audit.csv",rows)
    bg=[]
    for b in sorted({r["behavior_generator"] for r in rows}):
        x=[r for r in rows if r["behavior_generator"]==b];bg.append({"behavior_generator":b,"episodes":len(x),"mean_return":float(np.mean([r["episode_return"] for r in x])),"mean_payload_bits":float(np.mean([r["payload_bits"] for r in x])),"quality_distribution":{q:sum(r["quality_tier"]==q for r in x) for q in ("high","medium","low")},"uses_future_information":False,"safe_projection":True})
    _csv(output_dir/"behavior_generator_audit.csv",bg);dev=[e for e in eps if splits[e.scenario_id]!="holdout"];hold=[e for e in eps if splits[e.scenario_id]=="holdout"];da={e.episode_id:splits[e.scenario_id] for e in dev};ha={e.episode_id:"test" for e in hold};dv=validate_dataset_v1(dev,da);hv=validate_dataset_v1(hold,ha);save_dataset_v1(output_dir/"development_dataset_v2",dev,split_assignments=da,metadata={"dataset_version":DATASET,"training_performed":False,"source_dataset_version":"generated_round7"});save_dataset_v1(output_dir/"untouched_closed_loop_holdout_v2",hold,split_assignments=ha,metadata={"dataset_version":DATASET,"training_performed":False,"role":"untouched_holdout"})
    near,coverage,graph,stress=_stitch(_records(dev,rmap));_csv(output_dir/"state_coverage_metrics.csv",near);_csv(output_dir/"stitching_coverage_audit.csv",coverage);_csv(output_dir/"stitching_graph_metrics.csv",graph);_json(output_dir/"stitching_stress_manifest.json",stress);grows,gaudit,candidate=_goal_audit(dev);_csv(output_dir/"goal_information_audit.csv",grows);_json(output_dir/"deployable_goal_candidate.json",candidate);(output_dir/"goal_schema_audit.md").write_text("# Goal schema audit\n\n```json\n"+json.dumps(_plain(gaudit),indent=2)+"\n```\n\nAchieved terminal goals are hindsight-only; the pre-rollout desired task goal is the deployable candidate.\n",encoding="utf-8");hrows,multi,long=_horizon(dev);_csv(output_dir/"waypoint_horizon_audit.csv",hrows);_csv(output_dir/"waypoint_multimodality_audit.csv",multi);_csv(output_dir/"waypoint_long_term_information_audit.csv",long);wpm,wps,wpr=_waypoint(output_dir/"waypoint_samples_v2.npz",eps,rmap,splits);_csv(output_dir/"waypoint_v2_metadata.csv",wpr);_csv(output_dir/"waypoint_v2_statistics.csv",wps);_json(output_dir/"waypoint_v2_manifest.json",wpm)
    manifest={"schema_version":SCHEMA,"dataset_version":DATASET,"episode_count":len(eps),"transition_count":sum(e.episode_length for e in eps),"source_dataset_version":"generated_round7","development_dataset":"development_dataset_v2","untouched_holdout":"untouched_closed_loop_holdout_v2","training_performed":False,"dataset_generation_performed":True,"validation":dv,"holdout_validation":hv};_json(output_dir/"dataset_v2_manifest.json",manifest);_json(output_dir/"dataset_v2_schema.json",{"schema_version":DATASET,"physical_trajectory_storage":"schema-v1 arrays plus v2 behavior/quality sidecars","new_fields":["behavior_generator","base_policy","noise_level","switch_schedule","behavior_seed","difficulty_tier","quality_tier","source_dataset_version"]});_json(output_dir/"dataset_v2_split_manifest.json",{"split_unit":"scenario_id","assignments":splits,"scenario_leakage":False,"ratios":{"train":.6,"validation":.2,"test":.2},"stratified_by":["difficulty_tier","num_buoys"]});_json(output_dir/"untouched_holdout_v2_manifest.json",{"role":"future frozen-model evaluation only","scenario_ids":sorted(k for k,v in splits.items() if v=="holdout"),"not_used_for_training":True,"not_used_for_validation":True,"not_used_for_horizon_or_goal_selection":True})
    qc={q:sum(r["quality_tier"]==q for r in rows) for q in ("high","medium","low")};dc={q:sum(c["difficulty_tier"]==q for c in cases) for q in ("easy","medium","hard")};ready=all(qc.values()) and all(dc.values()) and stress["stitching_stress_constructible"];gate={"retraining_data_ready":bool(ready),"quality_counts":qc,"difficulty_counts":dc,"bridge_candidates":coverage,"deployable_goal_available":candidate["deployable_goal_available"],"recommended_horizon_candidates":[10,20],"mse_averaging_risk_by_horizon":{str(x["H_w"]):x["mse_averaging_risk"] for x in multi},"blockers":[] if ready else ["insufficient tier or measurable bridge coverage"],"training_performed":False};_json(output_dir/"round_07_readiness_gate.json",gate);figs=_figures(figures_dir,cases,rows,coverage,grows,hrows,multi);config={"schema_version":SCHEMA,"run_id":run_id,"smoke":smoke,"scenario_seeds":list(seeds),"num_buoys":[3,5],"episode_horizon":horizon,"behavior_generators":_specs(),"training_performed":False,"dataset_generation_performed":True};_json(output_dir/"run_config.json",config);_json(output_dir/"run_manifest.json",{"run_id":run_id,"schema_version":SCHEMA,"output_files":[p.name for p in output_dir.iterdir()],"figure_files":figs,"training_performed":False});return {"episodes":eps,"rows":rows,"cases":cases,"readiness":gate,"output_dir":output_dir,"figures_dir":figures_dir,"figures":figs}
