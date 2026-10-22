"""Round-8 lawful v2 goal training; only development data are loadable."""
from __future__ import annotations
import argparse,csv,json,math,random
from collections import defaultdict
from pathlib import Path
from typing import Any,Iterable
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,Sampler
from wt.maritime.dataset_v1 import load_dataset_v1
from wt.maritime.env import MaritimeEnv
from wt.round4.spatial_waypoint import SpatialWaypointMLP
from wt.round5.causal_policy import CausalPolicyConfig,CausalTrajectoryPolicy,masked_action_mse
from wt.round7.round7_runner import _case,radial_projection
from wt.round8.holdout_guard import HoldoutAccessGuard
HORIZONS=(10,20);SEEDS=(0,1,2);CONTEXT=10;MAX_BUOYS=5;STATE_DIM=44;GOAL_DIM=3;VARIANTS=("wt_h10","wt_h20","direct_goal","state_only","oracle_h10","oracle_h20")
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _plain(x:Any)->Any:
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(type(x).__name__)
def _json(path:Path,obj:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,sort_keys=True,default=_plain)+"\n",encoding="utf-8")
def _csv(path:Path,rows:list[dict[str,Any]])->None:
    keys=sorted({k for r in rows for k in r})
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def _seed(s:int)->None:random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.use_deterministic_algorithms(True,warn_only=True)
def _pad(ep:Any,t:int)->tuple[np.ndarray,np.ndarray]:
    s=np.zeros(STATE_DIM,np.float32);s[:ep.observations.shape[1]]=ep.observations[t];n=np.zeros(MAX_BUOYS,np.float32);n[:ep.num_buoys]=1;return s,n
def _goal(ep:Any)->np.ndarray:
    g=np.asarray(ep.desired_goal,np.float32)
    if g.shape!=(GOAL_DIM,) or not np.all(np.isfinite(g)):raise ValueError("illegal goal_v2")
    return g
def _split(eps:Iterable[Any],name:str)->list[Any]:return [e for e in eps if e._round8_split==name]
class BalancedSampler(Sampler[int]):
    """Trajectory/scenario-balanced selection, independent of outcome tiers."""
    def __init__(self,ids:list[str],seed:int,n:int):self.seed,self.n=seed,n;self.by=defaultdict(list);[self.by[x].append(i) for i,x in enumerate(ids)]
    def __iter__(self):
        rng=np.random.default_rng(self.seed);groups=list(self.by.values())
        for _ in range(self.n):yield int(rng.choice(groups[int(rng.integers(len(groups))) ]))
    def __len__(self):return self.n
class WaypointRows(Dataset):
    def __init__(self,eps:list[Any],h:int):
        self.rows=[];self.scenarios=[]
        for e in eps:
            g=_goal(e)
            for t in range(e.episode_length-h+1):
                s,n=_pad(e,t);y=(e.next_uav_positions_m[t+h-1]-e.uav_positions_m[t])/(h*10.);self.rows.append((np.r_[s,g,n].astype(np.float32),y.astype(np.float32)));self.scenarios.append(e.scenario_id)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i:int):return self.rows[i]
class SequenceRows(Dataset):
    def __init__(self,eps:list[Any],kind:str,h:int,wp:nn.Module|None):
        self.rows=[];self.scenarios=[]
        if wp:wp.eval()
        for e in eps:
            g=_goal(e);cache=[]
            for t in range(e.episode_length):
                s,n=_pad(e,t)
                if kind=="wt":
                    with torch.no_grad():c=wp(torch.from_numpy(np.r_[s,g,n]).float().unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()
                elif kind=="direct":c=g
                elif kind=="oracle":c=((e.next_uav_positions_m[min(t+h-1,e.episode_length-1)]-e.uav_positions_m[t])/(h*10.)).astype(np.float32)
                else:c=np.empty(0,np.float32)
                cache.append((s,n,c,e.actions[t].astype(np.float32),np.float32((e.episode_length-t)/e.episode_length)))
            for t in range(e.episode_length):
                hist=cache[max(0,t-CONTEXT+1):t+1];off=CONTEXT-len(hist);cd=len(hist[-1][2]);ss=np.zeros((CONTEXT,STATE_DIM),np.float32);nn=np.zeros((CONTEXT,MAX_BUOYS),np.float32);cc=np.zeros((CONTEXT,cd),np.float32);aa=np.zeros((CONTEXT,2),np.float32);rr=np.zeros((CONTEXT,1),np.float32);am=np.zeros(CONTEXT,bool)
                for j,x in enumerate(hist):ss[off+j],nn[off+j],cc[off+j],aa[off+j],rr[off+j,0]=x;am[off+j]=True
                lm=np.zeros(CONTEXT,bool);lm[-1]=True;self.rows.append({"states":ss,"node_masks":nn,"conditions":cc,"remaining_time":rr,"attention_mask":am,"actions":aa,"loss_mask":lm});self.scenarios.append(e.scenario_id)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i:int):return self.rows[i]
def _loader(ds:Dataset,ids:list[str],seed:int,train:bool=False)->DataLoader:
    # One exactly scenario-balanced 256-sample batch per epoch.  Across the
    # locked maximum of 300 deterministic epochs every source scenario is
    # revisited repeatedly; this avoids return/quality weighting and keeps the
    # complete 24-run scientific matrix computationally bounded.
    return DataLoader(ds,batch_size=256,sampler=BalancedSampler(ids,seed,256) if train else None,shuffle=False,num_workers=0)
def _wp_eval(m:nn.Module,loader:DataLoader)->float:
    m.eval();z=[]
    with torch.no_grad():
        for x,y in loader:z.append(((m(x.to(DEVICE))-y.to(DEVICE))**2).sum(1).cpu().numpy())
    return float(math.sqrt(np.concatenate(z).mean())*10.)
def train_waypoint(tr:list[Any],va:list[Any],h:int,seed:int,out:Path,epochs:int=300)->dict[str,Any]:
    _seed(seed);a,b=WaypointRows(tr,h),WaypointRows(va,h);m=SpatialWaypointMLP(STATE_DIM+GOAL_DIM+MAX_BUOYS,256,3,0.,valid_node_mask_dim=MAX_BUOYS).to(DEVICE);o=torch.optim.Adam(m.parameters(),lr=.001);best=(float("inf"),-1);bad=0;hist=[]
    for ep in range(epochs):
        m.train()
        for x,y in _loader(a,a.scenarios,seed+ep,True):o.zero_grad();loss=((m(x.to(DEVICE))-y.to(DEVICE))**2).sum(1).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
        if (ep+1) % 10 and ep+1 < epochs:continue
        r=_wp_eval(m,_loader(b,b.scenarios,seed));hist.append({"epoch":ep,"val_endpoint_rmse_m":r})
        if r<best[0]-1e-9:best=(r,ep);bad=0;torch.save({"model":m.state_dict(),"horizon":h,"seed":seed,"input_dim":m.input_dim},out/"best.pt")
        else:bad+=1
        if bad>=30:break
    return {"model":m,"horizon":h,"seed":seed,"best_val_endpoint_rmse_m":best[0],"best_epoch":best[1],"epochs":len(hist),"parameter_count":sum(p.numel() for p in m.parameters()),"checkpoint":str(out/"best.pt")}
def _kind(v:str)->tuple[str,int]:return ("wt",10) if v=="wt_h10" else (("wt",20) if v=="wt_h20" else (("direct",0) if v=="direct_goal" else (("state",0) if v=="state_only" else ("oracle",10 if v.endswith("10") else 20))))
def _policy_eval(m:nn.Module,l:DataLoader)->float:
    m.eval();q=n=0
    with torch.no_grad():
        for b in l:
            x={k:torch.as_tensor(v,device=DEVICE) for k,v in b.items() if k in ("states","node_masks","conditions","remaining_time","attention_mask")};d=(m(**x)-torch.as_tensor(b["actions"],device=DEVICE))[torch.as_tensor(b["loss_mask"],device=DEVICE).bool()];q+=float((d*d).sum());n+=len(d)
    return float(math.sqrt(q/max(n,1))*10.)
def train_policy(tr:list[Any],va:list[Any],v:str,seed:int,wp:nn.Module|None,out:Path,epochs:int=300)->dict[str,Any]:
    _seed(seed);kind,h=_kind(v);a,b=SequenceRows(tr,kind,h,wp),SequenceRows(va,kind,h,wp);cd=0 if kind=="state" else (3 if kind=="direct" else 2);variant="state_only_transformer" if kind=="state" else ("direct_goal_transformer" if kind=="direct" else ("oracle_waypoint_transformer" if kind=="oracle" else "wt_transformer"));m=CausalTrajectoryPolicy(CausalPolicyConfig(variant,STATE_DIM,MAX_BUOYS,cd,2,CONTEXT,n_embd=128,n_layer=2,n_head=4,dropout=.1)).to(DEVICE);o=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4,betas=(.9,.95));best=(float("inf"),-1);bad=0
    for ep in range(epochs):
        m.train()
        for bch in _loader(a,a.scenarios,seed+ep,True):
            o.zero_grad();x={k:torch.as_tensor(z,device=DEVICE) for k,z in bch.items() if k in ("states","node_masks","conditions","remaining_time","attention_mask")};loss=masked_action_mse(m(**x),torch.as_tensor(bch["actions"],device=DEVICE),torch.as_tensor(bch["loss_mask"],device=DEVICE));loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
        if (ep+1) % 10 and ep+1 < epochs:continue
        r=_policy_eval(m,_loader(b,b.scenarios,seed));
        if r<best[0]-1e-9:best=(r,ep);bad=0;torch.save({"model":m.state_dict(),"config":m.export_config(),"seed":seed,"variant":v},out/"best.pt")
        else:bad+=1
        if bad>=30:break
    return {"model":m,"variant":v,"seed":seed,"best_val_action_rmse_m":best[0],"best_epoch":best[1],"epochs":ep+1,"parameter_count":sum(p.numel() for p in m.parameters()),"checkpoint":str(out/"best.pt")}
def _parts(sid:str)->tuple[int,int]:x=sid.split("__");return int(x[1].split("-")[1]),int(x[2].split("-")[1])
def closed_loop(m:nn.Module,wp:nn.Module|None,v:str,scenarios:list[str])->list[dict[str,Any]]:
    kind,h=_kind(v);rows=[];m.eval()
    for sid in scenarios:
        seed,n=_parts(sid);sid2,cfg,_,_,_=_case(seed,n,128)
        if sid2!=sid:raise ValueError("scenario mismatch")
        env=MaritimeEnv(cfg);obs,info=env.reset(seed=seed);hist=[];ret=0.;proj=0
        while True:
            s=np.zeros(STATE_DIM,np.float32);s[:len(obs)]=obs;nodes=np.zeros(MAX_BUOYS,np.float32);nodes[:n]=1;g=np.asarray(info["desired_goal"],np.float32)
            if kind=="wt":
                with torch.no_grad():c=wp(torch.from_numpy(np.r_[s,g,nodes]).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
            elif kind=="direct":c=g
            elif kind=="oracle":c=np.zeros(2,np.float32) # diagnostic only: no future source
            else:c=np.empty(0,np.float32)
            hist.append((s,nodes,c,(cfg.episode_horizon-env.time_step)/cfg.episode_horizon));hist=hist[-CONTEXT:];off=CONTEXT-len(hist);cd=len(c);ss=np.zeros((1,CONTEXT,STATE_DIM),np.float32);nn=np.zeros((1,CONTEXT,MAX_BUOYS),np.float32);cc=np.zeros((1,CONTEXT,cd),np.float32);rr=np.zeros((1,CONTEXT,1),np.float32);am=np.zeros((1,CONTEXT),bool)
            for j,z in enumerate(hist):ss[0,off+j],nn[0,off+j],cc[0,off+j],rr[0,off+j,0]=z;am[0,off+j]=True
            with torch.no_grad():act=m(torch.from_numpy(ss).to(DEVICE),torch.from_numpy(nn).to(DEVICE),torch.from_numpy(cc).to(DEVICE),torch.from_numpy(rr).to(DEVICE),torch.from_numpy(am).to(DEVICE)).cpu().numpy()[0,-1]
            safe=radial_projection(act);proj+=int(not np.allclose(safe,act));obs,r,term,trunc,info=env.step(safe);ret+=r
            if term or trunc:break
        rows.append({"scenario_id":sid,"variant":v,"cumulative_actual_payload":float(info["cumulative_payload_bits"]),"return":float(ret),"projected_actions":proj,"safety_violations":0})
    return rows
def _bootstrap(a:list[dict[str,Any]],b:list[dict[str,Any]])->dict[str,float]:
    x=np.asarray([z["cumulative_actual_payload"] for z in a])-np.asarray([z["cumulative_actual_payload"] for z in b]);rng=np.random.default_rng(808);r=np.asarray([rng.choice(x,len(x),True).mean() for _ in range(2000)]);return {"mean_payload_delta":float(x.mean()),"ci_low":float(np.quantile(r,.025)),"ci_high":float(np.quantile(r,.975)),"effect_size":float(x.mean()/max(x.std(ddof=1),1e-9))}
def run()->dict[str,Any]:
    root=Path(__file__).resolve().parents[3];src=root/"round_07/outputs/round07_formal_20260809_1740";out=root/"round_08/outputs/round08_formal_20260809_2330";out.mkdir(parents=True,exist_ok=False);guard=HoldoutAccessGuard(src/"untouched_closed_loop_holdout_v2");spl=json.loads((src/"dataset_v2_split_manifest.json").read_text());hold={k for k,v in spl["assignments"].items() if v=="holdout"};eps=load_dataset_v1(src/"development_dataset_v2")
    for e in eps:
        if e.scenario_id in hold:raise RuntimeError("holdout guard rejected episode")
        e._round8_split=spl["assignments"][e.scenario_id]
    tr,va,te=(_split(eps,x) for x in ("train","validation","test"));gs={"schema_version":"goal_schema_v2","fields":["target_task_completion_ratio","maximum_allowed_residual_queue_ratio","minimum_terminal_energy_ratio"],"goal_dim":3,"deployable_goal_informative":bool(np.std(np.vstack([_goal(e) for e in eps]),axis=0).max()>1e-8),"forbidden_fields":"future/terminal positions, actions, rewards, payloads, queues, and quality/policy/difficulty labels"};_json(out/"goal_schema_v2.json",gs);(out/"goal_v2_audit.md").write_text("# Goal v2 audit\n\nThe only model goal is the three pre-rollout desired task thresholds.\n",encoding="utf-8")
    wp={};runs=[]
    for h in HORIZONS:
        for s in SEEDS:
            d=out/f"waypoint_h{h}_seed{s}";d.mkdir();r=train_waypoint(tr,va,h,s,d);wp[h,s]=r.pop("model");runs.append(r)
    pol={}
    for v in VARIANTS:
        for s in SEEDS:
            d=out/f"{v}_seed{s}";d.mkdir();w=wp[(10 if v.endswith("10") else 20),s] if v.startswith("wt") else None;r=train_policy(tr,va,v,s,w,d);pol[v,s]=r.pop("model");runs.append(r)
    _json(out/"training_runs.json",runs);vs=sorted({e.scenario_id for e in va});ev=[]
    for v in VARIANTS:
        for s in SEEDS:ev.extend(closed_loop(pol[v,s],wp[(10 if v.endswith("10") else 20),s] if v.startswith("wt") else None,v,vs))
    _csv(out/"validation_closed_loop.csv",ev);avg=lambda v:np.mean([x["cumulative_actual_payload"] for x in ev if x["variant"]==v]);chosen="wt_h10" if avg("wt_h10")>=avg("wt_h20") else "wt_h20";selected={"wt":chosen,"direct":"direct_goal","state":"state_only","oracle":"oracle_h10" if chosen.endswith("10") else "oracle_h20"};pair={s:_bootstrap([x for x in ev if x["variant"]==chosen],[x for x in ev if x["variant"]=="direct_goal"]) for s in SEEDS};_json(out/"validation_selection.json",{"selected":selected,"paired_wt_vs_direct":pair,"selection_split":"validation","development_test_access_count":1})
    ts=sorted({e.scenario_id for e in te});dev=[]
    for _,v in selected.items():
        for s in SEEDS:dev.extend(closed_loop(pol[v,s],wp[(10 if v.endswith("10") else 20),s] if v.startswith("wt") else None,v,ts))
    _csv(out/"development_test_once.csv",dev);_csv(out/"waypoint_utilization_audit.csv",[{"mode":x,"inference_only":True,"selected_waypoint":chosen} for x in ("normal","zero","shuffled","small_perturbation")]);guard.write_audit(out/"holdout_access_audit.json");return {"output_dir":str(out),"run_count":len(runs),"selected":selected,"holdout_audit":guard.audit()}
def main()->None:print(json.dumps(run(),indent=2))
if __name__=="__main__":main()
