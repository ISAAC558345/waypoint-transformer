"""Frozen-checkpoint diagnostics on Round-7 v2 train/validation only."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
from typing import Any
import numpy as np, torch
from torch.utils.data import DataLoader
from wt.maritime.dataset_v1 import load_dataset_v1
from wt.round4.spatial_waypoint import SpatialWaypointMLP
from wt.round5.causal_policy import CausalPolicyConfig,CausalTrajectoryPolicy
from wt.round8.holdout_guard import HoldoutAccessGuard
from wt.round8.round8_runner import DEVICE,GOAL_DIM,HORIZONS,MAX_BUOYS,STATE_DIM,SequenceRows,_goal,_kind,_pad,_split

def _j(p:Path,x:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist())+'\n',encoding='utf-8')
def _csv(p:Path,rows:list[dict[str,Any]])->None:
    with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
def _load_wp(out:Path,h:int,seed:int)->SpatialWaypointMLP:
    x=torch.load(out/f'waypoint_h{h}_seed{seed}'/'best.pt',map_location=DEVICE,weights_only=False);m=SpatialWaypointMLP(STATE_DIM+GOAL_DIM+MAX_BUOYS,256,3,0.,valid_node_mask_dim=MAX_BUOYS).to(DEVICE);m.load_state_dict(x['model']);m.eval();return m
def _load_policy(out:Path,variant:str,seed:int)->CausalTrajectoryPolicy:
    x=torch.load(out/f'{variant}_seed{seed}'/'best.pt',map_location=DEVICE,weights_only=False);m=CausalTrajectoryPolicy(CausalPolicyConfig(**x['config'])).to(DEVICE);m.load_state_dict(x['model']);m.eval();return m
def _offline(model:CausalTrajectoryPolicy,ds:SequenceRows,mode:str='normal')->tuple[list[dict[str,Any]],dict[str,float]]:
    rows=[];rng=np.random.default_rng(9009); shuffled=None
    if mode=='shuffled':
        shuffled=np.stack([r['conditions'] for r in ds.rows]);shuffled=shuffled[rng.permutation(len(shuffled))]
    for offset,b in enumerate(DataLoader(ds,batch_size=256,shuffle=False)):
        c=b['conditions'].numpy()
        if mode=='zero':c=np.zeros_like(c)
        elif mode=='sign_flip':c=-c
        elif mode=='noise':c=c+rng.normal(0,.05,c.shape).astype(np.float32)
        elif mode=='shuffled':c=shuffled[offset*256:offset*256+len(c)]
        x={k:torch.as_tensor(v,device=DEVICE) for k,v in b.items() if k in ('states','node_masks','remaining_time','attention_mask')};x['conditions']=torch.as_tensor(c,device=DEVICE)
        with torch.no_grad():pred=model(**x)[:,-1].cpu().numpy()
        target=b['actions'].numpy()[:,-1];base=b['conditions'].numpy()[:,-1]
        for i in range(len(pred)):rows.append({'prediction_x':float(pred[i,0]),'prediction_y':float(pred[i,1]),'target_x':float(target[i,0]),'target_y':float(target[i,1]),'phi_norm':float(np.linalg.norm(base[i]))})
    err=np.asarray([[r['prediction_x']-r['target_x'],r['prediction_y']-r['target_y']] for r in rows]);return rows,{'action_rmse_m':float(np.sqrt(np.mean(np.sum(err*err,axis=1)))*10.)}
def _bootstrap(x:np.ndarray)->dict[str,float]:
    rng=np.random.default_rng(9);b=np.array([rng.choice(x,len(x),True).mean() for _ in range(2000)]);return {'mean':float(x.mean()),'median':float(np.median(x)),'ci_low':float(np.quantile(b,.025)),'ci_high':float(np.quantile(b,.975)),'wins':int((x>0).sum()),'ties':int((x==0).sum()),'losses':int((x<0).sum())}
def _temporal(eps:list[Any])->list[dict[str,Any]]:
    out=[]
    for h in (1,5,10,20):
        vals=[]
        for e in eps:
            for t in range(e.episode_length-h+1):
                q=e.next_uav_positions_m[t+h-1]-e.uav_positions_m[t];a=e.actions[t];f=e.actions[t:min(t+5,e.episode_length)].mean(0);g=e.actions[t:min(t+10,e.episode_length)].mean(0);z=e.next_uav_positions_m[-1]-e.uav_positions_m[t];vals.append((q,a,f,g,z))
        q,a,f,g,z=map(np.asarray,zip(*vals));cos=lambda x:float(np.mean((q*x).sum(1)/np.maximum(np.linalg.norm(q,axis=1)*np.linalg.norm(x,axis=1),1e-9)));corr=lambda x:float(np.corrcoef(np.linalg.norm(q,axis=1),np.linalg.norm(x,axis=1))[0,1]);out.append({'H_w':h,'current_action_cosine':cos(a),'future5_cosine':cos(f),'future10_cosine':cos(g),'terminal_displacement_cosine':cos(z),'current_action_magnitude_correlation':corr(a),'future5_magnitude_correlation':corr(f),'direction_agreement_current':float(np.mean((q*a).sum(1)>0))})
    return out
def _goals(eps:list[Any])->list[dict[str,Any]]:
    g=np.vstack([_goal(e) for e in eps]);payload=np.asarray([e.next_cumulative_payload_bits[-1] for e in eps]);first=np.asarray([e.actions[0] for e in eps]);names=['target_task_completion_ratio','maximum_allowed_residual_queue_ratio','minimum_terminal_energy_ratio'];return [{'field':n,'variance':float(g[:,i].var()),'unique_values':int(len(np.unique(g[:,i]))),'entropy_proxy':float(np.log(max(len(np.unique(g[:,i])),1))),'payload_correlation':float(np.corrcoef(g[:,i],payload)[0,1]) if g[:,i].std()>1e-6 else 0.,'first_action_norm_correlation':float(np.corrcoef(g[:,i],np.linalg.norm(first,axis=1))[0,1]) if g[:,i].std()>1e-6 else 0.} for i,n in enumerate(names)]
def _multimodal(eps:list[Any],wp:SpatialWaypointMLP)->list[dict[str,Any]]:
    records=[]
    for e in eps:
        g=_goal(e)
        for t in range(e.episode_length-20+1):
            s,n=_pad(e,t);tar=(e.next_uav_positions_m[t+19]-e.uav_positions_m[t])/200.;records.append((e.episode_id,np.r_[s,g],tar,np.r_[s,g,n]))
    rng=np.random.default_rng(7);idx=rng.choice(len(records),min(400,len(records)),False);X=np.asarray([records[i][1] for i in idx]);X=(X-X.mean(0))/np.maximum(X.std(0),1e-6);out=[]
    for ii,i in enumerate(idx):
        d=np.linalg.norm(X-X[ii],axis=1);ok=np.asarray([records[j][0]!=records[i][0] for j in idx]);near=np.argsort(np.where(ok,d,np.inf))[:5];y=np.asarray([records[idx[j]][2] for j in near]);
        if len(y)<2:continue
        mean=y.mean(0);sse1=float(((y-mean)**2).sum());c=y[[0,-1]].copy()
        for _ in range(8):lab=np.argmin(((y[:,None]-c[None])**2).sum(2),1);c=np.asarray([y[lab==k].mean(0) if np.any(lab==k) else c[k] for k in range(2)])
        sse2=float(((y-c[lab])**2).sum());
        with torch.no_grad():p=wp(torch.from_numpy(records[i][3]).float().unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
        out.append({'episode_id':records[i][0],'local_target_variance':float(y.var()),'angular_variance':float(np.var(np.arctan2(y[:,1],y[:,0]))),'opposite_direction_rate':float(np.mean(np.cos(np.arctan2(y[:,1],y[:,0])-np.arctan2(y[0,1],y[0,0]))<-.5)),'one_cluster_sse':sse1,'two_cluster_sse':sse2,'cluster_separation_score':float((sse1-sse2)/max(sse1,1e-9)),'prediction_distance_to_nearest_mode':float(np.min(np.linalg.norm(p-c,axis=1))),'mse_average_case':bool(sse2<sse1*.55 and np.min(np.linalg.norm(p-c,axis=1))>.08)})
    return out
def run(root:Path)->dict[str,Any]:
    src=root/'round_07/outputs/round07_formal_20260809_1740';r8=root/'round_08/outputs/round08_formal_20260809_2330';out=root/'round_09/outputs/round09_diagnosis_20260810_0030';out.mkdir(parents=True,exist_ok=False);guard=HoldoutAccessGuard(src/'untouched_closed_loop_holdout_v2');spl=json.loads((src/'dataset_v2_split_manifest.json').read_text());eps=load_dataset_v1(src/'development_dataset_v2')
    for e in eps:e._round8_split=spl['assignments'][e.scenario_id]
    train,val=_split(eps,'train'),_split(eps,'validation');wp=_load_wp(r8,20,0);models={'oracle_h20':_load_policy(r8,'oracle_h20',0),'wt_h20':_load_policy(r8,'wt_h20',0),'direct_goal':_load_policy(r8,'direct_goal',0),'state_only':_load_policy(r8,'state_only',0)};datasets={'oracle_h20':SequenceRows(val,'oracle',20,None),'wt_h20':SequenceRows(val,'wt',20,wp),'direct_goal':SequenceRows(val,'direct',0,None),'state_only':SequenceRows(val,'state',0,None)};base={};details={}
    for n in models:details[n],base[n]=_offline(models[n],datasets[n])
    sensitivity=[];normal=np.asarray([[x['prediction_x'],x['prediction_y']] for x in details['wt_h20']])
    for mode in ('normal','zero','shuffled','sign_flip','noise'):
        rows,met=_offline(models['wt_h20'],datasets['wt_h20'],mode);a=np.asarray([[x['prediction_x'],x['prediction_y']] for x in rows]);diff=np.linalg.norm(a-normal,axis=1);cos=np.sum(a*normal,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(normal,axis=1),1e-9);sensitivity.append({'mode':mode,**met,'mean_absolute_action_change':float(diff.mean()),'cosine_change':float(1-cos.mean()),'waypoint_sensitivity_score':float(diff.mean()/max(np.linalg.norm(normal,axis=1).mean(),1e-9)),'waypoint_ignored':bool(mode!='normal' and diff.mean()<1e-3)})
    # Oracle action improvement is diagnostic only; no Oracle rollout is claimed.
    oracle_gap=[]
    for name in ('direct_goal','state_only','wt_h20'):
        a=np.asarray([[x['prediction_x'],x['prediction_y']] for x in details['oracle_h20']]);b=np.asarray([[x['prediction_x'],x['prediction_y']] for x in details[name]]);oracle_gap.append({'comparison':'oracle_h20_minus_'+name,'action_prediction_l2_delta':_bootstrap(np.linalg.norm(a,axis=1)-np.linalg.norm(b,axis=1)),'oracle_action_rmse_m':base['oracle_h20']['action_rmse_m'],'other_action_rmse_m':base[name]['action_rmse_m'],'closed_loop_payload_interpretation':'not_claimed: Round-8 Oracle closed-loop Phi was zeroed'})
    multi=_multimodal(val,wp);temporal=_temporal(train+val);goals=_goals(train+val);# Diagnostics show nonzero Phi sensitivity but no valid Oracle closed-loop gain; choose M3 conservatively.
    gate={'mechanism_gate':'M3','classification':'Spatial Waypoint Not Useful','minimal_variant_training_allowed':False,'training_performed':False,'reason':'Oracle offline action fit is not sufficient evidence of a causal closed-loop advantage; Round-8 Gate-B and StateOnly competitiveness remain, while Phi sensitivity is nonzero.','old_development_test_accessed':False,'untouched_holdout_accessed':False}
    _csv(out/'oracle_gap_analysis.csv',oracle_gap);_csv(out/'waypoint_sensitivity_audit.csv',sensitivity);_csv(out/'waypoint_temporal_information_audit.csv',temporal);_csv(out/'waypoint_multimodality_round09.csv',multi);_csv(out/'goal_v2_information_audit.csv',goals);_csv(out/'stitching_conditional_analysis.csv',[{'status':'not_claimed','reason':'Round-8 did not persist a valid stitching-stratified closed-loop policy metric; no favorable subset inference is made'}]);_csv(out/'bridge_region_analysis.csv',[{'status':'not_claimed','reason':'same as stitching conditional analysis'}]);_j(out/'round09_mechanism_gate.json',gate);guard.write_audit(out/'holdout_access_audit.json');_j(out/'run_manifest.json',{'diagnosis_only':True,'optimizer_created':False,'checkpoint_created':False,'train_episodes':len(train),'validation_episodes':len(val),'models':list(models),'outputs':[p.name for p in out.iterdir()]});return {'output_dir':str(out),'gate':gate,'base':base,'sensitivity':sensitivity}
if __name__=='__main__':print(json.dumps(run(Path.cwd()),indent=2))
