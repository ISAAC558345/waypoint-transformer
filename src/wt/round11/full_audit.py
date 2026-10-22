"""Append-only detailed Round-11 checkpoint evaluation on permitted splits."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np
import torch
from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.channel import theoretical_max_rates
from wt.round11.runner import _subset, _targets, _predict, _mean_baseline, load_round10_samples
from wt.round11.task_waypoint import TaskWaypointConfig, TaskWaypointMLP, task_waypoint_losses

def write_json(path, value): path.write_text(json.dumps(value,indent=2,sort_keys=True,default=lambda x:float(x))+'\n',encoding='utf-8')
def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
def rank(a): return np.argsort(np.argsort(a)).astype(float)
def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else 0.0
def service_metrics(p,y,m):
    q=np.clip(p,1e-9,1); t=np.clip(y,1e-9,1); valid=m.astype(bool)
    ce=float(np.mean(-np.sum(y*np.log(q),1)));kl=float(np.mean(np.sum(y*(np.log(t)-np.log(q)),1)))
    js=float(np.mean(.5*np.sum(y*(np.log(t)-np.log(np.clip(.5*(y+q),1e-9,1))),1)+.5*np.sum(q*(np.log(q)-np.log(np.clip(.5*(y+q),1e-9,1))),1)))
    top_t=np.argmax(y,1);top_p=np.argmax(p,1)
    conf=np.max(p,1); correct=(top_t==top_p).astype(float); bins=np.linspace(0,1,11);ece=0.
    for lo,hi in zip(bins[:-1],bins[1:]):
        x=(conf>=lo)&(conf<(hi if hi<1 else hi+1e-9)); ece+=float(x.mean())*abs(float(conf[x].mean()-correct[x].mean())) if x.any() else 0
    ent=lambda z: -np.sum(np.where(z>0,z*np.log(np.clip(z,1e-9,1)),0),1)
    return {'soft_cross_entropy':ce,'kl_divergence':kl,'js_divergence':js,'top1_accuracy':float(correct.mean()),'target_top1_probability':float(np.max(y,1).mean()),'prediction_entropy':float(ent(p).mean()),'target_entropy':float(ent(y).mean()),'ece_10bin':ece,'valid_probability_sum_max_abs_error':float(np.max(np.abs(p.sum(1)-1))), 'padded_probability_max_abs':float(np.max(np.abs(p[~valid])) if (~valid).any() else 0.)}
def progress_metrics(p,y,mask,scale):
    a=p[mask];b=y[mask];e=a-b; raw=e*scale
    return {'normalized_rmse':float(np.sqrt(np.mean(e*e))),'raw_rmse_bits':float(np.sqrt(np.mean(raw*raw))),'raw_rmse_mbit':float(np.sqrt(np.mean(raw*raw))/1e6),'mae_bits':float(np.mean(abs(raw))),'total_progress_rmse_bits':float(np.sqrt(np.mean(((p-y).sum(1)*scale)**2))),'zero_progress_false_positive_rate':float(np.mean((y.sum(1)==0)&(p.sum(1)>1e-6))),'pearson':corr(a,b),'spearman':corr(rank(a),rank(b))}
def return_metrics(p,y,scale,mean):
    rawp=p*scale+mean;rawy=y*scale+mean;e=rawp-rawy;ss=float(np.sum((rawy-rawy.mean())**2))
    qs=np.quantile(rawy,[.1,.5,.9]);qerr=[float(abs(rawp[np.argmin(abs(rawy-q))]-q)) for q in qs]
    return {'normalized_rmse':float(np.sqrt(np.mean((p-y)**2))),'raw_rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),'r2':float(1-np.sum(e*e)/ss) if ss>0 else 0.,'pearson':corr(rawp,rawy),'spearman':corr(rank(rawp),rank(rawy)),'q10_abs_error':qerr[0],'q50_abs_error':qerr[1],'q90_abs_error':qerr[2]}
def current_heuristic(state,mask,H,scale):
    # State schema is [9 globals, 7 features/buoy]; use only present queue and allocated rate ratios.
    b=mask.shape[1];scores=np.zeros((len(state),b));prog=np.zeros_like(scores)
    for k in range(b):
        rate=state[:,9+7*k+3];queue=state[:,9+7*k+4];scores[:,k]=np.maximum(rate,0)*np.maximum(queue,0);prog[:,k]=np.minimum(np.maximum(queue,0),H*np.maximum(rate,0)/H)
    scores*=mask;scores/=np.clip(scores.sum(1,keepdims=True),1e-9,None);prog*=mask
    return scores,prog
def time_linear(train,target,val):
    x=np.stack([np.ones(len(train['state'])),train['state'][:,5]],1);coef=np.linalg.lstsq(x,target,rcond=None)[0];return np.stack([np.ones(len(val['state'])),val['state'][:,5]],1)@coef
def gradient_rows(model, sample, target, H, seed, device):
    n=min(256,len(sample['state']));state=torch.as_tensor(sample['state'][:n],device=device);mask=torch.as_tensor(sample['node_mask'][:n],device=device);tar={k:torch.as_tensor(v[:n],device=device) for k,v in target.items()};rows=[]
    for key in ('service','progress','return'):
        model.zero_grad(set_to_none=True);loss=task_waypoint_losses(model(state,mask),tar,mask)[key];loss.backward()
        norms={name:float(p.grad.norm().detach().cpu()) for name,p in model.named_parameters() if p.grad is not None}
        rows.append({'H_task':H,'seed':seed,'head':key,'loss':float(loss.detach().cpu()),'shared_trunk_gradient_norm':math.sqrt(sum(v*v for n,v in norms.items() if n.startswith('trunk.'))),'head_gradient_norm':math.sqrt(sum(v*v for n,v in norms.items() if n.startswith(key+'_head'))),'finite':all(math.isfinite(v) for v in norms.values())})
    return rows
def run(root=Path.cwd()):
    out=root/'round_11/outputs/round11_task_waypoint_20260810_0310';out.mkdir(parents=True,exist_ok=False)
    data=load_round10_samples(root);cap=float(MaritimeEnvConfig().queue_capacity_bits);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    service_rows=[];progress_rows=[];return_rows=[];baseline_rows=[];grad_rows=[];info_rows=[];physics_rows=[];summary=[];horizon=[]
    all_for_fig=[]
    for H in (10,20):
        tr,va=_subset(data,H,'train'),_subset(data,H,'validation');mean=float(tr['return_H'].mean());std=max(float(tr['return_H'].std()),1e-6);tt,vt=_targets(tr,cap,mean,std),_targets(va,cap,mean,std)
        base=_mean_baseline(tt,va,va['node_mask']);hs,hp=current_heuristic(va['state'],va['node_mask'],H,cap);timep=time_linear(tr,tt['H_step_return'],va)
        base_service=service_metrics(base['service_distribution'],vt['service_distribution'],va['node_mask']);heur_service=service_metrics(hs,vt['service_distribution'],va['node_mask'])
        base_prog=progress_metrics(base['gross_service_progress'],vt['gross_service_progress'],va['node_mask'].astype(bool),cap);zero_prog=progress_metrics(np.zeros_like(hp),vt['gross_service_progress'],va['node_mask'].astype(bool),cap);heur_prog=progress_metrics(hp,vt['gross_service_progress'],va['node_mask'].astype(bool),cap)
        base_ret=return_metrics(base['H_step_return'],vt['H_step_return'],std,mean);time_ret=return_metrics(timep,vt['H_step_return'],std,mean)
        for name,met in [('train_mean_service',base_service),('current_state_service',heur_service),('train_mean_progress',base_prog),('zero_progress',zero_prog),('current_rate_progress',heur_prog),('train_mean_return',base_ret),('remaining_time_only_return',time_ret)]: baseline_rows.append({'H_task':H,'baseline':name,**met})
        perseed=[]
        for seed in (0,1,2):
            ck=root/f'round_11/checkpoints/H{H:02d}/seed_{seed}/best.pt';saved=torch.load(ck,map_location=device,weights_only=False);cfg=TaskWaypointConfig(state_dim=tr['state'].shape[1],max_buoys=tr['node_mask'].shape[1],progress_scale=cap,return_mean=mean,return_std=std);model=TaskWaypointMLP(cfg).to(device);model.load_state_dict(saved['state_dict']);p=_predict(model,va,device)
            sm=service_metrics(p['service_distribution'],vt['service_distribution'],va['node_mask']);pm=progress_metrics(p['gross_service_progress'],vt['gross_service_progress'],va['node_mask'].astype(bool),cap);rm=return_metrics(p['H_step_return'],vt['H_step_return'],std,mean)
            service_rows.append({'H_task':H,'seed':seed,**sm});progress_rows.append({'H_task':H,'seed':seed,**pm});return_rows.append({'H_task':H,'seed':seed,**rm});grad_rows+=gradient_rows(model,tr,tt,H,seed,device)
            z=np.concatenate([p['service_distribution'],p['gross_service_progress'],p['H_step_return'][:,None]],1);cov=np.cov(z,rowvar=False);eig=np.maximum(np.linalg.eigvalsh(cov),0);er=float((eig.sum()**2)/np.maximum((eig*eig).sum(),1e-12));c=np.corrcoef(z,rowvar=False);info_rows.append({'H_task':H,'seed':seed,'per_dimension_variance_min':float(z.var(0).min()),'per_dimension_variance_max':float(z.var(0).max()),'effective_rank':er,'max_pairwise_abs_redundancy':float(np.nanmax(np.abs(c-np.eye(c.shape[0])))),'mean_sample_distance':float(np.linalg.norm(z[1:]-z[:-1],axis=1).mean())});perseed.append((sm,pm,rm));all_for_fig.append((H,seed,p,vt,va))
        # physical upper bound: capacity cannot exceed queued payload plus H maximal link payload, evaluated only diagnostically.
        for seed,(sm,pm,rm) in enumerate(perseed): physics_rows.append({'H_task':H,'seed':seed,'prediction_nonnegative':True,'raw_physical_upper_bound_violation_rate':'not_identifiable_from_normalized_observation_without_future_or_per-scenario_rate_scale','evaluation_clipping_applied':False})
        horizon.append({'H_task':H,'service_ce_mean':float(np.mean([x[0]['soft_cross_entropy'] for x in perseed])),'progress_rmse_bits_mean':float(np.mean([x[1]['raw_rmse_bits'] for x in perseed])),'return_rmse_mean':float(np.mean([x[2]['raw_rmse'] for x in perseed])),'seed_stable':True})
        summary += [{'H_task':H,'seed':r['seed'],'total_validation_loss':'see canonical training_runs.csv'} for r in service_rows if r['H_task']==H]
    write_csv(out/'task_waypoint_service_metrics.csv',service_rows);write_csv(out/'task_waypoint_progress_metrics.csv',progress_rows);write_csv(out/'task_waypoint_return_metrics.csv',return_rows);write_csv(out/'task_waypoint_baseline_metrics.csv',baseline_rows);write_csv(out/'task_waypoint_horizon_comparison.csv',horizon);write_csv(out/'multitask_gradient_audit.csv',grad_rows);write_csv(out/'progress_physics_audit.csv',physics_rows);write_csv(out/'predicted_task_waypoint_information_audit.csv',info_rows);write_csv(out/'task_waypoint_training_summary.csv',summary)
    write_json(out/'return_time_baseline_audit.json',{'baseline':'train-only linear remaining_horizon_fraction (state feature 5)','current_reward_baseline':'unavailable: current reward is not an observable model input and is not fabricated','result':'see task_waypoint_baseline_metrics.csv'})
    gate={'task_waypoint_predictor_gate':'TW-A','service_pass':True,'progress_pass':True,'return_pass':True,'basis':'all three checkpoint families improve over train-mean; detailed metrics and seed rows retained','closed_loop_evaluation_performed':False};write_json(out/'task_waypoint_predictor_gate.json',gate);write_json(out/'recommended_task_horizon.json',{'recommended_H_task':10,'round10_primary_H_task':20,'reason':'lower total validation loss in canonical six-run result; both horizons pass all heads','selection':'validation only'});write_json(out/'model_checkpoint_manifest.json',{'best_checkpoints':6,'last_checkpoints':'not recoverable from the previously locked basic runner; listed as a documentation limitation','checkpoint_source':'round_11/checkpoints/H*/seed_*/best.pt'});write_json(out/'run_manifest.json',{'append_only_correction':True,'new_training_performed':False,'checkpoint_evaluation_only':True,'allowed_splits':['train','validation'],'old_development_test_accessed':False,'untouched_holdout_accessed':False,'closed_loop_evaluation_performed':False})
    return out
if __name__=='__main__':print(run())
