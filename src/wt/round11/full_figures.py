"""Generate the required append-only Round-11 audit figures from CSV evidence."""
from __future__ import annotations
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
def rows(p):
 with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def run(root=Path.cwd()):
 out=root/'round_11/outputs/round11_task_waypoint_20260810_0310';fig=root/'round_11/figures/round11_task_waypoint_20260810_0320';fig.mkdir(parents=True,exist_ok=False)
 service=rows(out/'task_waypoint_service_metrics.csv');prog=rows(out/'task_waypoint_progress_metrics.csv');ret=rows(out/'task_waypoint_return_metrics.csv');base=rows(out/'task_waypoint_baseline_metrics.csv');grad=rows(out/'multitask_gradient_audit.csv');info=rows(out/'predicted_task_waypoint_information_audit.csv')
 specs=[(service,'soft_cross_entropy','Service CE'),(service,'js_divergence','Service JS divergence'),(service,'top1_accuracy','Service top-1 accuracy'),(service,'prediction_entropy','Predicted service entropy'),(service,'target_entropy','Target service entropy'),(prog,'raw_rmse_mbit','Progress RMSE (Mbit)'),(prog,'mae_bits','Progress MAE (bit)'),(prog,'total_progress_rmse_bits','Total progress RMSE (bit)'),(prog,'pearson','Progress Pearson correlation'),(ret,'raw_rmse','Return RMSE'),(ret,'mae','Return MAE'),(ret,'r2','Return R2'),(ret,'pearson','Return Pearson correlation'),(grad,'shared_trunk_gradient_norm','Shared trunk gradient norm'),(grad,'head_gradient_norm','Head gradient norm'),(info,'effective_rank','Predicted waypoint effective rank'),(info,'mean_sample_distance','Predicted waypoint sample distance'),(info,'max_pairwise_abs_redundancy','Predicted waypoint redundancy')]
 for i,(r,key,title) in enumerate(specs,1):
  f,a=plt.subplots(figsize=(6.4,4),facecolor='white');a.set_facecolor('white');x=list(range(len(r)));a.scatter(x,[float(z[key]) for z in r],color='#9ac7db',s=28,label='raw seed metrics');
  for h,c in [('10','#0f4c81'),('20','#b05c2e')]:
   vals=[float(z[key]) for z in r if z['H_task']==h];a.axhline(sum(vals)/len(vals),color=c,lw=2,label=f'H{h} mean')
  a.set_title(title);a.set_xlabel('seed/run');a.set_ylabel(title);a.legend();f.tight_layout();f.savefig(fig/f'{i:02d}.png',dpi=140);f.savefig(fig/f'{i:02d}.svg');plt.close(f)
 for i in (19,20):
  f,a=plt.subplots(figsize=(6.4,4),facecolor='white');a.set_facecolor('white');r=base; key='raw_rmse' if i==19 else 'soft_cross_entropy';vals=[float(z.get(key) or 0) for z in r];a.bar(range(len(vals)),vals,color='#9ac7db');a.set_title('Baseline comparison: '+key);a.set_xlabel('baseline row');a.set_ylabel(key);f.tight_layout();f.savefig(fig/f'{i:02d}.png',dpi=140);f.savefig(fig/f'{i:02d}.svg');plt.close(f)
 return 40
if __name__=='__main__':print(run())
