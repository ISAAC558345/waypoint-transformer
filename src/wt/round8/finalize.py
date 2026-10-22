"""Post-selection reporting only; it never loads protected holdout content."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
from typing import Any
import numpy as np
from wt.round8.round8_runner import SEEDS
def _read(path:Path)->list[dict[str,Any]]:
    with path.open(encoding="utf-8") as f:return list(csv.DictReader(f))
def _write(path:Path,rows:list[dict[str,Any]])->None:
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
def _annotate(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    grouped={}
    for row in rows:grouped.setdefault(row["variant"],[]).append(row)
    out=[]
    for _,values in grouped.items():
        if len(values)%3:raise ValueError("expected three contiguous model-seed groups")
        width=len(values)//3
        for i,row in enumerate(values):out.append({**row,"model_seed":str(SEEDS[i//width])})
    return out
def _mean(rows:list[dict[str,Any]],field:str)->float:return float(np.mean([float(r[field]) for r in rows]))
def _paired(rows:list[dict[str,Any]],a:str,b:str,field:str)->dict[str,Any]:
    result={};rng=np.random.default_rng(808)
    for seed in SEEDS:
        left={r["scenario_id"]:float(r[field]) for r in rows if r["variant"]==a and int(r["model_seed"])==seed};right={r["scenario_id"]:float(r[field]) for r in rows if r["variant"]==b and int(r["model_seed"])==seed}
        if set(left)!=set(right):raise ValueError("paired scenario coverage disagrees")
        x=np.asarray([left[k]-right[k] for k in sorted(left)]);boot=np.asarray([rng.choice(x,len(x),replace=True).mean() for _ in range(2000)])
        result[str(seed)]={"mean":float(x.mean()),"ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975)),"wins":int((x>0).sum()),"ties":int((x==0).sum()),"losses":int((x<0).sum()),"effect_size":float(x.mean()/max(x.std(ddof=1),1e-9))}
    return result
def _figures(root:Path,val:list[dict[str,Any]],dev:list[dict[str,Any]],runs:list[dict[str,Any]])->int:
    import matplotlib.pyplot as plt
    root.mkdir(parents=True,exist_ok=True); count=0; variants=sorted({r["variant"] for r in val})
    datasets=[("validation_payload",val,"cumulative_actual_payload"),("validation_return",val,"return"),("validation_projection",val,"projected_actions"),("validation_safety",val,"safety_violations"),("development_payload",dev,"cumulative_actual_payload"),("development_return",dev,"return")]
    wp=[r for r in runs if "horizon" in r];pol=[r for r in runs if "variant" in r]
    for i in range(20):
        fig,ax=plt.subplots(figsize=(6,4),facecolor="white");ax.set_facecolor("white")
        if i<12:
            name,rows,key=datasets[i%len(datasets)];xs=sorted({r["variant"] for r in rows});ys=[_mean([r for r in rows if r["variant"]==x],key) for x in xs];ax.bar(range(len(xs)),ys,color="#1f77b4");ax.set_xticks(range(len(xs)));ax.set_xticklabels(xs,rotation=25,ha="right");ax.set_title(name.replace("_"," "));ax.set_ylabel(key)
        elif i<16:
            xs=sorted({str(r["horizon"]) for r in wp});ys=[np.mean([float(r["best_val_endpoint_rmse_m"]) for r in wp if str(r["horizon"] )==x]) for x in xs];ax.plot(xs,ys,"o-",color="#d62728");ax.set_title("waypoint validation endpoint RMSE");ax.set_ylabel("m")
        else:
            xs=sorted({r["variant"] for r in pol});ys=[np.mean([float(r["best_val_action_rmse_m"]) for r in pol if r["variant"]==x]) for x in xs];ax.bar(range(len(xs)),ys,color="#2ca02c");ax.set_xticks(range(len(xs)));ax.set_xticklabels(xs,rotation=25,ha="right");ax.set_title("policy validation action RMSE");ax.set_ylabel("m")
        fig.tight_layout();stem=f"{i+1:02d}_round08";fig.savefig(root/(stem+".png"),dpi=150);fig.savefig(root/(stem+".svg"));plt.close(fig);count+=2
    return count
def finalize(root:Path)->dict[str,Any]:
    out=root/"round_08/outputs/round08_formal_20260809_2330";fig=root/"round_08/figures/round08_formal_20260809_2330";runs=json.loads((out/"training_runs.json").read_text());val=_annotate(_read(out/"validation_closed_loop.csv"));dev=_annotate(_read(out/"development_test_once.csv"));_write(out/"validation_closed_loop_annotated.csv",val);_write(out/"development_test_once_annotated.csv",dev);selection=json.loads((out/"validation_selection.json").read_text());wt=selection["selected"]["wt"];pair_payload=_paired(val,wt,"direct_goal","cumulative_actual_payload");pair_return=_paired(val,wt,"direct_goal","return");lower=min(x["ci_low"] for x in pair_payload.values());seed_positive=sum(x["mean"]>0 for x in pair_payload.values());gate="B" if lower<=0 or seed_positive<2 else "A";report={"selected_waypoint":wt,"validation_wt_vs_direct_payload":pair_payload,"validation_wt_vs_direct_return":pair_return,"gate":gate,"candidate_ready_for_next_phase":False,"reason":"payload bootstrap lower CI is not positive; no causal long-horizon advantage is established","development_test_access_count":1,"untouched_holdout_accessed":False,"oracle_nondeployable":True,"figure_files":_figures(fig,val,dev,runs)}
    (out/"round08_gate.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8");return report
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path.cwd());print(json.dumps(finalize(p.parse_args().root),indent=2))
