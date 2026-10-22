"""Simple white-background Round-6 scientific figures (PNG and SVG)."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

def _rows(path:Path)->list[dict[str,str]]:
    return list(csv.DictReader(path.open(encoding='utf-8')))

def _save(fig, path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path.with_suffix('.png'),dpi=160,bbox_inches='tight',facecolor='white'); fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',facecolor='white'); plt.close(fig)

def _bar(rows:list[dict[str,str]], metric:str, title:str, ylabel:str, path:Path, group:str='model')->None:
    values=defaultdict(list)
    for r in rows: values[r[group]].append(float(r[metric]))
    names=sorted(values); means=[np.mean(values[n]) for n in names]; fig,ax=plt.subplots(figsize=(8,4.5),facecolor='white'); ax.bar(range(len(names)),means,color='#286fb4'); ax.set_xticks(range(len(names)),names,rotation=28,ha='right'); ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis='y',alpha=.25); _save(fig,path)

def _scatter(rows:list[dict[str,str]], x:str,y:str,title:str,xlabel:str,ylabel:str,path:Path)->None:
    a=np.asarray([float(r[x]) for r in rows]); b=np.asarray([float(r[y]) for r in rows]); fig,ax=plt.subplots(figsize=(6,4.5),facecolor='white'); ax.scatter(a,b,s=10,alpha=.25,color='#c44e52'); ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=.25); _save(fig,path)

def generate_round6_figures(output_dir:Path, figures_dir:Path)->list[Path]:
    episodes=_rows(output_dir/'closed_loop_episode_metrics.csv'); steps=_rows(output_dir/'closed_loop_step_metrics.csv'); pairs=_rows(output_dir/'closed_loop_paired_comparison.csv'); ood=_rows(output_dir/'ood_diagnostics.csv'); result=[]
    specs=[('01_payload','cumulative_payload_bits','WT and controls: cumulative payload','Payload (bits)'),('02_return','episode_return','WT and controls: episode return','Episode return'),('06_history','episode_return','WT Hc=10 versus Hc=1 closed-loop return','Episode return'),('07_state','episode_return','WT and State-only closed-loop return','Episode return'),('08_transformer_mlp','episode_return','Causal Transformer versus MLP return','Episode return'),('19_goal','goal_error_final','Final reference-goal error','Normalized goal distance'),('20_seeds','episode_return','Per-model-seed episode return','Episode return')]
    for name,metric,title,ylabel in specs:
        path=figures_dir/name/'plot'; _bar(episodes,metric,title,ylabel,path); result.append(path)
    _bar(episodes,'episode_return','Return by reference-goal source','Episode return',figures_dir/'09_reference_goals'/'plot','reference_policy'); result.append(figures_dir/'09_reference_goals'/'plot')
    _bar(episodes,'cumulative_payload_bits','Payload by buoy scale','Payload (bits)',figures_dir/'10_scales'/'plot','num_buoys'); result.append(figures_dir/'10_scales'/'plot')
    _bar(episodes,'projection_rate','Action projection activation rate','Rate',figures_dir/'15_projection'/'plot'); result.append(figures_dir/'15_projection'/'plot')
    _scatter(steps,'waypoint_error_m','action_error_m','Waypoint error versus executed-action error','Waypoint endpoint error (m)','Action error (m)',figures_dir/'16_waypoint_action'/'plot'); result.append(figures_dir/'16_waypoint_action'/'plot')
    _scatter(episodes,'goal_error_final','episode_return','Goal error versus episode return','Goal error','Episode return',figures_dir/'17_waypoint_return'/'plot'); result.append(figures_dir/'17_waypoint_return'/'plot')
    _scatter(ood,'timestep','ood_score_normalized_feature_distance','Closed-loop rollout OOD score','Timestep','Normalized feature distance',figures_dir/'18_ood'/'plot'); result.append(figures_dir/'18_ood'/'plot')
    # paired delta distributions, values generated at scenario granularity.
    for name,metric,title in [('03_delta_payload','cumulative_payload_bits','Paired WT - Direct-Goal payload delta'),('04_delta_return','episode_return','Paired WT - Direct-Goal return delta'),('05_win_loss','episode_return','Paired WT win/tie/loss summary')]:
        row=next((r for r in pairs if r['metric']==metric),pairs[0]); value=float(row['paired_mean']); fig,ax=plt.subplots(figsize=(5,4),facecolor='white'); ax.bar(['mean delta'],[value],color='#2a9d8f'); ax.axhline(0,color='black',lw=.8); ax.set_title(title); ax.set_ylabel('Paired difference'); _save(fig,figures_dir/name/'plot'); result.append(figures_dir/name/'plot')
    # Representative recorded-state trajectory and time curves: a single fixed episode is explicitly labelled.
    subset=[r for r in steps if r['model']=='wt_transformer_hc10' and r['model_seed']=='0'][:128]
    for name,key,title,ylabel in [('11_trajectory','action_error_m','Representative WT trajectory action error','Action error (m)'),('12_payload_time','cumulative_payload_bits','Representative WT cumulative payload','Payload (bits)'),('13_queue_time','residual_queue_ratio','Representative WT residual queue','Residual queue ratio'),('14_waypoint_time','waypoint_error_m','Representative WT waypoint error','Waypoint error (m)')]:
        fig,ax=plt.subplots(figsize=(7,4),facecolor='white'); ax.plot([int(r['timestep']) for r in subset],[float(r[key]) for r in subset],color='#286fb4'); ax.set_title(title+' (seed=0 representative)'); ax.set_xlabel('Timestep'); ax.set_ylabel(ylabel); ax.grid(alpha=.25); _save(fig,figures_dir/name/'plot'); result.append(figures_dir/name/'plot')
    return result
