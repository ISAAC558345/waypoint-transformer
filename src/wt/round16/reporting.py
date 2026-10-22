"""Scientific PNG/SVG figures for the Round-16 dynamic planning audit."""
from __future__ import annotations
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def _read(p):
    with Path(p).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def _f(r,k):return float(r[k]) if r.get(k,"") else 0.
def _save(fig,d,n):d.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(d/f"{n}.png",dpi=160);fig.savefig(d/f"{n}.svg");plt.close(fig)
def _bar(d,n,lab,val,title,ylabel,color="#1976b9"):
 fig,ax=plt.subplots(figsize=(7.4,4.2));ax.bar(lab,val,color=color);ax.set(title=title,ylabel=ylabel);ax.grid(axis="y",alpha=.25);ax.tick_params(axis="x",rotation=30);_save(fig,d,n)

def create_figures(output:Path,figures:Path)->list[str]:
 o,d=Path(output),Path(figures);hor=_read(o/"dynamic_planning_horizon_value.csv");head=_read(o/"dynamic_planning_headroom.csv");dpg=_read(o/"drift_induced_planning_gain.csv");delay=_read(o/"dynamic_delayed_benefit.csv");ant=_read(o/"anticipatory_motion_audit.csv");visit=_read(o/"dynamic_visitation_order.csv");queue=_read(o/"queue_drift_coupling_audit.csv");geom=_read(o/"geometry_drift_coupling_audit.csv");oracle=_read(o/"causal_oracle_drift_gap.csv");runtime=_read(o/"dynamic_runtime_events.csv");reg=_read(o/"static_drift_regression_audit.csv");trace=_read(o/"dynamic_trajectory_steps.csv");tiers=["static","low","medium","high"]
 ex=[r for r in trace if r["drift_tier"]=="medium" and r["policy_id"]=="causal_H20"][:64];fig,ax=plt.subplots(figsize=(6,5))
 if ex:
  bp=np.asarray([json.loads(r["buoy_positions_m"]) for r in ex]);up=np.asarray([json.loads(r["position_after_m"]) for r in ex]);ax.plot(up[:,0],up[:,1],lw=2,label="UAV H20")
  for k in range(bp.shape[1]):ax.plot(bp[:,k,0],bp[:,k,1],"--",label=f"buoy {k}")
 ax.set(title="Observable buoy-drift trajectory example",xlabel="x (m)",ylabel="y (m)");ax.legend(fontsize=8);ax.grid(alpha=.25);_save(fig,d,"01_buoy_drift_trajectory_examples")
 _bar(d,"02_static_regression",["48 controls"],[sum(r["passed"]=="True" for r in reg)/len(reg)],"Static-drift physics regression","pass fraction","#4e9a68")
 fig,ax=plt.subplots(figsize=(7.5,4.3))
 for tier in tiers:
  rows=sorted([r for r in hor if r["drift_tier"]==tier and r["mode"]=="causal"],key=lambda r:int(r["horizon"]));ax.plot(["H"+r["horizon"] for r in rows],[_f(r,"mean_actual_payload_bits")/1e6 for r in rows],marker="o",label=tier)
 ax.set(title="Actual payload versus horizon by drift tier",ylabel="mean payload (Mbit)");ax.legend();ax.grid(alpha=.25);_save(fig,d,"03_payload_vs_horizon_by_drift")
 fig,ax=plt.subplots(figsize=(7.5,4.3))
 for h,c in ((5,"#77aadd"),(10,"#ddaa33"),(20,"#bb5566")):
  rows=[r for r in head if r["group_key"]=="overall" and int(r["horizon"])==h and r["drift_tier"] in tiers];ax.plot([r["drift_tier"] for r in rows],[_f(r,"mean_delta_payload_bits")/1e6 for r in rows],marker="o",label=f"H{h}",color=c)
 ax.axhline(0,color="black",lw=.8);ax.set(title="Causal horizon delta payload",ylabel="mean delta (Mbit)");ax.legend();ax.grid(alpha=.25);_save(fig,d,"04_delta_payload_h5_h10_h20")
 rows=[r for r in dpg if int(r["horizon"])==20];_bar(d,"05_drift_induced_planning_gain",[r["drift_tier"] for r in rows],[_f(r,"mean_drift_induced_planning_gain_bits")/1e6 for r in rows],"Drift-induced planning gain: H20","Mbit")
 _bar(d,"06_delayed_benefit_ratio",[r["drift_tier"] for r in delay],[_f(r,"delayed_benefit_ratio") for r in delay],"Delayed-benefit ratio versus drift","ratio","#a36d3b")
 rows=[r for r in ant if r["policy_id"]=="causal_H20"];_bar(d,"07_anticipatory_action_ratio",[r["drift_tier"] for r in rows],[_f(r,"anticipatory_action_ratio") for r in rows],"H20 anticipatory-action ratio","ratio","#7b69ab")
 _bar(d,"08_current_vs_future_target_direction",[r["drift_tier"] for r in rows],[_f(r,"mean_angle_to_future_degrees")-_f(r,"mean_angle_to_current_degrees") for r in rows],"H20 future-target angle minus current-target angle","degrees")
 fig,ax=plt.subplots(figsize=(6,5))
 for policy,c in (("causal_H1","#777777"),("causal_H20","#1976b9")):
  rows=[r for r in trace if r["drift_tier"]=="high" and r["policy_id"]==policy][:64];p=np.asarray([json.loads(r["position_after_m"]) for r in rows]);ax.plot(p[:,0],p[:,1],label=policy,color=c)
 ax.set(title="H1 versus H20 trajectory: high drift",xlabel="x (m)",ylabel="y (m)");ax.legend();ax.grid(alpha=.25);_save(fig,d,"09_h1_vs_h20_trajectories")
 rows=[r for r in visit if int(r["horizon"])==20];_bar(d,"10_visitation_order_differences",[r["drift_tier"] for r in rows],[_f(r,"visitation_order_disagreement_rate") for r in rows],"H20 versus H1 visitation-order disagreement","ratio")
 rows=[r for r in queue if r["drift_tier"]=="medium"];_bar(d,"11_queue_pressure_planning_gain",[r["group"] for r in rows],[_f(r,"mean_delta_payload_bits")/1e6 for r in rows],"Medium-drift H20 gain by queue pressure","Mbit")
 rows=[r for r in geom if r["drift_tier"]=="medium"];_bar(d,"12_geometry_planning_gain",[r["group"] for r in rows],[_f(r,"mean_delta_payload_bits")/1e6 for r in rows],"Medium-drift H20 gain by geometry","Mbit")
 rows=[r for r in head if r["drift_tier"]=="medium" and int(r["horizon"])==20 and r["group_key"]=="num_buoys"];_bar(d,"13_three_vs_five_buoys",[r["group_value"]+" buoys" for r in rows],[_f(r,"mean_delta_payload_bits")/1e6 for r in rows],"Medium-drift H20 gain by buoy count","Mbit")
 rows=[r for r in oracle if int(r["horizon"])==20];_bar(d,"14_causal_vs_oracle_drift",[r["drift_tier"] for r in rows],[_f(r,"oracle_minus_causal_payload_bits")/1e6 for r in rows],"Oracle minus causal payload: H20","Mbit","#bd5560")
 groups={}
 for r in runtime:groups.setdefault((r["mode"],r["horizon"]),[]).append(_f(r,"runtime_seconds"))
 _bar(d,"15_runtime_vs_horizon",[f"{m} H{h}" for m,h in groups],[np.mean(v)*1e3 for v in groups.values()],"Planner runtime versus horizon","mean milliseconds")
 return sorted(p.name for p in d.glob("*.png"))
__all__=["create_figures"]
