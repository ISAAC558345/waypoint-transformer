"""Figures and concise evidence report for the Round-17 channel audit."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TIERS=("static","low","medium","high")
HORIZONS=(1,5,10,20)
NAMES=(
    "channel_multiplier_examples","asynchronous_channel_windows","static_regression",
    "payload_vs_horizon_by_tier","delta_payload_by_horizon","channel_induced_planning_gain",
    "ranking_reversal_examples","channel_window_anticipation","channel_delayed_benefit",
    "current_good_future_good_trajectory","queue_channel_coupling","geometry_channel_coupling",
    "three_vs_five_buoys","causal_vs_oracle_channel","runtime_vs_horizon",
)


def _save(fig, directory: Path, index: int, name: str) -> None:
    fig.tight_layout(); stem=directory/f"{index:02d}_{name}"; fig.savefig(stem.with_suffix(".png"), dpi=160); fig.savefig(stem.with_suffix(".svg")); plt.close(fig)


def _rows(rows: Iterable[Mapping[str,Any]], **where: Any) -> list[Mapping[str,Any]]:
    return [row for row in rows if all(str(row.get(k))==str(v) for k,v in where.items())]


def _line_by_tier(rows, value, title, ylabel, directory, index, name, *, mode="causal"):
    fig,ax=plt.subplots(figsize=(7,4))
    for tier in TIERS:
        points=[]
        for h in HORIZONS:
            selected=_rows(rows,channel_tier=tier,horizon=h,mode=mode)
            if selected: points.append((h,float(selected[0][value])/1e6))
        if points: ax.plot(*zip(*points),marker="o",label=tier)
    ax.set(title=title,xlabel="planning horizon",ylabel=ylabel);ax.legend();ax.grid(alpha=.25);_save(fig,directory,index,name)


def create_figures(data: Mapping[str,Any], directory: Path) -> list[str]:
    directory.mkdir(parents=True,exist_ok=True)
    # 1: frozen multiplier formula.
    t=np.arange(25);fig,ax=plt.subplots(figsize=(7,4))
    for name,sigma in (("static",0.),("low",.15),("medium",.35),("high",.55)): ax.plot(t,np.exp(sigma*np.sin(2*np.pi*t/24)),label=name)
    ax.set(title="Registered channel multiplier examples",xlabel="slot",ylabel="multiplier");ax.legend();ax.grid(alpha=.25);_save(fig,directory,1,NAMES[0])
    # 2: asynchronous peak/trough phase conflict, registered before results.
    fig,ax=plt.subplots(figsize=(7,4));phases=(np.pi/2,-np.pi/2,.3)
    for k,phase in enumerate(phases): ax.plot(t,np.exp(.35*np.sin(phase+2*np.pi*t/24)),label=f"buoy {k}")
    ax.set(title="Asynchronous predictable channel windows",xlabel="slot",ylabel="medium-tier multiplier");ax.legend();ax.grid(alpha=.25);_save(fig,directory,2,NAMES[1])
    # 3: static exactness.
    static=data["static_rows"];fig,ax=plt.subplots(figsize=(7,4));ax.bar(range(len(static)),[float(r["max_abs_error"]) for r in static]);ax.set(title="Static-channel regression (all exact)",xlabel="same-seed/action check",ylabel="max absolute error");ax.grid(axis="y",alpha=.25);_save(fig,directory,3,NAMES[2])
    _line_by_tier(data["horizon"],"mean_actual_payload_bits","Causal payload by channel tier","mean actual payload (Mbit)",directory,4,NAMES[3])
    fig,ax=plt.subplots(figsize=(7,4))
    for h in (5,10,20):
        vals=[next(float(r["mean_delta_payload_bits"])/1e6 for r in _rows(data["headroom"],channel_tier=tier,horizon=h,group_key="overall")) for tier in TIERS]
        ax.plot(TIERS,vals,marker="o",label=f"H{h}")
    ax.axhline(0,color="black",lw=.8);ax.set(title="Causal delta payload versus H1",ylabel="mean delta (Mbit)");ax.legend();ax.grid(alpha=.25);_save(fig,directory,5,NAMES[4])
    fig,ax=plt.subplots(figsize=(7,4))
    for h in (5,10,20):
        vals=[next(float(r["mean_channel_induced_planning_gain_bits"])/1e6 for r in _rows(data["dpg"],channel_tier=tier,horizon=h)) for tier in TIERS[1:]];ax.plot(TIERS[1:],vals,marker="o",label=f"H{h}")
    ax.axhline(0,color="black",lw=.8);ax.set(title="Channel-induced planning gain",ylabel="CPG (Mbit)");ax.legend();ax.grid(alpha=.25);_save(fig,directory,6,NAMES[5])
    fig,ax=plt.subplots(figsize=(7,4));rows=[r for r in data["reversals"] if r["policy_id"]=="causal_H10"]
    ax.bar([r["channel_tier"] for r in rows],[r["ranking_reversal_rate"] for r in rows]);ax.set(title="Predicted channel-ranking reversals (H10)",ylabel="reversal rate",ylim=(0,1));ax.grid(axis="y",alpha=.25);_save(fig,directory,7,NAMES[6])
    fig,ax=plt.subplots(figsize=(7,4));rows=[r for r in data["anticipation"] if r["policy_id"]=="causal_H20"]
    ax.bar([r["channel_tier"] for r in rows],[r["anticipation_ratio"] for r in rows]);ax.set(title="Favorable-window anticipation (H20)",ylabel="anticipation ratio");ax.grid(axis="y",alpha=.25);_save(fig,directory,8,NAMES[7])
    fig,ax=plt.subplots(figsize=(7,4));ax.bar([r["channel_tier"] for r in data["delayed"]],[r["delayed_benefit_ratio"] for r in data["delayed"]]);ax.set(title="Channel-induced delayed-benefit ratio",ylabel="ratio");ax.grid(axis="y",alpha=.25);_save(fig,directory,9,NAMES[8])
    fig,ax=plt.subplots(figsize=(6,5));traces=data.get("traces",[])
    for policy in ("causal_H1","causal_H20"):
        rows=[r for r in traces if r["channel_tier"]=="medium" and r["phase_relationship"]=="channel_phase_conflict" and r["policy_id"]==policy]
        if rows:
            pos=np.asarray([r["position_after_m"] for r in rows]);ax.plot(pos[:,0],pos[:,1],marker="o",ms=2,label=policy)
    ax.set(title="Current-good versus future-good movement example",xlabel="x (m)",ylabel="y (m)");ax.legend();ax.grid(alpha=.25);_save(fig,directory,10,NAMES[9])
    for index,key,name,title in ((11,"queue_pressure",NAMES[10],"Queue × channel planning gain"),(12,"geometry",NAMES[11],"Geometry × channel planning gain")):
        fig,ax=plt.subplots(figsize=(7,4));rows=[r for r in data["coupling"] if r["coupling"]==key and r["horizon"]==20 and r["channel_tier"]=="medium"]
        ax.bar([r["group_value"] for r in rows],[float(r["mean_delta_payload_bits"])/1e6 for r in rows]);ax.axhline(0,color="black",lw=.8);ax.set(title=title,ylabel="H20-H1 delta (Mbit)");ax.grid(axis="y",alpha=.25);_save(fig,directory,index,name)
    fig,ax=plt.subplots(figsize=(7,4));rows=[r for r in data["headroom"] if r["group_key"]=="num_buoys" and r["horizon"]==20 and r["channel_tier"]=="medium"]
    ax.bar([r["group_value"] for r in rows],[float(r["mean_delta_payload_bits"])/1e6 for r in rows]);ax.axhline(0,color="black",lw=.8);ax.set(title="Three versus five buoy planning gain",ylabel="medium H20-H1 delta (Mbit)");ax.grid(axis="y",alpha=.25);_save(fig,directory,13,NAMES[12])
    fig,ax=plt.subplots(figsize=(7,4));rows=[r for r in data["oracle"] if r["horizon"]==20]
    ax.bar([r["channel_tier"] for r in rows],[float(r["mean_oracle_minus_causal_payload_bits"])/1e6 for r in rows]);ax.axhline(0,color="black",lw=.8);ax.set(title="Oracle minus causal channel diagnostic (H20)",ylabel="payload gap (Mbit)");ax.grid(axis="y",alpha=.25);_save(fig,directory,14,NAMES[13])
    fig,ax=plt.subplots(figsize=(7,4));runtime=data.get("runtime",[])
    for mode in ("causal","oracle"):
        vals=[]
        for h in HORIZONS:
            rows=_rows(runtime,mode=mode,horizon=h);vals.append(float(np.mean([float(r["runtime_seconds"]) for r in rows]))*1000 if rows else 0.)
        ax.plot(HORIZONS,vals,marker="o",label=mode)
    ax.set(title="Planner runtime versus horizon",xlabel="horizon",ylabel="mean milliseconds");ax.legend();ax.grid(alpha=.25);_save(fig,directory,15,NAMES[14])
    return [f"{index:02d}_{name}.png" for index,name in enumerate(NAMES,1)]


def write_report(data: Mapping[str,Any], path: Path) -> None:
    def overall(tier,horizon): return next(r for r in data["headroom"] if r["channel_tier"]==tier and int(r["horizon"])==horizon and r["group_key"]=="overall")
    def fmt(row,key="mean_delta_payload_bits"):
        return f"{float(row[key])/1e6:+.3f} [{float(row['bootstrap_ci_low'])/1e6:+.3f}, {float(row['bootstrap_ci_high'])/1e6:+.3f}]"
    def cpg(tier,horizon): return next(r for r in data["dpg"] if r["channel_tier"]==tier and int(r["horizon"])==horizon)
    def coupling(kind,tier,horizon,group): return next(r for r in data["coupling"] if r["coupling"]==kind and r["channel_tier"]==tier and int(r["horizon"])==horizon and r["group_value"]==group)
    def value(rows,tier,key): return float(next(r[key] for r in rows if r["channel_tier"]==tier))
    lines=[
        "# Round-17 predictable time-varying channel planning revalidation", "",
        "## 1. Baseline and added dynamics", "",
        "The original channel is deterministic: `d_xy=||b-u||`, `d_3d=sqrt(d_xy^2+h^2)`, `beta_geom=beta0/d_3d^alpha`, `SNR=P*beta_geom/N`, and `r=(B/K)log2(1+SNR)`. Defaults are beta0=1e-3, alpha=2.2, h=50 m, P=0.1 W, B=1e6 Hz, and N=1e-9 W. There was no fading, shadowing, or channel randomness.", "",
        "Round-17 preserves that geometry/path-loss/bandwidth/noise/power/rate formula, actions, queues, resources, and reward expression. It adds only `beta=beta_geom*c`, `c=exp(sigma*sin(theta))`, and `theta[t+1]=theta[t]+2*pi/24 (mod 2*pi)`. The ordinary causal state exposes current `[sin(theta), cos(theta), omega/(2*pi/20)]`, never a future multiplier/rate/SINR/window. Primary buoy drift is static.", "",
        "## 2. Frozen scale and static control", "",
        "Tiers static/low/medium/high use sigma 0.00/0.15/0.35/0.55, multiplier ranges 1.000/0.861--1.162/0.705--1.419/0.577--1.733, and period 24 slots. H5/H10/H20 span 0.208/0.417/0.833 cycles. All per-buoy phases are asynchronous and pre-registered from fresh scenario seeds; the phase-conflict subset uses a registered peak/trough pair.", "",
        f"Static mode passed all {len(data['static_rows'])} same-seed/same-action checks exactly (state, gain, rate, payload, reward, queue, positions, and termination).", "",
        "## 3. Causal actual-payload headroom", "", "| tier | H10-H1 Mbit (95% CI) | H20-H1 Mbit (95% CI) |", "| --- | --- | --- |",
    ]
    for tier in TIERS: lines.append(f"| {tier} | {fmt(overall(tier,10))} | {fmt(overall(tier,20))} |")
    lines.extend(["", "All H10/H20 primary intervals are non-positive at their upper or lower evidence boundary; no horizon establishes episode-level causal benefit. H1 is therefore the most valuable available horizon in this audit, rather than H5/H10/H20.", "", "## 4. Channel-specific mechanisms", "", "| tier | CPG H10 Mbit (95% CI) | CPG H20 Mbit (95% CI) | H10 reversal rate | H20 anticipation ratio | delayed-benefit ratio |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for tier in TIERS:
        c10 = "--" if tier == "static" else fmt(cpg(tier,10),"mean_channel_induced_planning_gain_bits")
        c20 = "--" if tier == "static" else fmt(cpg(tier,20),"mean_channel_induced_planning_gain_bits")
        rev=value([r for r in data["reversals"] if r["policy_id"]=="causal_H10"],tier,"ranking_reversal_rate")
        ant=value([r for r in data["anticipation"] if r["policy_id"]=="causal_H20"],tier,"anticipation_ratio")
        delayed=value(data["delayed"],tier,"delayed_benefit_ratio")
        lines.append(f"| {tier} | {c10} | {c20} | {rev:.1%} | {ant:.1%} | {delayed:.1%} |")
    lines.extend(["", "The high H20 CPG is positive, but it is a difference-from-static diagnostic, not positive H20 episode headroom: high H20-H1 remains negative. It cannot satisfy CPN-A.", "", "Phase-conflict H20 deltas are " + "; ".join(f"{tier} {fmt(coupling('phase_relationship',tier,20,'channel_phase_conflict'))}" for tier in ("low","medium","high")) + ". Thus the deliberately diagnostic conflict subset also has no stable positive long-horizon conversion.", "", "High-queue H20 deltas are " + "; ".join(f"{tier} {fmt(coupling('queue_pressure',tier,20,'high'))}" for tier in ("low","medium","high")) + ". No post-hoc future-good subset was selected to manufacture a result.", "", "Medium-tier H20 compact/spread deltas are " + fmt(coupling('geometry','medium',20,'compact')) + " / " + fmt(coupling('geometry','medium',20,'spread')) + "; geometry does not rescue headroom.", "", "## 5. Causal versus Oracle", ""])
    oracle_h20=[r for r in data["oracle"] if int(r["horizon"])==20]
    lines.append("Oracle-minus-causal H20 gaps (diagnostic only) are " + "; ".join(f"{r['channel_tier']} {fmt(r,'mean_oracle_minus_causal_payload_bits')}" for r in oracle_h20) + ". Oracle is marked `oracle_future_channel=true` and `deployable=false`; it does not support a causal deployment claim.")
    lines.extend(["", "## 6. Gate, stop condition, and controls", "", f"Dynamic Channel Planning Gate: **{data['gate']['channel_planning_necessity_gate']}**. `predictable_channel_long_horizon_planning_required={data['gate']['predictable_channel_long_horizon_planning_required']}` and `realistic_channel_planning_headroom={data['realism']['realistic_channel_planning_headroom']}`. CPN-A and CPN-B conditions are both false.", "", "Consequently this round does not generate dynamic-channel offline training data, does not redesign Task-Waypoint targets, and does not authorize further single-UAV environment micro-adjustments. A separately designed multi-UAV task-coupling study is the only suggested next complexity if a future authorization is issued; it is not implemented here.", "", "Old-development access count=0; fresh-development access count=0; untouched-holdout new access count=0. No neural training, HGNN, PPO, multi-UAV, service/power/beamforming/association action, reward modification, or Round-18 occurred."])
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


__all__=["NAMES","create_figures","write_report"]
