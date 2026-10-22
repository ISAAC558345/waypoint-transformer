"""Scientific figures and concise evidence report for Round-19."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE_NAMES = (
    "01_shared_spectrum_uplink_schematic", "02_desired_cross_geometry",
    "03_orthogonal_regression", "04_independent_vs_h1_by_interference",
    "05_horizon_payload_by_interference", "06_coordination_delta_vs_eta",
    "07_planning_delta_vs_eta", "08_k3_vs_k5_planning",
    "09_interference_stress", "10_isr_distribution", "11_conflict_state_gain",
    "12_spatial_deconfliction", "13_assignment_divergence",
    "14_queue_interference_coupling", "15_runtime_vs_horizon",
)


def _save(fig: Any, directory: Path, name: str) -> None:
    fig.tight_layout(); fig.savefig(directory / f"{name}.png", dpi=150); fig.savefig(directory / f"{name}.svg"); plt.close(fig)


def _bar(directory: Path, name: str, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> None:
    fig, axis = plt.subplots(figsize=(7, 4)); axis.bar(labels, values, color="#2878B5"); axis.set(title=title, ylabel=ylabel); axis.tick_params(axis="x", rotation=18); axis.grid(axis="y", alpha=.25); _save(fig, directory, name)


def _mean(rows: Sequence[Mapping[str, Any]], **where: Any) -> float:
    values = [float(row.get("payload_mbit", row.get("mean", row.get("future_payload_gain_bits", 0.0)))) for row in rows if all(row.get(key) == value for key, value in where.items())]
    return float(np.mean(values)) if values else 0.0


def render_figures(directory: Path, result: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    metrics, steps, coord, plan = result["metrics"], result["steps"], result["coordination_summary"], result["planning_summary"]
    tiers = ("orthogonal", "low", "medium", "full"); eta = (0., .25, .5, 1.)
    fig, axis = plt.subplots(figsize=(7, 4)); axis.axis("off"); axis.text(.03,.70,"buoy k_m  ── desired S_m ──►  UAV m",fontsize=16); axis.text(.03,.42,"buoy k_j  ── cross I_m ──►  UAV m",fontsize=16,color="#C14953"); axis.text(.03,.15,"S=Pβ,  I=ηΣPβ,  SINR=S/(N+I),  R=B log₂(1+SINR)",fontsize=12); _save(fig,directory,FIGURE_NAMES[0])
    sample=next(row for row in steps if row["spectrum_tier"]=="full"); pos=np.asarray(sample["uav_positions_m"]); fig,axis=plt.subplots(figsize=(5,5)); axis.scatter(pos[:,0],pos[:,1],c=["#00798C","#D1495B"],s=70); axis.plot(pos[:,0],pos[:,1],"--",color="#C14953"); axis.set(xlim=(0,200),ylim=(0,200),title="Current receiver geometry",xlabel="x (m)",ylabel="y (m)"); axis.grid(alpha=.25); _save(fig,directory,FIGURE_NAMES[1])
    _bar(directory,FIGURE_NAMES[2],["M=1 rows","M=2 policy rows","max error ×1e15"],[len(result["m1"]),len(result["orthogonal"]),max(row["max_absolute_error"] for row in result["orthogonal"])*1e15],"Exact orthogonal controls","value")
    _bar(directory,FIGURE_NAMES[3],tiers,[_mean(metrics,policy="joint_h1",spectrum_tier=t)-_mean(metrics,policy="independent_queue_rate",spectrum_tier=t) for t in tiers],"Joint H1 − Independent","payload delta (Mbit)")
    fig,axis=plt.subplots(figsize=(7,4)); width=.22; x=np.arange(len(tiers));
    for off,policy,color in ((-.22,"joint_h1","#2878B5"),(0,"joint_h5","#5B8E7D"),(.22,"joint_h10","#C14953")): axis.bar(x+off,[_mean(metrics,policy=policy,spectrum_tier=t) for t in tiers],width,label=policy,color=color)
    axis.set(xticks=x,xticklabels=tiers,title="Joint payload by horizon and interference",ylabel="payload (Mbit)"); axis.legend(); axis.grid(axis="y",alpha=.25); _save(fig,directory,FIGURE_NAMES[4])
    _bar(directory,FIGURE_NAMES[5],[str(value) for value in eta],[_mean(coord,spectrum_tier=t,comparison="joint_h1_minus_independent",group="overall") for t in tiers],"Coordination delta vs overlap η","payload delta (Mbit)")
    _bar(directory,FIGURE_NAMES[6],[str(value) for value in eta],[_mean(plan,spectrum_tier=t,comparison="joint_h10_minus_joint_h1",group="overall") for t in tiers],"H10 − H1 planning delta vs η","payload delta (Mbit)")
    _bar(directory,FIGURE_NAMES[7],["K=3","K=5"],[_mean(plan,spectrum_tier="medium",comparison="joint_h10_minus_joint_h1",group=g) for g in ("K=3","K=5")],"Medium shared H10 − H1 by K","payload delta (Mbit)")
    _bar(directory,FIGURE_NAMES[8],tiers,[_mean(plan,spectrum_tier=t,comparison="joint_h10_minus_joint_h1",group="interference_stress") for t in tiers],"Pre-registered interference stress","payload delta (Mbit)")
    fig,axis=plt.subplots(figsize=(7,4)); axis.hist([np.mean(row["isr"]) for row in steps if row["spectrum_tier"]!="orthogonal"],bins=20,color="#C14953"); axis.axvline(.25,color="black",lw=1,label="frozen conflict threshold"); axis.set(title="ISR distribution",xlabel="I/S"); axis.legend(); _save(fig,directory,FIGURE_NAMES[9])
    _bar(directory,FIGURE_NAMES[10],["H5-H1","H10-H1"],[float(np.mean([row["future_payload_gain_bits"] for row in result["conflict"] if row["comparison"]==kind]))/1e6 for kind in ("joint_h5-joint_h1","joint_h10-joint_h1")],"Conflict-state future payload","delta (Mbit)")
    spatial_values = [
        float(np.mean([row["receiver_separation_h10_m"] - row["receiver_separation_h1_m"] for row in result["spatial"]])),
        float(np.mean([row["isr_reduction"] for row in result["spatial"]])),
        float(np.mean([row["sinr_improvement"] for row in result["spatial"]])),
    ]
    _bar(directory,FIGURE_NAMES[11],["receiver sep Δ","ISR reduction","SINR Δ"],spatial_values,"Spatial deconfliction indicators","difference")
    _bar(directory,FIGURE_NAMES[12],["H5-H1","H10-H1"],[float(np.mean([row["assignment_disagreement_rate"] for row in result["assignment"] if row["comparison"]==kind])) for kind in ("joint_h5-joint_h1","joint_h10-joint_h1")],"Assignment divergence","rate")
    _bar(directory,FIGURE_NAMES[13],tiers,[float(np.mean([float(row["delayed_coordination_benefit"]) for row in result["queue"] if row["spectrum_tier"]==tier])) if any(row["spectrum_tier"]==tier for row in result["queue"]) else 0.0 for tier in tiers],"High-queue delayed coordination benefit","ratio")
    _bar(directory,FIGURE_NAMES[14],["Independent","H1","H5","H10"],[float(np.mean([row["ms_per_decision"] for row in result["runtime"] if row["policy"]==policy])) for policy in ("independent_queue_rate","joint_h1","joint_h5","joint_h10")],"Runtime by horizon","ms per decision")
    if len(list(directory.glob("*.png"))) != 15 or len(list(directory.glob("*.svg"))) != 15: raise RuntimeError("Round-19 must render exactly fifteen PNG/SVG figure pairs")


def write_report(path: Path, result: Mapping[str, Any]) -> None:
    metrics, coord, plan = result["metrics"], result["coordination_summary"], result["planning_summary"]
    tiers=("orthogonal","low","medium","full")
    def payload(policy: str,tier: str) -> float: return _mean(metrics,policy=policy,spectrum_tier=tier)
    def stat(rows: Sequence[Mapping[str, Any]],tier: str,comparison: str,group: str="overall") -> Mapping[str,Any]: return next(row for row in rows if row["spectrum_tier"]==tier and row["comparison"]==comparison and row["group"]==group)
    lines=["# Round-19 Shared-Spectrum Multi-UAV Revalidation","","## Controls","","Round-18 lock/source/output hashes were verified before the audit. M=2, static buoy/static channel, fixed power/noise/path-loss/bandwidth, shared queues, exclusive matching, 15 m safety, and the frozen action/H={1,5,10} search are retained. `orthogonal` delegates exactly to Round-18; shared uplinks use buoy-to-UAV cross interference only. No training, HGNN, PPO/MARL, power control, protected-data access, or Round-20 work occurred.","","## Payload by coupling tier","","| tier | Independent | Joint H1 | Joint H5 | Joint H10 |","| --- | ---: | ---: | ---: | ---: |"]
    for tier in tiers: lines.append(f"| {tier} | {payload('independent_queue_rate',tier):.3f} | {payload('joint_h1',tier):.3f} | {payload('joint_h5',tier):.3f} | {payload('joint_h10',tier):.3f} |")
    lines.extend(["","## Paired headroom","", "| tier | H1−Independent [95% CI] | H5−H1 [95% CI] | H10−H1 [95% CI] |", "| --- | --- | --- | --- |"])
    for tier in tiers:
        c,h5,h10=stat(coord,tier,"joint_h1_minus_independent"),stat(plan,tier,"joint_h5_minus_joint_h1"),stat(plan,tier,"joint_h10_minus_joint_h1")
        lines.append(f"| {tier} | {c['mean']:.3f} [{c['ci95_low']:.3f}, {c['ci95_high']:.3f}] | {h5['mean']:.3f} [{h5['ci95_low']:.3f}, {h5['ci95_high']:.3f}] | {h10['mean']:.3f} [{h10['ci95_low']:.3f}, {h10['ci95_high']:.3f}] |")
    gates=(result["coordination_gate"],result["planning_gate"],result["realism_gate"],result["coupling_gate"],result["readiness"])
    lines.extend(["","## Registered gates","",f"- Interference Coordination: **{gates[0]['interference_coordination_gate']}**.",f"- Interference Planning Necessity: **{gates[1]['interference_planning_necessity_gate']}**.",f"- Realistic low/medium planning headroom: **{gates[2]['realistic_interference_planning_headroom']}**.",f"- Multi-UAV Coupling v2: **{gates[3]['multi_uav_coupling_gate_v2']}**.",f"- Shared-spectrum offline dataset ready: **{gates[4]['multi_uav_shared_spectrum_offline_dataset_ready']}**.","",f"The frozen next-step rule yields: `{gates[4]['recommended_next_step']}`.","", "## Mechanism diagnostics","",f"Mean H10 ISR reduction relative to H1: {np.mean([row['isr_reduction'] for row in result['spatial']]):.4f}. Mean assignment H10/H1 divergence: {np.mean([row['assignment_disagreement_rate'] for row in result['assignment'] if row['comparison']=='joint_h10-joint_h1']):.3f}. High-queue delayed-benefit ratio: {np.mean([float(row['delayed_coordination_benefit']) for row in result['queue']]) if result['queue'] else 0.0:.3f}.","", "All interpretation remains conditional on the registered gates; a full-shared-only result is not a realistic planning claim."])
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


__all__=["FIGURE_NAMES","render_figures","write_report"]
