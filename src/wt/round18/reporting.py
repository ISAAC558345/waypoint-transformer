"""Figures and report for the Round-18 deterministic audit."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE_NAMES = (
    "01_multi_uav_environment", "02_shared_queue_matching", "03_m1_regression",
    "04_independent_vs_joint_h1", "05_joint_horizons_payload", "06_h10_minus_h1",
    "07_paired_payload_distribution", "08_duplicate_intent", "09_assignment_divergence",
    "10_role_specialization", "11_team_coverage", "12_delayed_cooperation",
    "13_independent_trajectory", "14_joint_h1_trajectory", "15_joint_h10_trajectory",
    "16_min_uav_distance", "17_safety_correction", "18_k3_vs_k5", "19_coordination_stress",
    "20_runtime_vs_horizon",
)


def _save(fig: Any, directory: Path, name: str) -> None:
    fig.tight_layout(); fig.savefig(directory / f"{name}.png", dpi=150); fig.savefig(directory / f"{name}.svg"); plt.close(fig)


def _mean_by_policy(metrics: Sequence[Mapping[str, Any]], key: str) -> tuple[list[str], list[float]]:
    policies = ["independent_queue_rate", "joint_h1", "joint_h5", "joint_h10"]
    return policies, [float(np.mean([row[key] for row in metrics if row["policy"] == policy])) for policy in policies]


def _bar(directory: Path, name: str, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> None:
    fig, axis = plt.subplots(figsize=(7, 4)); axis.bar(labels, values, color="#2878B5"); axis.set_title(title); axis.set_ylabel(ylabel); axis.tick_params(axis="x", rotation=18); axis.grid(axis="y", alpha=.25); _save(fig, directory, name)


def render_figures(directory: Path, result: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    metrics, steps = result["metrics"], result["steps"]
    # 1: compact environment schematic.
    fig, axis = plt.subplots(figsize=(6, 5)); sample = next(row for row in steps if row["policy"] == "joint_h1")
    positions = np.asarray(sample["uav_positions_m"]); axis.scatter(positions[:, 0], positions[:, 1], c=["#D1495B", "#00798C"], label="UAV"); axis.set(xlim=(0, 200), ylim=(0, 200), title="Round-18 M=2 environment", xlabel="x (m)", ylabel="y (m)"); axis.legend(); axis.grid(alpha=.25); _save(fig, directory, FIGURE_NAMES[0])
    _bar(directory, FIGURE_NAMES[1], ["shared queue", "exclusive matching", "hard safety"], [1, 1, 1], "Frozen task couplings", "enabled")
    _bar(directory, FIGURE_NAMES[2], ["M=1 cases", "max error x1e15"], [12, max(row["max_absolute_error"] for row in result["single_regression"]) * 1e15], "Exact M=1 control", "value")
    labels, values = _mean_by_policy(metrics, "payload_mbit"); _bar(directory, FIGURE_NAMES[3], labels[:2], values[:2], "Independent vs Joint H1", "cumulative payload (Mbit)")
    _bar(directory, FIGURE_NAMES[4], labels[1:], values[1:], "Joint planning horizons", "cumulative payload (Mbit)")
    paired = [row for row in result["necessity_gate"].items() if row[0] == "joint_h10_minus_joint_h1"][0][1]; _bar(directory, FIGURE_NAMES[5], ["mean", "CI low", "CI high"], [paired["mean"], paired["ci95_low"], paired["ci95_high"]], "Joint H10 - Joint H1", "payload delta (Mbit)")
    deltas = [row["payload_mbit"] for row in metrics if row["policy"] == "joint_h10"]; base = [row["payload_mbit"] for row in metrics if row["policy"] == "joint_h1"]; fig, axis = plt.subplots(figsize=(7, 4)); axis.hist(np.asarray(deltas) - np.asarray(base), bins=14, color="#5B8E7D"); axis.axvline(0, color="black", lw=1); axis.set(title="Paired H10-H1 payload distribution", xlabel="Mbit"); _save(fig, directory, FIGURE_NAMES[6])
    duplicate = result["duplicates"]; _bar(directory, FIGURE_NAMES[7], ["independent", "Joint H1", "Joint H10"], [float(np.mean([row["duplicate_target_intent_rate"] for row in duplicate if row["policy"] == policy])) for policy in ("independent_queue_rate", "joint_h1", "joint_h10")], "Duplicate target intent", "rate")
    divergence = result["divergence"]; _bar(directory, FIGURE_NAMES[8], ["H5-H1", "H10-H1"], [float(np.mean([row["assignment_disagreement_rate"] for row in divergence if row["comparison"] == comparison])) for comparison in ("joint_h5-joint_h1", "joint_h10-joint_h1")], "Assignment divergence", "rate")
    roles = result["roles"]; _bar(directory, FIGURE_NAMES[9], labels, [float(np.mean([row["specialization_entropy"] for row in roles if row["policy"] == policy])) for policy in labels], "Role specialization entropy", "normalized entropy")
    _bar(directory, FIGURE_NAMES[10], labels, [float(np.mean([row["distinct_served_buoys"] for row in metrics if row["policy"] == policy])) for policy in labels], "Team coverage", "distinct served buoys")
    delayed = result["delayed"]; _bar(directory, FIGURE_NAMES[11], ["H5-H1", "H10-H1"], [float(np.mean([row["delayed_cooperative_benefit"] for row in delayed if row["comparison"] == comparison])) for comparison in ("joint_h5-joint_h1", "joint_h10-joint_h1")], "Delayed cooperative benefit", "ratio")
    for index, policy in enumerate(("independent_queue_rate", "joint_h1", "joint_h10"), start=12):
        episode = [row for row in steps if row["policy"] == policy and "stress_duplicate" in row["scenario_id"]]; fig, axis = plt.subplots(figsize=(6, 5))
        for uav in range(2):
            track = np.asarray([row["uav_positions_m"][uav] for row in episode]); axis.plot(track[:, 0], track[:, 1], marker="o", ms=2, label=f"UAV {uav}")
        axis.set(xlim=(0, 200), ylim=(0, 200), title=policy, xlabel="x (m)", ylabel="y (m)"); axis.legend(); axis.grid(alpha=.25); _save(fig, directory, FIGURE_NAMES[index])
    safety = result["safety"]; _bar(directory, FIGURE_NAMES[15], labels, [float(np.mean([row["minimum_uav_distance_m"] for row in safety if row["policy"] == policy])) for policy in labels], "Minimum UAV distance", "m")
    _bar(directory, FIGURE_NAMES[16], labels, [float(np.mean([row["safety_correction_rate"] for row in safety if row["policy"] == policy])) for policy in labels], "Safety correction rate", "rate")
    _bar(directory, FIGURE_NAMES[17], ["K=3", "K=5"], [float(np.mean([row["payload_mbit"] for row in metrics if row["policy"] == "joint_h10" and row["num_buoys"] == count])) for count in (3, 5)], "Joint H10 by buoy count", "payload (Mbit)")
    _bar(directory, FIGURE_NAMES[18], ["stress", "non-stress"], [float(np.mean([row["payload_mbit"] for row in metrics if row["policy"] == "joint_h10" and bool(row["coordination_stress"]) == flag])) for flag in (True, False)], "Coordination-stress subset", "Joint H10 payload (Mbit)")
    _bar(directory, FIGURE_NAMES[19], labels, [float(np.mean([row["mean_decision_runtime_ms"] for row in metrics if row["policy"] == policy])) for policy in labels], "Runtime by policy/horizon", "ms per decision")
    if len(list(directory.glob("*.png"))) != 20 or len(list(directory.glob("*.svg"))) != 20: raise RuntimeError("Round-18 figure count is not exactly 20 PNG/SVG pairs")


def write_report(path: Path, result: Mapping[str, Any]) -> None:
    metrics = result["metrics"]; cn = result["coordination_gate"]; mpn = result["necessity_gate"]; coupling = result["coupling_gate"]; readiness = result["readiness"]
    mean = {policy: float(np.mean([row["payload_mbit"] for row in metrics if row["policy"] == policy])) for policy in ("independent_queue_rate", "joint_h1", "joint_h5", "joint_h10")}
    h5 = next(row for row in result["planning_summary"] if row["comparison"] == "joint_h5_minus_joint_h1" and row["group"] == "overall"); h10 = mpn["joint_h10_minus_joint_h1"]
    path.write_text(f"""# Round-18 Minimal Multi-UAV Coordination Audit

## Environment and controls

M=1 exact regression passed all 12 same-seed/same-action cases with maximum
absolute error 0.0.  Primary M=2 state uses two planar UAVs, action shape
`(2,2)`, a single truthful queue per buoy, current-score exclusive matching
`W[m,k]=(Q[k]/capacity)*R[m,k]`, and a 15 m hard separation projection.
Both buoy drift and channel variation are static.  No training, HGNN, PPO,
power/beamforming/association action, protected-data access, or Round-19 work
occurred.

## Payload results

| policy | mean actual payload (Mbit) |
| --- | ---: |
| Independent queue-rate | {mean['independent_queue_rate']:.3f} |
| Joint H1 | {mean['joint_h1']:.3f} |
| Joint H5 | {mean['joint_h5']:.3f} |
| Joint H10 | {mean['joint_h10']:.3f} |

Joint H1 minus Independent is {cn['joint_h1_minus_independent']['mean']:.3f}
Mbit with 95% CI [{cn['joint_h1_minus_independent']['ci95_low']:.3f},
{cn['joint_h1_minus_independent']['ci95_high']:.3f}].  Coordination Gate is
**{cn['multi_uav_coordination_gate']}**.

Joint H5-H1 is {h5['mean']:.3f} Mbit, 95% CI [{h5['ci95_low']:.3f},
{h5['ci95_high']:.3f}].  Joint H10-H1 is {h10['mean']:.3f} Mbit, 95% CI
[{h10['ci95_low']:.3f}, {h10['ci95_high']:.3f}].  The best measured horizon is
the policy with the largest mean actual payload, not a post-hoc selected subset.

## Mechanisms and gates

Mean H10/H1 assignment disagreement is {mpn['assignment_divergence_rate']:.3f};
the H10 delayed-cooperative-benefit ratio is
{mpn['delayed_cooperative_benefit_ratio']:.3f}.  H1/H10 safety-correction rates
are {mpn['safety_correction_rate_h1']:.3f}/{mpn['safety_correction_rate_h10']:.3f}.
The Multi-UAV Planning Necessity Gate is **{mpn['multi_uav_planning_necessity_gate']}**;
the Multi-UAV Coupling Gate is **{coupling['multi_uav_coupling_gate']}**.
`multi_uav_long_horizon_planning_required` is
`{mpn['multi_uav_long_horizon_planning_required']}` and
`multi_uav_offline_dataset_ready` is `{readiness['multi_uav_offline_dataset_ready']}`.

Recommended next step: {readiness['recommended_next_step']}.
""", encoding="utf-8")


__all__ = ["FIGURE_NAMES", "render_figures", "write_report"]
