"""Debug figures for one completed episode. They are not used for tuning."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_figures(scenario_id: int, steps: list[dict], uav_rows: list[dict], decisions: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / f"scenario_{int(scenario_id)}"
    uavs = sorted({int(row["uav"]) for row in uav_rows})
    figure, axis = plt.subplots(figsize=(6, 4))
    for uav in uavs:
        rows = [row for row in uav_rows if int(row["uav"]) == uav]
        axis.plot([row["position_x"] for row in rows], [row["position_y"] for row in rows], label=f"uav {uav}")
    axis.set(title="UAV trajectories", xlabel="x", ylabel="y")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(prefix.with_name(prefix.name + "_trajectories.png"))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 3))
    for uav in uavs:
        rows = [row for row in uav_rows if int(row["uav"]) == uav]
        axis.step([row["episode_step"] for row in rows], [row["target"] for row in rows], where="post", label=f"uav {uav}")
    axis.set(title="UAV target vs time", xlabel="step", ylabel="target")
    axis.legend(fontsize=7, ncol=3)
    figure.tight_layout()
    figure.savefig(prefix.with_name(prefix.name + "_targets.png"))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 3))
    axis.plot([row["episode_step"] for row in steps], [row["cumulative_payload_bits"] for row in steps])
    axis.set(title="Cumulative payload vs time", xlabel="step", ylabel="bits")
    figure.tight_layout()
    figure.savefig(prefix.with_name(prefix.name + "_payload.png"))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 3))
    axis.plot([row["episode_step"] for row in steps], [row["total_q_bits"] for row in steps])
    axis.set(title="Total queue vs time", xlabel="step", ylabel="bits")
    figure.tight_layout()
    figure.savefig(prefix.with_name(prefix.name + "_queue.png"))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 3))
    axis.plot([row["episode_step"] for row in steps], [row["served_bits"] for row in steps])
    axis.set(title="Served bits per step", xlabel="step", ylabel="bits")
    figure.tight_layout()
    figure.savefig(prefix.with_name(prefix.name + "_served.png"))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 3))
    axis.plot([row["decision_id"] for row in decisions], [row["selected_j15"] for row in decisions], label="selected J15")
    axis.plot([row["decision_id"] for row in decisions], [row["j15_margin"] for row in decisions], label="J15 margin")
    axis.set(title="Selected J15 and margin", xlabel="decision", ylabel="bits")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(prefix.with_name(prefix.name + "_j15.png"))
    plt.close(figure)
