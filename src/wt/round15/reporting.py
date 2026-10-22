"""Small, dependency-light visual reporting for the Round-15 audit."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(row: Mapping[str, str], key: str) -> float:
    return float(row[key]) if row.get(key, "") else 0.0


def _save(fig: Any, figures: Path, name: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures / f"{name}.png", dpi=160)
    fig.savefig(figures / f"{name}.svg")
    plt.close(fig)


def _bar(figures: Path, name: str, labels: list[str], values: list[float], *, title: str, ylabel: str, colour: str = "#1677b8") -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(labels, values, color=colour)
    axis.set(title=title, ylabel=ylabel)
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=.25)
    _save(fig, figures, name)


def create_figures(output_dir: Path, figures: Path) -> list[str]:
    """Create the 15 required paired PNG/SVG diagnostic figures."""
    output_dir, figures = Path(output_dir), Path(figures)
    horizon = _read_csv(output_dir / "planning_horizon_value.csv")
    metrics = _read_csv(output_dir / "planning_episode_metrics.csv")
    pair = _read_csv(output_dir / "planning_headroom_metrics.csv")
    runtime = _read_csv(output_dir / "planning_runtime_audit.csv")
    divergence = _read_csv(output_dir / "trajectory_divergence_audit.csv")
    delayed = _read_csv(output_dir / "delayed_benefit_audit.csv")
    gaps = _read_csv(output_dir / "heuristic_planning_gap.csv")
    gate = json.loads((output_dir / "planning_necessity_gate.json").read_text(encoding="utf-8"))
    causal = sorted([row for row in horizon if row["mode"] == "causal"], key=lambda row: int(row["horizon"]))
    oracle = sorted([row for row in horizon if row["mode"] == "oracle"], key=lambda row: int(row["horizon"]))
    labels = [f"H{row['horizon']}" for row in causal]
    _bar(figures, "01_causal_payload_by_horizon", labels, [_num(row, "mean_cumulative_actual_payload_bits") / 1e6 for row in causal], title="Causal actual payload by planning horizon", ylabel="mean payload (Mbit)")
    _bar(figures, "02_oracle_payload_by_horizon", labels, [_num(row, "mean_cumulative_actual_payload_bits") / 1e6 for row in oracle], title="Oracle upper-bound payload by horizon", ylabel="mean payload (Mbit)", colour="#b84a62")
    _bar(figures, "03_causal_residual_by_horizon", labels, [_num(row, "mean_final_residual_queue_bits") / 1e6 for row in causal], title="Causal residual queue by horizon", ylabel="mean residual (Mbit)", colour="#7a6eaa")
    _bar(figures, "04_causal_energy_by_horizon", labels, [_num(row, "mean_cumulative_energy_consumption") for row in causal], title="Causal movement/hover energy by horizon", ylabel="mean energy", colour="#4f9368")
    _bar(figures, "05_heuristic_to_planner_gap", [row["scenario_id"].split("__")[-1] for row in gaps], [_num(row, "causal_h20_minus_best_heuristic_bits") / 1e6 for row in gaps], title="Causal H20 minus best existing heuristic", ylabel="payload gap (Mbit)")
    payload_pairs = [row for row in pair if row["metric"] == "cumulative_actual_payload_bits" and row["group_key"] == "overall"]
    _bar(figures, "06_paired_payload_confidence", [row["comparison"].replace("-minus-causal_H1", "") for row in payload_pairs], [_num(row, "mean_paired_delta") / 1e6 for row in payload_pairs], title="Paired causal payload improvement", ylabel="mean delta (Mbit)")
    _bar(figures, "07_planner_runtime", [f"{row['mode']} H{row['horizon']}" for row in runtime], [_num(row, "mean_runtime_seconds") * 1000 for row in runtime], title="Finite-lattice decision runtime", ylabel="mean milliseconds")
    _bar(figures, "08_planner_nodes", [f"{row['mode']} H{row['horizon']}" for row in runtime], [_num(row, "mean_nodes_expanded") for row in runtime], title="Finite-lattice nodes expanded", ylabel="mean nodes", colour="#a37a3a")
    div20 = [row for row in divergence if row["comparison"] == "causal_H20-vs-causal_H1"]
    _bar(figures, "09_h20_action_divergence", [row["scenario_id"].split("__")[-1] for row in div20], [_num(row, "action_divergence_rate") for row in div20], title="Causal H20 action divergence from H1", ylabel="fraction of steps")
    _bar(figures, "10_h20_visitation_divergence", [row["scenario_id"].split("__")[-1] for row in div20], [_num(row, "visitation_divergence_rate") for row in div20], title="Causal H20 nearest-buoy visitation divergence", ylabel="fraction of steps", colour="#7a6eaa")
    delayed_groups = [row for row in delayed if row["group_key"] in ("overall", "difficulty")]
    _bar(figures, "11_delayed_benefit_rate", [f"{row['group_key']}:{row['group_value']}" for row in delayed_groups], [_num(row, "delayed_benefit_rate") for row in delayed_groups], title="Delayed-benefit-state rate", ylabel="fraction")
    _bar(figures, "12_action_reversal_rate", [f"{row['group_key']}:{row['group_value']}" for row in delayed_groups], [_num(row, "action_reversal_rate") for row in delayed_groups], title="H20 versus H1 action reversal rate", ylabel="fraction", colour="#b84a62")
    scenario_metrics = [row for row in metrics if row["policy_id"] in ("causal_H1", "causal_H20", "oracle_H20")]
    fig, axis = plt.subplots(figsize=(8.0, 4.4))
    for name, colour in (("causal_H1", "#777777"), ("causal_H20", "#1677b8"), ("oracle_H20", "#b84a62")):
        selected = [row for row in scenario_metrics if row["policy_id"] == name]
        axis.plot(range(len(selected)), [_num(row, "cumulative_actual_payload_bits") / 1e6 for row in selected], marker="o", label=name, color=colour)
    axis.set(title="Actual payload across diagnostic scenarios", xlabel="scenario index", ylabel="payload (Mbit)")
    axis.legend(); axis.grid(alpha=.25); _save(fig, figures, "13_payload_scenario_profile")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(labels, [_num(row, "mean_cumulative_actual_payload_bits") / 1e6 for row in causal], marker="o", label="causal")
    axis.plot(labels, [_num(row, "mean_cumulative_actual_payload_bits") / 1e6 for row in oracle], marker="o", label="oracle (future-informed)")
    axis.set(title="Causal versus oracle planning headroom", ylabel="mean payload (Mbit)")
    axis.legend(); axis.grid(alpha=.25); _save(figure, figures, "14_causal_oracle_headroom")
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.axis("off")
    axis.text(.03, .90, f"Planning necessity gate: {gate['gate']}", fontsize=16, weight="bold")
    axis.text(.03, .69, f"Required: {gate['long_horizon_planning_required']}", fontsize=12)
    axis.text(.03, .53, f"Saturation horizon: H{gate['planning_saturation_horizon']}", fontsize=12)
    axis.text(.03, .32, "No artificial PASS threshold; oracle remains non-deployable.", fontsize=11, wrap=True)
    axis.text(.03, .15, "One pre-round metadata-only holdout access is disclosed and excluded from analysis.", fontsize=10, wrap=True)
    _save(fig, figures, "15_gate_and_controls")
    return sorted(path.name for path in figures.glob("*.png"))


__all__ = ["create_figures"]
