"""Fixed, post-fresh Round-30 scientific figure set (PNG and SVG)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import read_csv
from .runner import OUTPUT, write_json


ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "round_30" / "figures" / "round30_deepsets_ppo_20260830_1845"


def number(row: dict[str, Any], key: str) -> float: return float(row[key])


def save(name: str, title: str, xlabel: str, ylabel: str, draw) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    draw(axis); axis.set_title(title); axis.set_xlabel(xlabel); axis.set_ylabel(ylabel); axis.grid(alpha=.22)
    for suffix in ("png", "svg"): figure.savefig(FIGURES / f"{name}.{suffix}", dpi=180 if suffix == "png" else None)
    plt.close(figure)


def lines(axis, curves: list[dict[str, Any]], field: str, *, smooth: int | None = None) -> None:
    for seed in range(5):
        rows = sorted((row for row in curves if int(row["seed"]) == seed), key=lambda row: int(row["environment_steps"]))
        x, y = np.asarray([number(row, "environment_steps") for row in rows]), np.asarray([number(row, field) for row in rows])
        valid = np.isfinite(y); x, y = x[valid], y[valid]
        if smooth and len(y) >= smooth:
            kernel = np.ones(smooth) / smooth; y = np.convolve(y, kernel, mode="same")
        axis.plot(x, y, alpha=.85, linewidth=2 if smooth else .8, label=f"seed {seed}")
    axis.legend(ncol=3, fontsize=8)


def delta_bars(axis, deltas: list[dict[str, Any]], comparators: Iterable[str], title_suffix: str = "") -> None:
    values, labels = [], []
    for comparator in comparators:
        rows = [row for row in deltas if row["comparator"] == comparator]
        values.append(np.mean([number(row, "delta_cumulative_payload_bits") for row in rows])); labels.append(comparator.replace("state_only_bc_transformer", "BC").replace("independent_myopic", "Independent").replace("random_valid", "Random"))
    axis.bar(labels, values, color="#4477AA"); axis.axhline(0., color="black", linewidth=.8); axis.tick_params(axis="x", rotation=18)
    if title_suffix: axis.text(.02, .96, title_suffix, transform=axis.transAxes, va="top", fontsize=8)


def generate_figures() -> list[str]:
    """Generate exactly twenty preregistered visualizations after the single fresh run."""
    if not (OUTPUT / "ppo_fresh_metrics.csv").is_file(): raise RuntimeError("Round-30 figures require completed frozen fresh evaluation")
    FIGURES.mkdir(parents=True, exist_ok=True)
    curves, deltas, stability = read_csv(OUTPUT / "ppo_learning_curves.csv"), read_csv(OUTPUT / "ppo_baseline_comparison.csv"), read_csv(OUTPUT / "ppo_seed_stability.csv")
    validation, fresh, robust = read_csv(OUTPUT / "ppo_validation_metrics.csv"), read_csv(OUTPUT / "ppo_fresh_metrics.csv"), read_csv(OUTPUT / "ppo_K_eta_robustness.csv")
    save("01_raw_training_return", "PPO raw episodic return", "environment steps", "completed episodic return", lambda axis: lines(axis, curves, "mean_completed_return"))
    save("02_smoothed_training_return", "PPO smoothed episodic return (fixed 5-update mean)", "environment steps", "completed episodic return", lambda axis: lines(axis, curves, "mean_completed_return", smooth=5))
    save("03_payload_learning_curve", "PPO actual payload learning curve", "environment steps", "completed payload bits", lambda axis: lines(axis, curves, "mean_completed_payload_bits", smooth=5))
    save("04_completion_learning_curve", "PPO completion learning curve", "environment steps", "completion", lambda axis: lines(axis, curves, "mean_completed_completion", smooth=5))
    save("05_residual_queue_learning_curve", "PPO residual queue", "environment steps", "residual queue bits", lambda axis: lines(axis, curves, "mean_completed_residual_queue_bits", smooth=5))
    save("06_entropy", "PPO entropy", "environment steps", "entropy", lambda axis: lines(axis, curves, "entropy"))
    save("07_approx_kl", "PPO approximate KL", "environment steps", "KL", lambda axis: lines(axis, curves, "approx_kl"))
    save("08_clip_fraction", "PPO clip fraction", "environment steps", "clip fraction", lambda axis: lines(axis, curves, "clip_fraction"))
    save("09_explained_variance", "PPO critic explained variance", "environment steps", "explained variance", lambda axis: lines(axis, curves, "explained_variance"))
    save("10_action_std", "PPO action standard deviation", "environment steps", "action std", lambda axis: lines(axis, curves, "action_std"))
    save("11_projection_rate", "PPO training safety projection rate", "environment steps", "projection rate", lambda axis: lines(axis, curves, "mean_completed_projection_rate", smooth=5))
    def validation_bar(axis):
        means = [np.mean([number(row, "cumulative_payload_bits") for row in validation if int(row["model_seed"]) == seed]) for seed in range(5)]
        axis.bar([f"seed {seed}" for seed in range(5)], means, color="#44AA99")
    save("12_final_validation_by_seed", "Final PPO validation payload", "seed", "payload bits", validation_bar)
    save("13_ppo_vs_random", "PPO − Random Valid", "comparator", "mean paired payload delta (bits)", lambda axis: delta_bars(axis, deltas, ("random_valid",)))
    save("14_ppo_vs_independent", "PPO − Independent Myopic", "comparator", "mean paired payload delta (bits)", lambda axis: delta_bars(axis, deltas, ("independent_myopic",)))
    save("15_ppo_vs_bc", "PPO − frozen State-only BC", "comparator", "mean paired payload delta (bits)", lambda axis: delta_bars(axis, deltas, ("state_only_bc_transformer",)))
    save("16_ppo_vs_planners", "PPO − joint planners", "planner", "mean paired payload delta (bits)", lambda axis: delta_bars(axis, deltas, ("joint_h1", "joint_h5", "joint_h10")))
    def k_plot(axis):
        rows = [row for row in robust if row["population"] == "fresh" and row["axis"] == "num_buoys"]
        axis.bar([row["group"] for row in rows], [number(row, "mean_delta_payload_bits") for row in rows], color="#CC6677"); axis.axhline(0., color="black", linewidth=.8)
    save("17_k_robustness", "Fresh PPO − Independent by K", "K", "mean paired payload delta (bits)", k_plot)
    def eta_plot(axis):
        rows = [row for row in robust if row["population"] == "fresh" and row["axis"] == "eta"]
        axis.bar([row["group"] for row in rows], [number(row, "mean_delta_payload_bits") for row in rows], color="#AA4499"); axis.axhline(0., color="black", linewidth=.8)
    save("18_eta_robustness", "Fresh PPO − Independent by eta", "eta", "mean paired payload delta (bits)", eta_plot)
    def fresh_seed(axis):
        rows = [row for row in deltas if row["population"] == "fresh" and row["comparator"] == "independent_myopic"]
        values = [np.mean([number(row, "delta_cumulative_payload_bits") for row in rows if int(row["model_seed"]) == seed]) for seed in range(5)]
        axis.bar([f"seed {seed}" for seed in range(5)], values, color="#DDCC77"); axis.axhline(0., color="black", linewidth=.8)
    save("19_fresh_seed_deltas", "Fresh PPO − Independent seed deltas", "seed", "mean paired payload delta (bits)", fresh_seed)
    def trajectory(axis):
        rows = read_csv(OUTPUT / "ppo_fresh_trajectory_steps.csv")
        chosen = [row for row in rows if row["policy_id"] == "deepsets_ppo" and int(row["model_seed"]) == 0]
        scenario = chosen[0]["scenario_id"]; chosen = [row for row in chosen if row["scenario_id"] == scenario]
        positions = [np.asarray(json.loads(row["uav_positions_m"]), dtype=float) for row in chosen]
        for uav in range(2): axis.plot([row[uav, 0] for row in positions], [row[uav, 1] for row in positions], marker="o", label=f"UAV {uav}")
        axis.legend(); axis.set_aspect("equal", adjustable="box")
    save("20_representative_fresh_trajectory", "Representative deterministic fresh PPO trajectory", "x (m)", "y (m)", trajectory)
    names = sorted(path.name for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"})
    if len(names) != 40: raise RuntimeError("Round-30 requires twenty PNG/SVG figure pairs")
    write_json(OUTPUT / "round30_figure_manifest.json", {"round": "round_30", "figure_pairs": 20, "png_svg_files": names, "smoothing": "fixed centered 5-update arithmetic mean; raw curves retained", "fresh_outcomes_reused_only_from_single_saved_access": True})
    return names


__all__ = ["FIGURES", "generate_figures"]
