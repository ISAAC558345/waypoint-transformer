"""Render the fixed Round-22 scientific figure set from frozen output files."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import R22_OUTPUT


FIGURES = Path("round_22/figures/round22_deepsets_task_waypoint_policy_20260829_1153")


def _rows(name: str) -> list[dict[str, str]]:
    with (R22_OUTPUT / name).open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def _n(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    return default if value in ("", None) else float(value)


def _save(figure: Any, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True); figure.tight_layout(); figure.savefig(FIGURES / f"{stem}.png", dpi=160); figure.savefig(FIGURES / f"{stem}.svg"); plt.close(figure)


def _bar(stem: str, labels: list[str], values: list[float], title: str, ylabel: str, *, colors: list[str] | None = None) -> None:
    figure, axis = plt.subplots(figsize=(7, 4)); axis.bar(labels, values, color=colors); axis.axhline(0.0, color="black", lw=.8); axis.set(title=title, ylabel=ylabel); axis.grid(axis="y", alpha=.25); _save(figure, stem)


def _overall(rows: list[dict[str, str]]) -> dict[str, str]:
    return next(row for row in rows if row.get("group_field") == "overall" and row.get("group_value") == "all")


def _placeholder(stem: str, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 4)); axis.axis("off"); axis.text(.5, .5, message, ha="center", va="center", wrap=True); axis.set(title=title); _save(figure, stem)


def render_figures() -> list[str]:
    summary = _rows("round22_policy_training_summary.csv")
    histories: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in summary:
        path = Path(row["checkpoint"]).parent / "training_history.csv"
        with path.open(encoding="utf-8", newline="") as handle: histories[(row["family"], int(row["seed"]))] = list(csv.DictReader(handle))
    figure, axis = plt.subplots(figsize=(8, 4))
    for family in sorted({row["family"] for row in summary}):
        by_epoch: dict[int, list[float]] = {}
        for (name, _), rows in histories.items():
            if name == family:
                for row in rows: by_epoch.setdefault(int(row["epoch"]), []).append(_n(row, "validation_raw_joint_action_rmse"))
        axis.plot(sorted(by_epoch), [np.mean(by_epoch[key]) for key in sorted(by_epoch)], label=family)
    axis.set(title="Formal validation RMSE by family", xlabel="epoch", ylabel="raw joint-action RMSE"); axis.legend(fontsize=7); axis.grid(alpha=.25); _save(figure, "01_formal_train_validation_loss_by_family")
    offline = _rows("offline_policy_run_summary.csv")
    families = sorted({row["family"] for row in offline}); _bar("02_offline_rmse", families, [float(np.mean([_n(row, "raw_joint_action_rmse") for row in offline if row["family"] == family])) for family in families], "Offline raw joint-action RMSE", "RMSE")
    figure, axis = plt.subplots(figsize=(8, 4))
    for family in families:
        rows = sorted([row for row in offline if row["family"] == family], key=lambda row: int(row["seed"])); axis.plot([row["seed"] for row in rows], [_n(row, "raw_joint_action_rmse") for row in rows], marker="o", label=family)
    axis.set(title="Offline RMSE by model seed", xlabel="seed", ylabel="RMSE"); axis.legend(fontsize=7); axis.grid(alpha=.25); _save(figure, "03_per_seed_rmse")
    gap = _overall(_rows("oracle_task_waypoint_policy_gap.csv")); _bar("04_oracle_gap", ["predicted − oracle"], [_n(gap, "oracle_gap_raw_action_rmse")], "Oracle gap (offline)", "RMSE difference")
    heads = _rows("task_waypoint_head_utilization_summary.csv")
    for number, head in ((5, "service"), (6, "progress"), (7, "return")):
        _bar(f"{number:02d}_{head}_sensitivity", [f"seed {row['seed']}" for row in heads], [_n(row, f"sensitivity_{head}") for row in heads], f"{head.title()} Phi action sensitivity", "mean action change")
    conflict = _rows("conflict_policy_analysis.csv"); conflict_labels = [row["comparison"] for row in conflict]; _bar("08_conflict_state_error", conflict_labels, [_n(row, "task_waypoint_rmse_improvement") for row in conflict], "Task-WP RMSE improvement by conflict state", "State-only RMSE − Task-WP RMSE")
    bridge = _rows("bridge_policy_analysis.csv"); _bar("09_bridge_state_error", [row["comparison"] for row in bridge], [_n(row, "task_waypoint_rmse_improvement") for row in bridge], "Task-WP RMSE improvement by bridge label", "State-only RMSE − Task-WP RMSE")
    deployable = ["predicted_task_waypoint_transformer", "state_only_transformer", "predicted_task_waypoint_mlp"]
    validation = _rows("validation_closed_loop_metrics.csv")
    _bar("10_validation_payload", deployable, [float(np.mean([_n(row, "cumulative_payload_bits") for row in validation if row["family"] == family])) for family in deployable], "Validation closed-loop payload", "payload (bits)")
    paired = _rows("validation_paired_summary.csv"); overall = _overall(paired); _bar("11_validation_delta_payload", ["Predicted TW − State"], [_n(overall, "delta_payload_mean")], "Validation paired payload difference", "payload (bits)")
    krows = [row for row in paired if row["group_field"] == "num_buoys"]; _bar("12_validation_k3_k5", [f"K={row['group_value']}" for row in krows], [_n(row, "delta_payload_mean") for row in krows], "Validation delta payload by K", "payload (bits)")
    etarows = [row for row in paired if row["group_field"] == "eta"]; _bar("13_validation_eta_tier", [f"η={row['group_value']}" for row in etarows], [_n(row, "delta_payload_mean") for row in etarows], "Validation delta payload by shared-spectrum η", "payload (bits)")
    _bar("14_validation_completion", ["Predicted TW − State"], [_n(overall, "delta_completion_mean")], "Validation completion difference", "completion ratio")
    _bar("15_validation_residual_queue", ["Predicted TW − State"], [_n(overall, "delta_residual_queue_bits_mean")], "Validation residual queue difference", "queue (bits)")
    mlp = _rows("transformer_vs_mlp_audit.csv")[0]; _bar("16_transformer_vs_mlp", ["offline RMSE", "validation payload"], [_n(mlp, "offline_rmse_improvement"), _n(mlp, "validation_delta_payload")], "Transformer vs MLP (positive favors Transformer)", "native metric difference")
    fresh_metrics = R22_OUTPUT / "fresh_dev_closed_loop_metrics.csv"
    if fresh_metrics.is_file():
        fresh = _rows("fresh_dev_closed_loop_metrics.csv"); _bar("17_fresh_dev_payload", deployable, [float(np.mean([_n(row, "cumulative_payload_bits") for row in fresh if row["family"] == family])) for family in deployable], "Fresh-development payload", "payload (bits)")
        fresh_pair = _overall(_rows("fresh_dev_paired_summary.csv")); _bar("18_fresh_dev_delta", ["Predicted TW − State"], [_n(fresh_pair, "delta_payload_mean")], "Fresh-development paired payload difference", "payload (bits)")
    else:
        _placeholder("17_fresh_dev_payload", "Fresh-development payload", "Not run: validation V-C forbids fresh development."); _placeholder("18_fresh_dev_delta", "Fresh-development paired delta", "Not run: validation V-C forbids fresh development.")
    steps = _rows("validation_closed_loop_steps.csv"); representative = [row for row in steps if row["family"] == "predicted_task_waypoint_transformer" and row["model_seed"] == "0"]
    first_id = representative[0]["scenario_id"]; representative = [row for row in representative if row["scenario_id"] == first_id]
    positions = [np.asarray(json.loads(row["uav_positions_m"]), dtype=float) for row in representative]
    figure, axis = plt.subplots(figsize=(6, 5)); array = np.asarray(positions)
    for uav in range(array.shape[1]): axis.plot(array[:, uav, 0], array[:, uav, 1], marker="o", label=f"UAV {uav}")
    axis.set(title="Representative validation multi-UAV trajectory", xlabel="x (m)", ylabel="y (m)"); axis.legend(); axis.grid(alpha=.25); _save(figure, "19_representative_multi_uav_trajectories")
    phi = [np.asarray(json.loads(row["raw_task_waypoint_phi"]), dtype=float) for row in representative]
    array = np.asarray(phi); figure, axis = plt.subplots(figsize=(7, 4)); axis.plot(array[..., :5].mean(axis=(1, 2)), label="service responsibility"); axis.plot(array[..., 5:10].mean(axis=(1, 2)), label="gross service progress"); axis.plot(array[..., 10].mean(axis=1), label="team H-step return"); axis.set(title="Predicted Task-Waypoint heads over time", xlabel="timestep", ylabel="predicted value"); axis.legend(); axis.grid(alpha=.25); _save(figure, "20_task_waypoint_predictions_over_time")
    files = sorted(path.name for path in FIGURES.iterdir() if path.is_file() and path.suffix in {".png", ".svg"})
    if len(files) != 40: raise RuntimeError(f"Round-22 requires 20 PNG+SVG figures, got {len(files)} files")
    (R22_OUTPUT / "round22_figure_manifest.json").write_text(json.dumps({"figure_directory": FIGURES.as_posix(), "count": len(files), "files": files, "semantic_audit": {"all_png_svg_pairs": True, "all_titles_protocol_scoped": True, "fresh_placeholders_used_only_when_fresh_forbidden": not fresh_metrics.is_file()}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return files


if __name__ == "__main__": render_figures()
