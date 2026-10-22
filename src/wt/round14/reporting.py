"""Round-14 artifact text and figure rendering helpers."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def materialize_interface_artifacts(output: Path, selected: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, str]:
    """Write self-contained deployment contracts without any future inputs."""
    output = Path(output)
    schema = {
        "schema_version": "round14_actionable_motion_guidance_v1",
        "purpose": "deterministic current-state Task-Waypoint-to-motion guidance",
        "input": {
            "phi_task": {
                "shape": "[2*max_buoys+1]",
                "layout": "[p_svc[max_buoys], DeltaD_srv[max_buoys], J_H]",
                "provenance": "frozen H=10 Task-Waypoint predictor, evaluation mode only",
            },
            "current_observables": {
                "valid_node_mask": "[max_buoys] boolean",
                "uav_position_m": "[2]",
                "buoy_positions_m": "[max_buoys,2]",
                "rates_bps": "[max_buoys] current allocated rates already exposed by MaritimeEnv",
            },
        },
        "output": {
            "direction_xy": "[2], unit vector or [0,0] deterministic fallback",
            "step_fraction": "[0,1]",
            "movement_action": "unchanged normalized [delta_x, delta_y], radially bounded",
        },
        "direction_variants": {
            "A_service": "masked p_svc weighted unit vectors from UAV to buoy",
            "B_service_progress": "A weights multiplied by positive predicted DeltaD_srv",
            "C_service_progress_rate": "B weights multiplied by normalized current rates_bps",
        },
        "return_role": "J_H sets only sigmoid-bounded magnitude; it never changes direction or resource allocation",
        "fallback": "all-zero movement when direction cancels or no valid nonzero geometry vector remains",
        "forbidden_inputs": [
            "future arrivals", "future queue", "future actions", "future rewards",
            "future Task-Waypoint labels", "old-development-test", "untouched-holdout",
        ],
        "selected_mapping": dict(selected),
        "mapping_gate": str(gate["task_to_motion_mapping_gate"]),
        "service_action_extension": False,
    }
    _write_json(output / "actionable_motion_guidance_schema.json", schema)
    (output / "task_to_motion_mapping_spec.md").write_text(
        "# Round-14 Task-Waypoint-to-Motion Mapping\n\n"
        "The deployable input is the frozen H=10 Task-Waypoint prediction "
        "`Φ_task=[p_svc, ΔD_srv, J_H]` plus current valid-node geometry and current allocated rates. "
        "It produces the existing normalized movement action only; it does not predict a future physical waypoint.\n\n"
        "For each valid buoy `k`, let `u_k` be the current unit vector from UAV to buoy. "
        "A uses `w_k=normalize(p_svc,k)`, B uses "
        "`w_k=normalize(p_svc,k·max(ΔD_srv,k,0))`, and C uses "
        "`w_k=normalize(p_svc,k·max(ΔD_srv,k,0)·normalize(r_k))`. "
        "The direction is `normalize(sum_k w_k u_k)`.  A cancelled vector deterministically becomes zero movement.\n\n"
        "`J_H` has no directional or service-allocation role.  Its only candidate role is the fixed, "
        "current-only magnitude `sigmoid(J_H)`.  The resulting action is radial-clipped to the unchanged "
        "two-dimensional `[delta_x, delta_y]` interface.\n\n"
        "Any future outcomes in this round are offline diagnostics and counterfactual evaluation labels, not mapping inputs.\n",
        encoding="utf-8",
    )
    (output / "future_policy_mapping_interface.md").write_text(
        "# Future Policy Mapping Interface\n\n"
        "A future policy consumer may call the selected deterministic mapping with only the current frozen "
        "Task-Waypoint prediction, `valid_node_mask`, current UAV and buoy positions, and current `rates_bps`. "
        "It receives a bounded two-dimensional movement intent.  It must not supply future arrivals, queues, "
        "actions, rewards, labels, or service decisions.  The environment action and fixed 1/K allocation semantics "
        "remain unchanged.\n\n"
        f"Round-14 selected `{selected['selected_direction_mapping']}` as a diagnostic direction mapping; "
        f"the readiness gate is `{gate['task_to_motion_mapping_gate']}`.  This file is an interface contract, "
        "not permission to create a later round or train a policy.\n",
        encoding="utf-8",
    )
    return {
        "mapping_spec": "task_to_motion_mapping_spec.md",
        "schema": "actionable_motion_guidance_schema.json",
        "future_interface": "future_policy_mapping_interface.md",
    }


def readiness_gate(gate: Mapping[str, Any], selected: Mapping[str, Any], alignment_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    validation = [row for row in alignment_rows if row["split"] == "validation" and row["group_dimension"] == "overall"]
    best = next(row for row in validation if row["mapping_name"] == selected["selected_direction_mapping"])
    ready = gate["task_to_motion_mapping_gate"] == "MAP-A" and float(best["mean_teacher_alignment_cosine"]) > 0.0
    return {
        "schema_version": "round14_readiness_gate_v1",
        "task_to_motion_mapping_gate": gate["task_to_motion_mapping_gate"],
        "selected_direction_mapping": selected["selected_direction_mapping"],
        "validation_teacher_alignment_cosine": float(best["mean_teacher_alignment_cosine"]),
        "counterfactual_payload_ci": [
            float(gate["payload"]["bootstrap_ci_low"]), float(gate["payload"]["bootstrap_ci_high"]),
        ],
        "future_policy_mapping_ready": ready,
        "decision": "actionable mapping evidence is sufficient" if ready else "diagnostic-only; do not begin a future policy round",
        "training_performed": False,
        "neural_policy_training_performed": False,
        "service_action_extension_performed": False,
        "old_development_test_accessed": False,
        "untouched_holdout_accessed": False,
        "round15_created": False,
    }


def _save(figure: Any, destination: Path) -> None:
    figure.patch.set_facecolor("white")
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=150, facecolor="white")
    figure.savefig(destination.with_suffix(".svg"), facecolor="white")
    plt.close(figure)


def _bar_figure(destination: Path, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str, *, colors: Sequence[str] | None = None) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.2), facecolor="white")
    x = np.arange(len(labels))
    bars = axis.bar(x, values, color=colors or "#3478bf")
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3g}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    _save(figure, destination)


def render_figures(output: Path, figure_dir: Path) -> list[str]:
    """Produce the required fifteen white-background PNG/SVG figure pairs."""
    output, figure_dir = Path(output), Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=False)
    alignment = _rows(output / "mapping_quality_stratified_alignment.csv")
    future = _rows(output / "mapping_future_relevance.csv")
    counter = _rows(output / "mapping_counterfactual_metrics.csv")
    paired = _rows(output / "mapping_paired_comparison.csv")
    magnitude = _rows(output / "return_magnitude_modulation_audit.csv")
    roles = _rows(output / "task_waypoint_head_role_audit.csv")
    selected = json.loads((output / "selected_task_to_motion_mapping.json").read_text(encoding="utf-8"))
    gate = json.loads((output / "task_to_motion_mapping_gate.json").read_text(encoding="utf-8"))

    overall = [row for row in alignment if row["split"] == "validation" and row["group_dimension"] == "overall"]
    _bar_figure(figure_dir / "01_teacher_alignment", [r["mapping_name"] for r in overall], [float(r["mean_teacher_alignment_cosine"]) for r in overall], "Validation teacher-action alignment", "mean cosine")
    by_quality = [row for row in alignment if row["split"] == "validation" and row["group_dimension"] == "quality_tier"]
    _bar_figure(figure_dir / "02_quality_stratified_alignment", [f"{r['mapping_name']}:{r['group_value']}" for r in by_quality], [float(r["mean_teacher_alignment_cosine"]) for r in by_quality], "Quality-stratified alignment", "mean cosine")
    by_difficulty = [row for row in alignment if row["split"] == "validation" and row["group_dimension"] == "difficulty_tier"]
    _bar_figure(figure_dir / "03_difficulty_stratified_alignment", [f"{r['mapping_name']}:{r['group_value']}" for r in by_difficulty], [float(r["mean_teacher_alignment_cosine"]) for r in by_difficulty], "Difficulty-stratified alignment", "mean cosine")
    direction_future = [row for row in future if row["split"] == "validation" and row["diagnostic"] == "predicted_direction_vs_oracle_task_waypoint_direction"]
    _bar_figure(figure_dir / "04_future_direction_relevance", [r["mapping_name"] for r in direction_future], [float(r["mean_direction_cosine"]) for r in direction_future], "Offline predicted-vs-oracle direction diagnostic", "mean cosine")
    return_future = [row for row in future if row["split"] == "validation" and row["mapping_name"] == "return_head"]
    _bar_figure(figure_dir / "05_future_return_relevance", ["Pearson", "Spearman"], [float(return_future[0]["pearson_r"]), float(return_future[0]["spearman_r"])], "Offline return-head relevance", "correlation")

    selected_rows = [row for row in counter if row["mapping_name"] == selected["selected_direction_mapping"] and row["magnitude_mode"] == "full_step"]
    for number, field, title, ylabel in (
        (6, "total_service_bits", "Same-snapshot payload by first action", "H-step payload (bits)"),
        (7, "final_completion", "Same-snapshot completion by first action", "final completion"),
        (8, "final_residual_queue_bits", "Same-snapshot residual queue by first action", "residual queue (bits)"),
        (9, "H_step_return", "Same-snapshot return by first action", "H-step return"),
    ):
        labels = sorted({str(row["action_name"]) for row in selected_rows})
        values = [float(np.mean([float(row[field]) for row in selected_rows if row["action_name"] == label])) for label in labels]
        _bar_figure(figure_dir / f"{number:02d}_{field}", labels, values, title, ylabel)

    overall_paired = [row for row in paired if row["mapping_name"] == selected["selected_direction_mapping"] and row["group_dimension"] == "overall" and row["comparison"] == "guidance_minus_queue_rate_energy"]
    for number, metric, title in (
        (10, "Delta_payload_bits", "Paired payload delta versus queue-rate-energy"),
        (11, "Delta_completion", "Paired completion delta versus queue-rate-energy"),
        (12, "Delta_residual_queue_bits", "Paired residual-queue improvement versus queue-rate-energy"),
        (13, "Delta_return", "Paired return delta versus queue-rate-energy"),
    ):
        row = next(item for item in overall_paired if item["metric"] == metric)
        _bar_figure(figure_dir / f"{number:02d}_{metric}", ["mean", "CI low", "CI high"], [float(row["paired_mean"]), float(row["bootstrap_ci_low"]), float(row["bootstrap_ci_high"])], title, metric)

    payload_magnitude = [row for row in magnitude if row["metric"] == "Delta_payload_bits"]
    _bar_figure(figure_dir / "14_return_magnitude_audit", [row["comparison"] for row in payload_magnitude], [float(row["paired_mean"]) for row in payload_magnitude], "Magnitude variants versus full step", "paired payload delta (bits)")
    role_service = [row for row in roles if row["head_role"] == "service_progress_direction"]
    labels = list(dict.fromkeys(row["mapping_name"] for row in role_service))
    role_values = [float(np.mean([float(row["payload_vs_qre_bits"]) for row in role_service if row["mapping_name"] == label])) for label in labels]
    _bar_figure(figure_dir / "15_head_role_summary", labels, role_values, f"Head-role summary; mapping gate {gate['task_to_motion_mapping_gate']}", "mean payload delta versus QRE (bits)")
    return [path.name for path in sorted(figure_dir.glob("*.png"))]


__all__ = ["materialize_interface_artifacts", "readiness_gate", "render_figures"]
