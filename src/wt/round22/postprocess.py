"""Frozen-result gates and mechanism summaries for Round-22."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .data import R22_OUTPUT


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def _number(row: Mapping[str, str], field: str) -> float:
    return float(row[field])


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields = sorted({key for row in values for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)


def _overall(path: Path) -> dict[str, str]:
    rows = _rows(path)
    return next(row for row in rows if row.get("group_field") == "overall" and row.get("group_value") == "all")


def _temporal_audit(fresh_accessed: bool) -> dict[str, Any]:
    offline = _overall(R22_OUTPUT / "transformer_vs_mlp_offline.csv")
    validation = _rows(R22_OUTPUT / "validation_transformer_vs_mlp_comparison.csv")
    validation_payload = float(np.mean([_number(row, "delta_cumulative_payload_bits") for row in validation]))
    fresh_payload = None
    if fresh_accessed:
        fresh = _rows(R22_OUTPUT / "fresh_dev_transformer_vs_mlp_comparison.csv")
        fresh_payload = float(np.mean([_number(row, "delta_cumulative_payload_bits") for row in fresh]))
    offline_advantage = _number(offline, "transformer_rmse_improvement_over_mlp")
    if offline_advantage > 0.0 and validation_payload > 0.0 and (fresh_payload is None or fresh_payload > 0.0): value = True
    elif offline_advantage <= 0.0 or validation_payload <= 0.0 or (fresh_payload is not None and fresh_payload <= 0.0): value = False
    else: value = "uncertain"
    row = {"comparison": "predicted_task_waypoint_transformer_minus_mlp", "offline_rmse_improvement": offline_advantage, "validation_delta_payload": validation_payload, "fresh_delta_payload": fresh_payload, "temporal_transformer_value": value, "protocol": "fixed checkpoints; no retraining"}
    _csv(R22_OUTPUT / "transformer_vs_mlp_audit.csv", [row])
    return row


def _utilization() -> dict[str, Any]:
    rows = _rows(R22_OUTPUT / "task_waypoint_head_utilization_summary.csv")
    head_means = {head: float(np.mean([_number(row, f"sensitivity_{head}") for row in rows])) for head in ("service", "progress", "return")}
    return {"sensitivity_service": head_means["service"], "sensitivity_progress": head_means["progress"], "sensitivity_return": head_means["return"], "dominant_policy_head": max(head_means, key=head_means.get), "normal_vs_zero_shuffle_has_nonzero_action_change": bool(max(head_means.values()) > 1.0e-12)}


def _fresh_gate(validation_gate: Mapping[str, Any], utilization: Mapping[str, Any]) -> dict[str, Any]:
    if validation_gate["gate"] == "V-C":
        return {"gate": "MU-WP-C", "reason": "validation_V-C_forbids_fresh_development", "multi_uav_task_waypoint_policy_advantage": False, "candidate_ready_for_untouched_eval": False, "fresh_dev_access_count": 0}
    overall = _overall(R22_OUTPUT / "fresh_dev_paired_summary.csv")
    rows = _rows(R22_OUTPUT / "fresh_dev_paired_summary.csv")
    per_seed = [row for row in rows if row["group_field"] == "model_seed"]
    conflict = next((row for row in rows if row["group_field"] == "interference_conflict" and row["group_value"] == "True"), None)
    positive = sum(_number(row, "delta_payload_mean") > 0.0 for row in per_seed)
    completion_ok = _number(overall, "delta_completion_mean") >= -1.0e-12
    residual_ok = _number(overall, "delta_residual_queue_bits_mean") <= 1.0e-12
    safety_ok = _number(overall, "delta_safety_correction_rate_mean") <= 1.0e-12
    subset_ok = conflict is not None and conflict.get("available") == "True" and _number(conflict, "delta_payload_mean") >= _number(overall, "delta_payload_mean")
    conditions = {"positive_mean_payload": _number(overall, "delta_payload_mean") > 0.0, "positive_ci_lower": _number(overall, "delta_payload_ci95_lower") > 0.0, "at_least_two_positive_seeds": positive >= 2, "completion_not_degraded": completion_ok, "residual_queue_not_degraded": residual_ok, "safety_not_degraded": safety_ok, "task_waypoint_utilized": bool(utilization["normal_vs_zero_shuffle_has_nonzero_action_change"]), "conflict_or_bridge_not_weaker_than_overall": subset_ok}
    stable_negative = _number(overall, "delta_payload_mean") < 0.0 and _number(overall, "delta_payload_ci95_upper") < 0.0 and sum(_number(row, "delta_payload_mean") < 0.0 for row in per_seed) >= 2
    if all(conditions.values()): gate = "MU-WP-A"
    elif stable_negative: gate = "MU-WP-C"
    else: gate = "MU-WP-B"
    return {"gate": gate, "fresh_dev_access_count": 1, "overall_delta_payload_mean": _number(overall, "delta_payload_mean"), "overall_delta_payload_ci95": [_number(overall, "delta_payload_ci95_lower"), _number(overall, "delta_payload_ci95_upper")], "positive_seed_count": positive, "conditions": conditions, "multi_uav_task_waypoint_policy_advantage": gate == "MU-WP-A", "candidate_ready_for_untouched_eval": gate == "MU-WP-A", "bootstrap_unit": "scenario", "fresh_bridge_status": "not_available_without_prohibited_future_labels"}


def run_postprocess(validation_gate: Mapping[str, Any], fresh_accessed: bool) -> dict[str, Any]:
    utilization = _utilization(); temporal = _temporal_audit(fresh_accessed); mu_gate = _fresh_gate(validation_gate, utilization)
    oracle = _overall(R22_OUTPUT / "oracle_task_waypoint_policy_gap.csv")
    validation = _overall(R22_OUTPUT / "validation_paired_summary.csv")
    labels: list[str] = []
    if _number(oracle, "oracle_gap_raw_action_rmse") > 0.0: labels.append("P-MU1")
    if mu_gate["multi_uav_task_waypoint_policy_advantage"]: labels.append("P-MU2")
    if not mu_gate["multi_uav_task_waypoint_policy_advantage"] and abs(_number(oracle, "oracle_gap_raw_action_rmse")) <= 1.0e-12: labels.append("P-MU3")
    conflict = next((row for row in _rows(R22_OUTPUT / "validation_paired_summary.csv") if row["group_field"] == "interference_conflict" and row["group_value"] == "True"), None)
    if conflict is not None and _number(conflict, "delta_payload_mean") > _number(validation, "delta_payload_mean") and not mu_gate["multi_uav_task_waypoint_policy_advantage"]: labels.append("P-MU4")
    classification = {"mechanism_classes": labels or ["inconclusive"], "oracle_gap_raw_action_rmse": _number(oracle, "oracle_gap_raw_action_rmse"), "validation_delta_payload": _number(validation, "delta_payload_mean"), "mu_wp_gate": mu_gate["gate"], "temporal_transformer_value": temporal["temporal_transformer_value"], "interpretation": "descriptive frozen-checkpoint classification; no retraining or selection"}
    _json(R22_OUTPUT / "multi_uav_task_waypoint_policy_gate.json", mu_gate)
    _json(R22_OUTPUT / "round22_mechanism_classification.json", classification)
    _json(R22_OUTPUT / "round22_postprocess_summary.json", {"validation_gate": validation_gate, "mu_wp_gate": mu_gate, "utilization": utilization, "temporal": temporal, "classification": classification})
    return {"utilization": utilization, "temporal": temporal, "mu_gate": mu_gate, "classification": classification}


__all__ = ["run_postprocess"]
