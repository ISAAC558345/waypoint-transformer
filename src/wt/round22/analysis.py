"""Read-only Round-22 formal-checkpoint offline analyses.

The functions here are deliberately validation-only.  They never construct a
rollout environment and never decode the Round-20 development-test split.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .data import FrozenDeepSetsPredictor, R22_OUTPUT, VALIDATION, load_policy_sequences
from .evaluation import head_utilization_baselines, offline_metrics
from .models import ORACLE_TRANSFORMER, PREDICTED_MLP, PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER
from .training import TrainingConfig, load_checkpoint, predict, raw_rmse


ETA = {"orthogonal": 0.0, "low": 0.25, "medium": 0.5, "full": 1.0}
SCALAR_METRICS = ("normalized_mse", "raw_action_rmse", "mae", "cosine", "angular_error_deg", "magnitude_error", "projection_rate", "per_uav_0_rmse", "per_uav_1_rmse")


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise RuntimeError(f"required Round-22 analysis is empty: {path.name}")
    fields = sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(values)


def _formal_runs() -> list[dict[str, Any]]:
    manifest = json.loads((R22_OUTPUT / "round22_formal_training_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("formal_training_completed") != "12/12" or len(manifest.get("runs", [])) != 12:
        raise RuntimeError("offline analysis requires the completed 12/12 formal training matrix")
    return list(manifest["runs"])


def _with_eta(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        copy = dict(row); copy["eta"] = ETA[str(copy["interference_tier"])]
        result.append(copy)
    return result


def _aggregate(rows: list[dict[str, Any]], *, family: str, seed: int, field: str, value: str) -> dict[str, Any]:
    return {"family": family, "seed": seed, "split": VALIDATION, "group_field": field, "group_value": value, "sample_count": len(rows), **{metric: float(np.mean([float(row[metric]) for row in rows])) for metric in SCALAR_METRICS}}


def _grouped(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for family in sorted({str(row["family"]) for row in rows}):
        for seed in sorted({int(row["seed"]) for row in rows if str(row["family"]) == family}):
            current = [row for row in rows if str(row["family"]) == family and int(row["seed"]) == seed]
            grouped.append(_aggregate(current, family=family, seed=seed, field="overall", value="all"))
            for field in ("num_buoys", "eta", "quality_tier", "behavior_generator", "conflict", "bridge"):
                for value in sorted({str(row[field]) for row in current}):
                    grouped.append(_aggregate([row for row in current if str(row[field]) == value], family=family, seed=seed, field=field, value=value))
    return grouped


def _paired(left: list[dict[str, Any]], right: list[dict[str, Any]], *, comparison: str) -> list[dict[str, Any]]:
    index = {(int(row["seed"]), str(row["sample_id"])): row for row in right}
    result: list[dict[str, Any]] = []
    for row in left:
        peer = index.get((int(row["seed"]), str(row["sample_id"])))
        if peer is None:
            raise RuntimeError(f"offline paired alignment lost {row['sample_id']}")
        result.append({"comparison": comparison, "seed": int(row["seed"]), "sample_id": row["sample_id"], "scenario_id": row["scenario_id"], "num_buoys": int(row["num_buoys"]), "eta": float(row["eta"]), "conflict": bool(row["conflict"]), "bridge": bool(row["bridge"]), **{f"delta_{metric}": float(row[metric]) - float(peer[metric]) for metric in SCALAR_METRICS}})
    return result


def _comparison_summary(rows: list[dict[str, Any]], *, comparison: str, field: str, value: str) -> dict[str, Any]:
    return {"comparison": comparison, "group_field": field, "group_value": value, "sample_count": len(rows), **{f"mean_delta_{metric}": float(np.mean([row[f"delta_{metric}"] for row in rows])) for metric in SCALAR_METRICS}, "task_waypoint_rmse_improvement": float(-np.mean([row["delta_raw_action_rmse"] for row in rows]))}


def _comparison_groups(rows: list[dict[str, Any]], comparison: str) -> list[dict[str, Any]]:
    result = [_comparison_summary(rows, comparison=comparison, field="overall", value="all")]
    for field in ("num_buoys", "eta", "conflict", "bridge"):
        for value in sorted({str(row[field]) for row in rows}):
            result.append(_comparison_summary([row for row in rows if str(row[field]) == value], comparison=comparison, field=field, value=value))
    return result


def _utilization(model, normalizer, sequences, seed: int, config: TrainingConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baselines = head_utilization_baselines(np.empty((0,), dtype=np.float32), sequences, seed)
    normal = None; rows: list[dict[str, Any]] = []
    for variant, phi in baselines.items():
        predicted = predict(PREDICTED_TRANSFORMER, model, normalizer.apply(sequences.replace(predicted_phi=phi)), config)
        rmse = raw_rmse(predicted, sequences.actions, sequences.uav_mask[:, -1])
        if normal is None:
            normal = predicted
            baseline_rmse = rmse
        valid = sequences.uav_mask[:, -1][..., None]
        sensitivity = float(np.sum(np.linalg.norm(predicted - normal, axis=-1) * valid[..., 0]) / max(int(valid.sum()), 1))
        diff = predicted - sequences.actions
        dot = (predicted * sequences.actions).sum(axis=-1)
        denominator = np.linalg.norm(predicted, axis=-1) * np.linalg.norm(sequences.actions, axis=-1)
        cosine = np.divide(dot, denominator, out=np.ones_like(dot), where=denominator > 1.0e-8)
        magnitude = np.abs(np.linalg.norm(predicted, axis=-1) - np.linalg.norm(sequences.actions, axis=-1))
        rows.append({"family": PREDICTED_TRANSFORMER, "seed": seed, "variant": variant, "raw_joint_action_rmse": rmse, "rmse_change_from_normal": rmse - baseline_rmse, "action_sensitivity": sensitivity, "cosine": float((cosine * valid[..., 0]).sum() / max(int(valid.sum()), 1)), "magnitude_error": float((magnitude * valid[..., 0]).sum() / max(int(valid.sum()), 1)), "uses_true_phi": False, "validation_only": True, "retrained": False})
    by_variant = {row["variant"]: row for row in rows}
    heads = {"service": by_variant["zero_service"]["action_sensitivity"], "progress": by_variant["zero_progress"]["action_sensitivity"], "return": by_variant["zero_return"]["action_sensitivity"]}
    summary = {"seed": seed, "sensitivity_service": heads["service"], "sensitivity_progress": heads["progress"], "sensitivity_return": heads["return"], "dominant_policy_head": max(heads, key=heads.get)}
    return rows, summary


def run_offline_analysis() -> dict[str, Any]:
    """Evaluate all formal checkpoints and inference-only Phi perturbations."""
    runs = _formal_runs(); predictor = FrozenDeepSetsPredictor(); config = TrainingConfig()
    sequences = load_policy_sequences(VALIDATION, predictor)
    all_rows: list[dict[str, Any]] = []; run_summary: list[dict[str, Any]] = []; utilization: list[dict[str, Any]] = []; head_summary: list[dict[str, Any]] = []
    for run in runs:
        family, model, normalizer, payload = load_checkpoint(Path(run["checkpoint"]), config.device)
        if family != run["family"] or int(payload["seed"]) != int(run["seed"]):
            raise RuntimeError("formal checkpoint identity changed before offline analysis")
        predicted = predict(family, model, normalizer.apply(sequences), config)
        raw_rows, _ = offline_metrics(family, int(run["seed"]), predicted, sequences)
        rows = _with_eta(raw_rows); all_rows.extend(rows)
        run_summary.append({"family": family, "seed": int(run["seed"]), "raw_joint_action_rmse": raw_rmse(predicted, sequences.actions, sequences.uav_mask[:, -1]), "best_epoch": int(run["best_epoch"]), "best_validation_raw_joint_action_rmse": float(run["best_validation_raw_joint_action_rmse"]), "checkpoint": run["checkpoint"], "checkpoint_sha256": run["checkpoint_sha256"]})
        if family == PREDICTED_TRANSFORMER:
            detail, summary = _utilization(model, normalizer, sequences, int(run["seed"]), config)
            utilization.extend(detail); head_summary.append(summary)
    grouped = _grouped(all_rows)
    by_family = {family: [row for row in all_rows if row["family"] == family] for family in (PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER, PREDICTED_MLP, ORACLE_TRANSFORMER)}
    state_pair = _paired(by_family[PREDICTED_TRANSFORMER], by_family[STATE_ONLY_TRANSFORMER], comparison="predicted_task_waypoint_transformer_minus_state_only")
    mlp_pair = _paired(by_family[PREDICTED_TRANSFORMER], by_family[PREDICTED_MLP], comparison="predicted_task_waypoint_transformer_minus_mlp")
    oracle_pair = _paired(by_family[PREDICTED_TRANSFORMER], by_family[ORACLE_TRANSFORMER], comparison="predicted_task_waypoint_transformer_minus_oracle")
    conflict = _comparison_groups([row for row in state_pair if row["conflict"]], "task_waypoint_minus_state_conflict") + _comparison_groups([row for row in state_pair if not row["conflict"]], "task_waypoint_minus_state_nonconflict")
    bridge = _comparison_groups([row for row in state_pair if row["bridge"]], "task_waypoint_minus_state_bridge") + _comparison_groups([row for row in state_pair if not row["bridge"]], "task_waypoint_minus_state_nonbridge")
    oracle_gap = []
    for row in _comparison_groups(oracle_pair, "predicted_minus_oracle"):
        oracle_gap.append({**row, "oracle_gap_raw_action_rmse": row["mean_delta_raw_action_rmse"]})
    transformer_mlp = _comparison_groups(mlp_pair, "transformer_minus_mlp")
    for row in transformer_mlp:
        row["transformer_rmse_improvement_over_mlp"] = row["task_waypoint_rmse_improvement"]
    _csv(R22_OUTPUT / "offline_validation_metrics.csv", all_rows)
    _csv(R22_OUTPUT / "offline_grouped_metrics.csv", grouped)
    _csv(R22_OUTPUT / "offline_policy_run_summary.csv", run_summary)
    _csv(R22_OUTPUT / "oracle_task_waypoint_policy_gap.csv", oracle_gap)
    _csv(R22_OUTPUT / "task_waypoint_head_utilization.csv", utilization)
    _csv(R22_OUTPUT / "task_waypoint_head_utilization_summary.csv", head_summary)
    _csv(R22_OUTPUT / "conflict_policy_analysis.csv", conflict)
    _csv(R22_OUTPUT / "bridge_policy_analysis.csv", bridge)
    _csv(R22_OUTPUT / "transformer_vs_mlp_offline.csv", transformer_mlp)
    _json(R22_OUTPUT / "round22_offline_analysis_manifest.json", {"completed": True, "split": VALIDATION, "formal_checkpoint_count": len(runs), "uses_true_phi_for_deployable": False, "oracle_offline_only": True, "predictor_sha256_before": predictor.sha256, "predictor_sha256_after": predictor.sha256, "development_test_access_count": 0, "untouched_eval_access_count": 0})
    return {"run_summary": run_summary, "head_summary": head_summary, "state_comparison": _comparison_groups(state_pair, "task_waypoint_minus_state"), "transformer_mlp": transformer_mlp, "oracle_gap": oracle_gap, "predictor_sha256": predictor.sha256}


def main() -> None:
    result = run_offline_analysis()
    print(json.dumps({"offline_runs": len(result["run_summary"]), "predictor_sha256": result["predictor_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
