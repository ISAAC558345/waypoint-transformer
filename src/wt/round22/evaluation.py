"""Offline behavioral-cloning evaluation and required subgroup summaries."""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import PolicySequences


def _per_record(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    diff = predicted - target
    valid = np.asarray(mask, dtype=bool)
    squared = (diff * diff).sum(axis=-1)
    mae = np.abs(diff).mean(axis=-1)
    dot = (predicted * target).sum(axis=-1); norm = np.linalg.norm(predicted, axis=-1) * np.linalg.norm(target, axis=-1)
    cosine = np.divide(dot, norm, out=np.ones_like(dot), where=norm > 1.0e-8)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    magnitude = np.abs(np.linalg.norm(predicted, axis=-1) - np.linalg.norm(target, axis=-1))
    projection = np.linalg.norm(predicted, axis=-1) > 1.0
    def mean(value: np.ndarray) -> np.ndarray:
        return np.divide((value * valid).sum(axis=1), valid.sum(axis=1), out=np.zeros(value.shape[0]), where=valid.sum(axis=1) > 0)
    return {"normalized_mse": mean(squared), "raw_action_rmse": np.sqrt(mean(squared)), "per_uav_rmse": np.sqrt(squared), "mae": mean(mae), "cosine": mean(cosine), "angular_error_deg": mean(angle), "magnitude_error": mean(magnitude), "projection_rate": mean(projection.astype(np.float64))}


def offline_metrics(family: str, seed: int, predicted: np.ndarray, sequences: PolicySequences) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values = _per_record(predicted, sequences.actions, sequences.uav_mask[:, -1])
    meta = sequences.metadata; rows: list[dict[str, Any]] = []
    scalar_metrics = [name for name, value in values.items() if value.ndim == 1]
    for index in range(len(sequences)):
        base = {"family": family, "seed": seed, "split": sequences.split, "sample_id": str(meta["sample_id"][index]), "scenario_id": str(meta["scenario_id"][index]), "num_buoys": int(meta["num_buoys"][index]), "interference_tier": str(meta["interference_tier"][index]), "quality_tier": str(meta["quality_tier"][index]), "behavior_generator": str(meta["behavior_generator"][index]), "conflict": bool(meta["conflict"][index]), "bridge": bool(meta["bridge"][index])}
        metrics = {name: float(values[name][index]) for name in scalar_metrics}
        metrics.update({f"per_uav_{uav}_rmse": float(values["per_uav_rmse"][index, uav]) for uav in range(values["per_uav_rmse"].shape[1])})
        rows.append({**base, **metrics})
    grouped: list[dict[str, Any]] = []
    for field in ("num_buoys", "interference_tier", "quality_tier", "behavior_generator", "conflict", "bridge", "seed"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows: groups.setdefault(str(row[field]), []).append(row)
        for value, items in sorted(groups.items()):
            metric_names = scalar_metrics + [f"per_uav_{uav}_rmse" for uav in range(values["per_uav_rmse"].shape[1])]
            grouped.append({"family": family, "split": sequences.split, "group_field": field, "group_value": value, "sample_count": len(items), **{metric: float(np.mean([row[metric] for row in items])) for metric in metric_names}})
    return rows, grouped


def head_utilization_baselines(predicted: np.ndarray, sequences: PolicySequences, seed: int) -> dict[str, np.ndarray]:
    """Validation-only no-retraining Phi perturbations for later formal analysis."""
    rng = np.random.default_rng(2200 + seed)
    phi = sequences.predicted_phi.copy(); shuffled = phi[rng.permutation(len(phi))]
    result = {"normal": phi, "zero_full": np.zeros_like(phi), "shuffle_full": shuffled, "zero_service": phi.copy(), "zero_progress": phi.copy(), "zero_return": phi.copy(), "shuffle_service": phi.copy(), "shuffle_progress": phi.copy(), "shuffle_return": phi.copy()}
    result["zero_service"][..., :5] = 0; result["zero_progress"][..., 5:10] = 0; result["zero_return"][..., 10:] = 0
    result["shuffle_service"][..., :5] = shuffled[..., :5]; result["shuffle_progress"][..., 5:10] = shuffled[..., 5:10]; result["shuffle_return"][..., 10:] = shuffled[..., 10:]
    return result


__all__ = ["head_utilization_baselines", "offline_metrics"]
