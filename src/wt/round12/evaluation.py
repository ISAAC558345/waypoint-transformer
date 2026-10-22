"""Offline-only action metrics for validation selection and smoke checks."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.round5.round5_evaluation import compute_action_metrics, radial_action_projection


def evaluate_offline_actions(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    max_step_distance_m: float,
) -> dict[str, Any]:
    """Return finite raw/projected action metrics without environment rollout."""
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if target.ndim != 2 or target.shape[1] != 2 or prediction.shape != target.shape:
        raise ValueError("Round-12 action evaluation requires matching [N,2] arrays")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise FloatingPointError("action evaluation received NaN or Inf")
    projection = radial_action_projection(prediction, max_step_distance_m)
    metrics = compute_action_metrics(
        target,
        prediction,
        max_step_distance_m,
        prediction_space="raw",
        projection_activation_mask=projection.activation_mask,
    )
    # These concise aliases are stable Round-12 selection/audit names.
    metrics.update({
        "raw_action_rmse": float(metrics["action_l2_rmse_m"]),
        "raw_action_mae": float(metrics["action_l2_mae_m"]),
        "cosine_similarity": float(metrics["action_cosine_similarity"]),
        "angular_error_deg": float(metrics["action_angular_error_deg"]),
        "magnitude_error": float(metrics["action_magnitude_mae_m"]),
    })
    return metrics


__all__ = ["evaluate_offline_actions"]
