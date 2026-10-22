"""Diagnostic gate. Thresholds stay in protocol.py."""
from __future__ import annotations

import numpy as np


def choose_gate(
    *,
    aligned: bool,
    sinr_ok: bool,
    progress_ok: bool,
    interference_ok: bool,
    all_single_below: bool,
    combined_ok: bool,
) -> str:
    """ERR-F, then ERR-A, then B or C, then ERR-D, otherwise ERR-E."""
    if not bool(aligned):
        return "ERR-F"
    if bool(sinr_ok):
        return "ERR-A"
    if bool(progress_ok) and not bool(interference_ok):
        return "ERR-B"
    if bool(interference_ok) and not bool(progress_ok):
        return "ERR-C"
    if bool(all_single_below) and bool(combined_ok):
        return "ERR-D"
    return "ERR-E"


def pairwise_accuracy(truth: np.ndarray, prediction: np.ndarray, eps: float) -> float:
    left = np.asarray(truth, dtype=np.float64)
    right = np.asarray(prediction, dtype=np.float64)
    delta_true = left[:, None] - left[None, :]
    delta_pred = right[:, None] - right[None, :]
    upper = np.triu(np.ones(left.shape[0], dtype=bool), k=1)
    valid = upper & np.isfinite(delta_true) & np.isfinite(delta_pred) & (np.abs(delta_true) > float(eps))
    if not np.any(valid):
        return float("nan")
    return float(np.mean(delta_true[valid] * delta_pred[valid] > 0.0))


def recovery_ratio(gain: float, gap: float) -> float:
    if not np.isfinite(gain) or not np.isfinite(gap) or float(gap) <= 0.0:
        return float("nan")
    return float(gain) / float(gap)


__all__ = ["choose_gate", "pairwise_accuracy", "recovery_ratio"]
