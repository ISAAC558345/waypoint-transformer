"""Pure prospective service-intent decoder contracts for Round-13.

These helpers are deliberately not connected to ``MaritimeEnv.action_space``.
They specify a future physical decoder and support diagnostic-only calculations.
"""
from __future__ import annotations

import numpy as np


EPSILON = 1.0e-12


def valid_masked_priority_fractions(priority: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Map finite nonnegative priorities to a masked resource simplex."""
    score = np.asarray(priority, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if score.ndim != 1 or valid.shape != score.shape:
        raise ValueError("priority and valid mask must be matching one-dimensional arrays")
    if not np.all(np.isfinite(score)) or np.any(score < 0.0):
        raise ValueError("priority must be finite and nonnegative")
    result = np.zeros_like(score)
    total = float(score[valid].sum())
    if total > EPSILON:
        result[valid] = score[valid] / total
    elif np.any(valid):
        result[valid] = 1.0 / float(valid.sum())
    if np.any(result < 0.0) or not np.isclose(float(result.sum()), 1.0 if np.any(valid) else 0.0):
        raise RuntimeError("service-priority decoder violated its simplex contract")
    return result


def discrete_service_fractions(selected_buoy: int | None, valid_mask: np.ndarray) -> np.ndarray:
    """Map a valid discrete service intent to one masked resource fraction."""
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 1:
        raise ValueError("valid mask must be one-dimensional")
    result = np.zeros(valid.size, dtype=np.float64)
    if selected_buoy is None:
        return result
    selected = int(selected_buoy)
    if selected < 0 or selected >= valid.size or not valid[selected]:
        raise ValueError("discrete service intent must select a valid buoy")
    result[selected] = 1.0
    return result


def current_state_service_heuristic(queue_bits: np.ndarray, full_bandwidth_rate_bps: np.ndarray, valid_mask: np.ndarray) -> int | None:
    """Choose ``argmax(Q_i * R_i)`` from current observable quantities only."""
    queue = np.asarray(queue_bits, dtype=np.float64)
    rate = np.asarray(full_bandwidth_rate_bps, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if queue.ndim != 1 or queue.shape != rate.shape or valid.shape != queue.shape:
        raise ValueError("queue, rate, and mask must have matching one-dimensional shapes")
    if not np.all(np.isfinite(queue)) or not np.all(np.isfinite(rate)) or np.any(queue < 0.0) or np.any(rate < 0.0):
        raise ValueError("queue and rate must be finite and nonnegative")
    eligible = valid & (queue > 0.0)
    if not np.any(eligible):
        return None
    return int(np.argmax(np.where(eligible, queue * rate, -1.0)))


__all__ = ["EPSILON", "current_state_service_heuristic", "discrete_service_fractions", "valid_masked_priority_fractions"]
