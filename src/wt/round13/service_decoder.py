"""Diagnostic-only continuous service-priority decoder for the Round-13 amendment.

The functions in this module deliberately do not import or modify
``MaritimeEnv``.  They map *current* observable queue and link-rate vectors to
legal fractions and payloads for a sidecar diagnostic only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPSILON = 1.0e-12


@dataclass(frozen=True)
class ContinuousServiceAllocation:
    """A finite masked-simplex allocation derived from current observations."""

    priority: np.ndarray
    fraction: np.ndarray
    used_fixed_equal_fallback: bool


def _vectors(
    queue_bits: np.ndarray,
    full_bandwidth_rate_bps: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queue = np.asarray(queue_bits, dtype=np.float64)
    rate = np.asarray(full_bandwidth_rate_bps, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if queue.ndim != 1 or rate.shape != queue.shape or valid.shape != queue.shape:
        raise ValueError("queue, full-bandwidth rate, and valid mask must be matching vectors")
    if not np.all(np.isfinite(queue)) or not np.all(np.isfinite(rate)):
        raise ValueError("decoder inputs must be finite")
    if np.any(queue < 0.0) or np.any(rate < 0.0):
        raise ValueError("queue and full-bandwidth rate must be nonnegative")
    return queue, rate, valid


def masked_simplex(priority: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Validate and normalize an explicit priority, retaining zero padding."""
    score = np.asarray(priority, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if score.ndim != 1 or valid.shape != score.shape:
        raise ValueError("priority and valid mask must be matching vectors")
    if not np.all(np.isfinite(score)) or np.any(score < 0.0):
        raise ValueError("priority must be finite and nonnegative")
    if np.any(score[~valid] != 0.0):
        raise ValueError("padding-buoy priority must be zero")
    result = np.zeros_like(score)
    if not np.any(valid):
        return result
    total = float(score[valid].sum())
    if total <= EPSILON:
        raise ValueError("a nonempty valid mask requires positive priority mass")
    result[valid] = score[valid] / total
    if np.any(result < 0.0) or not np.isclose(float(result.sum()), 1.0):
        raise RuntimeError("masked simplex contract violated")
    return result


def continuous_priority_from_current_state(
    queue_bits: np.ndarray,
    full_bandwidth_rate_bps: np.ndarray,
    valid_mask: np.ndarray,
) -> ContinuousServiceAllocation:
    """Return ``normalized_queue * normalized_current_rate`` priority.

    When every valid score is zero, the deterministic fallback is the current
    fixed-equal resource mechanism, not a random assignment.  No future queue,
    rate, reward, payload, position, or trajectory value is accepted here.
    """
    queue, rate, valid = _vectors(queue_bits, full_bandwidth_rate_bps, valid_mask)
    priority = np.zeros_like(queue)
    if not np.any(valid):
        return ContinuousServiceAllocation(priority, priority.copy(), True)
    queue_scale = float(queue[valid].max())
    rate_scale = float(rate[valid].max())
    if queue_scale > EPSILON and rate_scale > EPSILON:
        priority[valid] = (queue[valid] / queue_scale) * (rate[valid] / rate_scale)
    if float(priority[valid].sum()) <= EPSILON:
        fraction = np.zeros_like(queue)
        fraction[valid] = 1.0 / float(valid.sum())
        return ContinuousServiceAllocation(fraction.copy(), fraction, True)
    fraction = masked_simplex(priority, valid)
    return ContinuousServiceAllocation(fraction.copy(), fraction, False)


def allocate_payload(
    queue_bits: np.ndarray,
    full_bandwidth_rate_bps: np.ndarray,
    fraction: np.ndarray,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return capacity and payload subject to capacity and queue constraints."""
    queue = np.asarray(queue_bits, dtype=np.float64)
    rate = np.asarray(full_bandwidth_rate_bps, dtype=np.float64)
    share = np.asarray(fraction, dtype=np.float64)
    if queue.ndim != 1 or rate.shape != queue.shape or share.shape != queue.shape:
        raise ValueError("queue, rate, and fraction must be matching vectors")
    if not np.all(np.isfinite(share)) or np.any(share < 0.0):
        raise ValueError("resource fraction must be finite and nonnegative")
    if float(share.sum()) > 1.0 + EPSILON:
        raise ValueError("resource fractions must sum to at most one")
    if not np.isfinite(float(delta_t)) or float(delta_t) <= 0.0:
        raise ValueError("delta_t must be finite and positive")
    capacity = share * rate * float(delta_t)
    payload = np.minimum(queue, capacity)
    if np.any(payload > queue + EPSILON) or np.any(payload > capacity + EPSILON):
        raise RuntimeError("payload constraints violated")
    return capacity, payload


def jain_fairness(values: np.ndarray) -> float:
    """Return Jain fairness on a nonnegative vector, with zero-service = 1."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("fairness values must be a finite nonnegative vector")
    denominator = float(array.size * np.square(array).sum())
    return 1.0 if denominator <= EPSILON else float(np.square(array.sum()) / denominator)


__all__ = [
    "ContinuousServiceAllocation",
    "EPSILON",
    "allocate_payload",
    "continuous_priority_from_current_state",
    "jain_fairness",
    "masked_simplex",
]
