"""Locked sensitivity gate. Bars stay in protocol.py."""
from __future__ import annotations

import numpy as np

from .protocol import UNIQUE_RELATIVE_GAP, VALUE_EPS


def choose_gate(*, aligned: bool, first_order_ok: bool, interaction_ok: bool, ordering_failure: bool) -> str:
    """SENS-D, then SENS-A, then SENS-B, then SENS-C, otherwise SENS-E."""
    if not bool(aligned):
        return "SENS-D"
    if bool(first_order_ok):
        return "SENS-A"
    if bool(interaction_ok):
        return "SENS-B"
    if bool(ordering_failure):
        return "SENS-C"
    return "SENS-E"


def sign_agreement(left, right, eps: float = VALUE_EPS) -> float:
    observed = np.asarray(left, dtype=np.float64).reshape(-1)
    predicted = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if not np.any(valid):
        return float("nan")
    observed = observed[valid]
    predicted = predicted[valid]
    same = ((observed > eps) & (predicted > eps)) | ((observed < -eps) & (predicted < -eps))
    both_small = (np.abs(observed) <= eps) & (np.abs(predicted) <= eps)
    return float(np.mean(same | both_small))


def unique_dominant(contrib: dict[str, float], gap: float = UNIQUE_RELATIVE_GAP) -> str | None:
    """Most negative contribution, or None when the lead is not unique."""
    if len(contrib) < 2:
        return None
    ordered = sorted(contrib, key=lambda name: (float(contrib[name]), str(name)))
    best, second = ordered[0], ordered[1]
    if float(contrib[best]) >= -float(VALUE_EPS):
        return None
    scale = float(sum(abs(float(value)) for value in contrib.values()))
    if scale <= float(VALUE_EPS):
        return None
    if float(contrib[second]) - float(contrib[best]) < float(gap) * scale:
        return None
    return str(best)


def trapezoid(samples: np.ndarray) -> np.ndarray:
    """Integrate samples[0]=f(0) ... samples[-1]=f(1) with the trapezoid rule."""
    values = np.asarray(samples, dtype=np.float64)
    count = int(values.shape[0])
    if count < 2:
        raise ValueError("a path integral needs both endpoints")
    step = 1.0 / float(count - 1)
    weights = np.full(count, step, dtype=np.float64)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return np.tensordot(weights, values, axes=(0, 0))


def group_dot(grad: np.ndarray, error: np.ndarray, indexes: tuple[int, ...]) -> np.ndarray:
    chosen = list(indexes)
    return np.sum(grad[..., chosen] * error[..., chosen], axis=(-1, -2))


__all__ = ["choose_gate", "group_dot", "sign_agreement", "trapezoid", "unique_dominant"]
