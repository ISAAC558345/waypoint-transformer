"""Per-state robust utility normalization and soft targets. Frozen before validation outcomes."""
from __future__ import annotations

import numpy as np

from .protocol import DELTA_RANGE_FRAC, NORM_EPS, TAU


def mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def normalize_state(utilities: np.ndarray) -> np.ndarray:
    arr = np.asarray(utilities, dtype=np.float64)
    center = float(np.median(arr))
    scale = mad(arr) + float(NORM_EPS)
    return (arr - center) / scale


def soft_targets(utilities: np.ndarray, tau: float = TAU) -> np.ndarray:
    u = normalize_state(utilities)
    z = u / max(float(tau), 1.0e-9)
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / max(float(exp.sum()), 1.0e-12)


def ranking_pairs(utilities: np.ndarray) -> list[tuple[int, int, float, float]]:
    arr = np.asarray(utilities, dtype=np.float64)
    span = float(np.max(arr) - np.min(arr))
    delta = max(span * float(DELTA_RANGE_FRAC), 1.0)
    pairs = []
    for i in range(arr.size):
        for j in range(i + 1, arr.size):
            diff = float(arr[i] - arr[j])
            if abs(diff) <= delta:
                continue
            y = 1.0 if diff > 0 else -1.0
            w = min(abs(diff) / (span + 1.0e-9), 1.0)
            pairs.append((i, j, y, w))
    return pairs
