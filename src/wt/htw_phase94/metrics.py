"""Phase-9.4 assignment metrics. Ranking accuracy is not primary."""
from __future__ import annotations

import numpy as np

from wt.htw_phase9.const import IDLE_INDEX


def oracle_regret(j_star: float, j_hat: float) -> float:
    return float(j_star) - float(j_hat)


def normalized_regret(j_star: float, j_hat: float, eps: float = 1.0e-9) -> float:
    return oracle_regret(j_star, j_hat) / (abs(float(j_star)) + float(eps))


def exact_match(pred: np.ndarray, oracle: np.ndarray) -> float:
    return float(np.array_equal(np.asarray(pred, dtype=np.int64), np.asarray(oracle, dtype=np.int64)))


def per_uav_accuracy(pred: np.ndarray, oracle: np.ndarray) -> float:
    left = np.asarray(pred, dtype=np.int64)
    right = np.asarray(oracle, dtype=np.int64)
    return float(np.mean(left == right))


def idle_fraction(assignment: np.ndarray) -> float:
    values = np.asarray(assignment, dtype=np.int64)
    return float(np.mean(values < 0))


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    if values.size == 0:
        return 0.0, 0.0, 0.0
    means = np.empty(int(n_boot), dtype=np.float64)
    for index in range(int(n_boot)):
        draw = rng.choice(values, size=values.size, replace=True)
        means[index] = float(np.mean(draw))
    lo, hi = np.percentile(means, (2.5, 97.5))
    return float(np.mean(values)), float(lo), float(hi)


def sequence_from_assignment(assignment: np.ndarray) -> np.ndarray:
    values = np.asarray(assignment, dtype=np.int64)
    seq = np.full(values.shape, IDLE_INDEX, dtype=np.int64)
    seq = np.where(values >= 0, values, seq)
    return seq


__all__ = [
    "exact_match",
    "idle_fraction",
    "normalized_regret",
    "oracle_regret",
    "paired_bootstrap_ci",
    "per_uav_accuracy",
    "sequence_from_assignment",
]
