"""Hierarchical and seed-cluster bootstrap for paired HGNN-DeepSets deltas."""
from __future__ import annotations

import numpy as np


def hierarchical_bootstrap(delta: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    """delta shape (R, S). Resample model seeds, then scenarios within seed."""
    values = np.asarray(delta, dtype=np.float64)
    n_r, n_s = values.shape
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    for index in range(int(n_boot)):
        rows = rng.integers(0, n_r, size=n_r)
        acc = []
        for row in rows:
            cols = rng.integers(0, n_s, size=n_s)
            acc.append(values[int(row), cols])
        means[index] = float(np.mean(np.concatenate(acc)))
    lo, hi = np.percentile(means, (2.5, 97.5))
    return {"mean": float(np.mean(values)), "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": int(n_boot)}


def seed_cluster_bootstrap(delta: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    """Resample model-seed mean deltas only."""
    values = np.asarray(delta, dtype=np.float64)
    seed_means = values.mean(axis=1)
    rng = np.random.default_rng(int(seed) + 17)
    means = np.empty(int(n_boot), dtype=np.float64)
    for index in range(int(n_boot)):
        draw = rng.choice(seed_means, size=seed_means.size, replace=True)
        means[index] = float(np.mean(draw))
    lo, hi = np.percentile(means, (2.5, 97.5))
    return {"mean": float(np.mean(seed_means)), "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": int(n_boot)}


__all__ = ["hierarchical_bootstrap", "seed_cluster_bootstrap"]
