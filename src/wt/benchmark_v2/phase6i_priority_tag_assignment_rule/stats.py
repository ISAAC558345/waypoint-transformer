"""Scenario-level bootstrap. Strategies and timesteps are not independent draws."""
from __future__ import annotations

import numpy as np

from .protocol import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, SCENARIOS


def bootstrap_mean(values) -> dict:
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    if sample.size != len(SCENARIOS):
        raise RuntimeError("bootstrap expects one paired difference per registered scenario")
    if not np.all(np.isfinite(sample)):
        raise RuntimeError("bootstrap values are not finite")
    rng = np.random.default_rng(int(BOOTSTRAP_SEED))
    draws = rng.integers(0, int(sample.size), size=(int(BOOTSTRAP_DRAWS), int(sample.size)))
    means = sample[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(sample.mean()),
        "median": float(np.median(sample)),
        "ci_low": float(low),
        "ci_high": float(high),
        "crosses_zero": bool(float(low) <= 0.0 <= float(high)),
        "draws": int(BOOTSTRAP_DRAWS),
        "seed": int(BOOTSTRAP_SEED),
        "unit": "scenario",
    }
