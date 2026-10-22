"""Bootstrap that does not copy shared L0 into extra independent samples."""
from __future__ import annotations

from typing import Any

import numpy as np


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(np.percentile(samples, 100.0 * alpha / 2.0))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_deltas(delta: np.ndarray, *, n_boot: int, seed: int) -> dict[str, Any]:
    if delta.ndim != 2:
        raise ValueError("delta must be (n_seeds, n_scenarios)")
    n_seeds, n_scen = delta.shape
    rng = np.random.default_rng(int(seed))
    seed_means = delta.mean(axis=1)
    overall = float(delta.mean())
    scen_means = delta.mean(axis=0)
    scen_boot = np.empty(int(n_boot), dtype=np.float64)
    seed_boot = np.empty(int(n_boot), dtype=np.float64)
    hier_boot = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        scen_idx = rng.integers(0, n_scen, size=n_scen)
        scen_boot[b] = float(delta[:, scen_idx].mean())
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        seed_boot[b] = float(seed_means[seed_idx].mean())
        hier_vals = []
        hier_seeds = rng.integers(0, n_seeds, size=n_seeds)
        for s in hier_seeds:
            inner = rng.integers(0, n_scen, size=n_scen)
            hier_vals.append(float(delta[int(s), inner].mean()))
        hier_boot[b] = float(np.mean(hier_vals))
    scen_ci = percentile_ci(scen_boot)
    seed_ci = percentile_ci(seed_boot)
    hier_ci = percentile_ci(hier_boot)
    return {
        "n_seeds": int(n_seeds),
        "n_scenarios": int(n_scen),
        "n_boot": int(n_boot),
        "boot_seed": int(seed),
        "overall_mean_delta": overall,
        "median_delta_all_pairs": float(np.median(delta)),
        "median_delta_seed_means": float(np.median(seed_means)),
        "per_seed_mean_delta": [float(x) for x in seed_means],
        "per_scenario_mean_delta": [float(x) for x in scen_means],
        "n_positive_seeds": int(np.sum(seed_means > 0.0)),
        "positive_seed_fraction": float(np.mean(seed_means > 0.0)),
        "scenario_paired_ci95": {"low": scen_ci[0], "high": scen_ci[1]},
        "seed_cluster_ci95": {"low": seed_ci[0], "high": seed_ci[1]},
        "hierarchical_ci95": {"low": hier_ci[0], "high": hier_ci[1]},
        "pairing_note": "L0 is shared per scenario; never duplicated as five independent baselines",
    }
