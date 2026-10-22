"""Cluster bootstrap and the generalization gate. Thresholds live in protocol.py."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .protocol import PAYLOAD_BLOCK_BITS, POOL_HIT_FRACTION


def cluster_bootstrap(pairs: list[dict], key: str, *, draws: int, seed: int) -> dict[str, float]:
    """Resample scenarios, then keep every time and model seed inside the draw."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in pairs:
        value = float(row[key])
        if np.isfinite(value):
            grouped[int(row["scenario_id"])].append(value)
    scenarios = sorted(grouped)
    if not scenarios:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    observed = [value for scenario in scenarios for value in grouped[scenario]]
    rng = np.random.default_rng(int(seed))
    stats = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        draw = rng.choice(scenarios, size=len(scenarios), replace=True)
        sample = [value for scenario in draw for value in grouped[int(scenario)]]
        stats[index] = float(np.mean(sample))
    return {
        "mean": float(np.mean(observed)),
        "low": float(np.percentile(stats, 2.5)),
        "high": float(np.percentile(stats, 97.5)),
    }


def choose_gate(
    *,
    mean_delta_j: float,
    delta_j_low: float,
    n_scenarios_up: int,
    n_scenarios: int,
    n_seeds_up: int,
    n_seeds: int,
    mean_delta_regret: float,
    regret_high: float,
    coverage_ok: bool,
    pool_hit_fraction: float,
    mean_coverage_regret: float,
) -> str:
    """One conclusion. Coverage failure is checked before calling the scorer confirmed."""
    if int(n_scenarios) <= 0 or int(n_seeds) <= 0:
        raise ValueError("gate needs scenarios and seeds")
    majority_scenarios = int(n_scenarios_up) * 2 > int(n_scenarios)
    majority_seeds = int(n_seeds_up) * 2 > int(n_seeds)
    regret_down = float(mean_delta_regret) < 0.0 and float(regret_high) < 0.0
    stable_gain = (
        float(mean_delta_j) > 0.0
        and float(delta_j_low) > 0.0
        and majority_scenarios
        and majority_seeds
        and regret_down
    )
    coverage_bottleneck = float(pool_hit_fraction) >= float(POOL_HIT_FRACTION) and float(mean_coverage_regret) >= float(PAYLOAD_BLOCK_BITS)
    if coverage_bottleneck:
        return "R-C"
    if stable_gain and bool(coverage_ok):
        return "R-A"
    if float(mean_delta_j) > 0.0:
        return "R-B"
    return "R-D"


__all__ = ["choose_gate", "cluster_bootstrap"]
