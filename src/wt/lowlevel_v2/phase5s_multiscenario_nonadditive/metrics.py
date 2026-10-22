"""Gate for the data-diversity test. Thresholds are fixed in protocol.py."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .protocol import CLOSE_ORACLE_BITS, PAYLOAD_BLOCK_BITS, TIE_ABS_BITS


def cluster_bootstrap(pairs: list[dict], key: str, *, draws: int, seed: int) -> dict[str, float]:
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
    mean_delta_regret: float,
    n_scenarios_up: int,
    n_scenarios: int,
    n_seeds_up: int,
    n_seeds: int,
    regret_phase5q: float,
    regret_phase5s: float,
    coverage_regret: float,
) -> str:
    """Phase-5S minus Phase-5Q. Confirmation scenarios are not used to retune this rule."""
    if int(n_scenarios) <= 0 or int(n_seeds) <= 0:
        raise ValueError("gate needs scenarios and seeds")
    majority_scenarios = int(n_scenarios_up) * 2 > int(n_scenarios)
    majority_seeds = int(n_seeds_up) * 2 > int(n_seeds)
    stable = (
        float(mean_delta_j) > 0.0
        and float(delta_j_low) > 0.0
        and float(mean_delta_regret) < 0.0
        and majority_scenarios
        and majority_seeds
    )
    if stable:
        return "S-A"
    regret_down = float(mean_delta_regret) < 0.0
    if float(mean_delta_j) > 0.0 and (
        float(delta_j_low) <= 0.0 or not majority_scenarios or not majority_seeds or not regret_down
    ):
        return "S-C"
    tied = abs(float(mean_delta_j)) < float(TIE_ABS_BITS)
    both_close = float(regret_phase5q) < float(CLOSE_ORACLE_BITS) and float(regret_phase5s) < float(CLOSE_ORACLE_BITS)
    if tied and both_close and float(coverage_regret) > float(regret_phase5s):
        return "S-B"
    return "S-D"


__all__ = ["choose_gate", "cluster_bootstrap", "PAYLOAD_BLOCK_BITS"]
