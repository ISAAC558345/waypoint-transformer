"""Baseline utility. Coefficients are fixed and not fit."""
from __future__ import annotations

BASELINE_CONFIG = {
    "name": "phase5e_baseline",
    "weights_fitted": False,
    "alpha": 1.0,
    "beta": 0.0,
    "gamma": 0.0,
    "delta": 0.0,
    "eta": 0.0,
    "hold_steps": 15,
    "executor": "fly_toward",
    "energy_model_present": False,
    "energy_cost_fixed": 0.0,
    "rationale": {
        "alpha": "payload_gain stays in bits and is the ranking target",
        "beta": "urgency is only the current queue ratio; the coefficient is not fit",
        "gamma": "distance/ETA equals the fixed step length, so it does not rank candidates",
        "delta": "the 15-step payload already includes SINR, so a second penalty would double count",
        "eta": "L3-B exposes no energy state",
    },
}


def utility_score(
    payload_gain: float,
    urgency: float,
    travel_cost: float,
    interference_cost: float,
    energy_cost: float,
    config: dict | None = None,
) -> float:
    cfg = BASELINE_CONFIG if config is None else config
    return float(
        float(cfg["alpha"]) * float(payload_gain)
        + float(cfg["beta"]) * float(urgency)
        - float(cfg["gamma"]) * float(travel_cost)
        - float(cfg["delta"]) * float(interference_cost)
        - float(cfg["eta"]) * float(energy_cost)
    )


def descending_rank(scores: list[float] | tuple[float, ...]) -> list[int]:
    """Rank 1 is the best score. Ties share the minimum rank."""
    values = [float(item) for item in scores]
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    ranks = [0] * len(values)
    for place, index in enumerate(order):
        if place > 0 and values[index] < values[order[place - 1]]:
            ranks[index] = place + 1
        elif place == 0:
            ranks[index] = 1
        else:
            ranks[index] = ranks[order[place - 1]]
    return ranks


__all__ = ["BASELINE_CONFIG", "descending_rank", "utility_score"]
