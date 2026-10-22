"""Independent closed-loop evaluation. Reuses Phase-10.3 episode metrics and gate."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase103.evaluate import episode_metrics, run_episode
from wt.round30.ppo import PPOAgent


def extra_times(events: list[dict[str, Any]]) -> dict[str, Any]:
    first_complete: dict[int, int] = {}
    first_in_range: dict[int, int] = {}
    for event in events:
        uav = int(event["uav"])
        t = int(event["t_before"])
        if event.get("completed_block") and uav not in first_complete:
            first_complete[uav] = t
        if event.get("in_range_after") and (not event.get("idle")) and uav not in first_in_range:
            first_in_range[uav] = t
    complete_vals = list(first_complete.values())
    return {
        "mean_first_complete_t_success_only": float(np.mean(complete_vals)) if complete_vals else None,
        "n_uav_with_first_complete": int(len(complete_vals)),
        "first_complete_t": [first_complete.get(u) for u in range(6)],
        "first_in_range_t": [first_in_range.get(u) for u in range(6)],
        "note_times": "None means the event never occurred; not filled with 0",
    }


def evaluate_family(
    high_model,
    seed: int,
    *,
    family: str,
    schema: dict[str, Any] | None,
    agent: PPOAgent | None,
    gated: bool,
    low_seed: int | None,
) -> dict[str, Any]:
    result = run_episode(high_model, int(seed), family=family, schema=schema, agent=agent, gated=gated)
    metrics = episode_metrics(result["events"], family, int(seed))
    metrics.update(extra_times(result["events"]))
    metrics["low_seed"] = None if low_seed is None else int(low_seed)
    metrics["independent_validation"] = True
    metrics["not_untouched"] = True
    return metrics


__all__ = ["evaluate_family", "extra_times"]
