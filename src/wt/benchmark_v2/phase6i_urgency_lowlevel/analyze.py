"""Paired L0, L1, and L2 summaries. The retention rule is fixed before the runs."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import POSITIVE_BIT
from wt.benchmark_v2.phase6i_urgency_stage_b.analyze import paired_contrast

CLOSED_METRICS = (
    "payload_bits",
    "urgent_service_bits",
    "high_urgency_served_bits",
    "urgency_at_least_half_served_bits",
    "oldest_wait_mean",
    "oldest_wait_max",
    "mean_wait_mean",
    "positive_service_steps",
    "in_radius_uav_steps",
    "path_length_m",
    "projector_trigger_rate",
    "safety_violation_steps",
    "dropped_bits",
    "final_queue_bits",
    "first_service_radius_step",
    "first_high_urgency_service_step",
    "age_weighted_backlog_mean",
    "mean_sinr",
    "mean_total_interference",
    "sinr_served_pearson",
    "mean_idle_slots",
)


def choose_gate(*, smoke_ok: bool, training_complete: bool, eval_complete: bool, failures: int, inconclusive: bool) -> str:
    if not smoke_ok or not training_complete or not eval_complete or int(failures) > 0:
        return "URGENCY_LOWLEVEL_BLOCKED"
    if inconclusive:
        return "URGENCY_LOWLEVEL_INCONCLUSIVE"
    return "URGENCY_LOWLEVEL_EVAL_COMPLETE"


def retain_l2(payload_l2_minus_l1: dict, urgent_l2_minus_l1: dict) -> str:
    """Keep L2 only when urgent service rises and payload is not significantly worse than L1.

    A gain in urgent service together with a payload loss is reported as a tradeoff.
    The one-bit payload tie band is the formal accounting tolerance.
    """
    urgent_up = float(urgent_l2_minus_l1.get("ci95_low", 0.0)) > 0.0
    payload_down = float(payload_l2_minus_l1.get("ci95_high", 0.0)) < -float(POSITIVE_BIT)
    if urgent_up and payload_down:
        return "TRADEOFF"
    if urgent_up and not payload_down:
        return "RETAIN"
    return "DO_NOT_RETAIN"


def condition_level(rows: list[dict], protocol: str) -> dict[tuple[int, str], dict[str, float]]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        if row.get("status") != "ok" or str(row.get("protocol")) != protocol:
            continue
        grouped.setdefault((int(row["scenario_id"]), str(row["variant"])), []).append(row)
    level = {}
    for key, items in grouped.items():
        level[key] = {metric: _mean_defined(items, metric) for metric in CLOSED_METRICS}
        level[key]["replications"] = float(len(items))
    return level


def seed_payload_spread(rows: list[dict], variant: str, protocol: str = "closed48") -> dict:
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        if row.get("status") != "ok" or str(row.get("protocol")) != protocol or str(row.get("variant")) != variant:
            continue
        by_seed.setdefault(int(row["model_seed"]), []).append(float(row["payload_bits"]))
    means = {seed: float(np.mean(values)) for seed, values in by_seed.items() if values}
    if len(means) < 2:
        return {"variant": variant, "seeds": len(means), "range_bits": 0.0}
    ordered = list(means.values())
    return {
        "variant": variant,
        "seeds": {str(seed): value for seed, value in means.items()},
        "range_bits": float(max(ordered) - min(ordered)),
        "std_bits": float(np.std(ordered)),
    }


def _mean_defined(items: list[dict], metric: str):
    if metric == "projector_trigger_rate":
        values = []
        for item in items:
            steps = max(int(item.get("final_t", 0)), 1)
            values.append(float(item["projector_trigger_steps"]) / float(steps))
        return float(np.mean(values))
    values = []
    for item in items:
        value = item.get(metric)
        if value is None or value == "":
            continue
        values.append(float(value))
    if not values:
        return None
    return float(np.mean(values))


def contrast(level, left: str, right: str, metric: str, scenario_ids, rng) -> dict:
    return paired_contrast(level, left, right, metric, scenario_ids, rng)
