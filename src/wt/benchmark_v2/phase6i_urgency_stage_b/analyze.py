"""Paired summaries for the four task distributions. Urgency stays a diagnostic."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, POSITIVE_BIT

PRIMARY_CONTRASTS = (("B", "A"), ("C", "A"), ("D", "A"))
ALL_CONTRASTS = PRIMARY_CONTRASTS + (("C", "B"), ("D", "B"), ("D", "C"))
SCENARIO_METRICS = (
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
    "mean_sinr",
    "sinr_served_pearson",
)


def choose_gate(
    *,
    expected_episodes: int,
    complete_episodes: int,
    failures: int,
    log_mismatches: int,
    primary_payload_ci_excludes_zero: bool,
) -> str:
    if int(complete_episodes) != int(expected_episodes) or int(failures) > 0 or int(log_mismatches) > 0:
        return "CROSSTASK_BASELINE_BLOCKED"
    if not primary_payload_ci_excludes_zero:
        return "CROSSTASK_BASELINE_INCONCLUSIVE"
    return "CROSSTASK_BASELINE_COMPLETE"


def scenario_level(rows: list[dict]) -> dict[tuple[int, str], dict[str, float]]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault((int(row["scenario_id"]), str(row["condition_id"])), []).append(row)
    reduced = {}
    for key, items in grouped.items():
        reduced[key] = {metric: _mean_defined(items, metric) for metric in SCENARIO_METRICS}
        reduced[key]["replications"] = float(len(items))
    return reduced


def paired_contrast(level: dict[tuple[int, str], dict[str, float]], left: str, right: str, metric: str, scenario_ids: tuple[int, ...], rng: np.random.Generator) -> dict:
    deltas = []
    used = []
    for scenario_id in scenario_ids:
        left_row = level.get((int(scenario_id), left))
        right_row = level.get((int(scenario_id), right))
        if left_row is None or right_row is None:
            continue
        left_value = left_row[metric]
        right_value = right_row[metric]
        if left_value is None or right_value is None:
            continue
        deltas.append(float(left_value) - float(right_value))
        used.append(int(scenario_id))
    array = np.asarray(deltas, dtype=np.float64)
    if array.size == 0:
        return {"contrast": f"{left}-{right}", "metric": metric, "scenarios": 0}
    draws = rng.integers(0, int(array.size), size=(int(BOOTSTRAP_DRAWS), int(array.size)))
    means = array[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    positive = int(np.sum(array > float(POSITIVE_BIT))) if metric == "payload_bits" else int(np.sum(array > 0.0))
    negative = int(np.sum(array < -float(POSITIVE_BIT))) if metric == "payload_bits" else int(np.sum(array < 0.0))
    if metric != "payload_bits":
        positive = int(np.sum(array > 0.0))
        negative = int(np.sum(array < 0.0))
    tie_band = float(POSITIVE_BIT) if metric == "payload_bits" else 0.0
    ties = int(np.sum(np.abs(array) <= tie_band))
    worst = int(np.argmin(array))
    return {
        "contrast": f"{left}-{right}",
        "metric": metric,
        "scenarios": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci_excludes_zero": bool(float(low) > 0.0 or float(high) < 0.0),
        "positive_scenarios": positive,
        "negative_scenarios": negative,
        "tie_scenarios": ties,
        "worst_scenario_id": int(used[worst]),
        "worst_delta": float(array[worst]),
        "best_scenario_id": int(used[int(np.argmax(array))]),
        "best_delta": float(np.max(array)),
    }


def replication_variance(rows: list[dict], condition_id: str, metric: str = "payload_bits") -> dict:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row.get("status") != "ok" or str(row["condition_id"]) != str(condition_id):
            continue
        value = row.get(metric)
        if value is None or value == "":
            continue
        grouped.setdefault(int(row["scenario_id"]), []).append(float(value))
    variances = []
    widest_id = None
    widest = -1.0
    for scenario_id, values in grouped.items():
        if len(values) < 2:
            continue
        variance = float(np.var(np.asarray(values, dtype=np.float64), ddof=1))
        variances.append(variance)
        if variance > widest:
            widest = variance
            widest_id = int(scenario_id)
    if not variances:
        return {"condition_id": condition_id, "metric": metric, "scenarios": 0}
    return {
        "condition_id": condition_id,
        "metric": metric,
        "scenarios": int(len(variances)),
        "mean_variance": float(np.mean(variances)),
        "median_variance": float(np.median(variances)),
        "max_variance": float(np.max(variances)),
        "widest_scenario_id": widest_id,
    }


def condition_summary(level: dict[tuple[int, str], dict[str, float]], condition_id: str, scenario_ids: tuple[int, ...]) -> dict:
    rows = [level[(int(scenario_id), condition_id)] for scenario_id in scenario_ids if (int(scenario_id), condition_id) in level]
    out = {"condition_id": condition_id, "scenarios": int(len(rows))}
    for metric in ("payload_bits", "urgent_service_bits", "oldest_wait_mean", "oldest_wait_max", "positive_service_steps", "projector_trigger_rate", "in_radius_uav_steps", "path_length_m", "dropped_bits", "final_queue_bits", "high_urgency_served_bits", "mean_sinr"):
        values = np.asarray([float(row[metric]) for row in rows if row[metric] is not None], dtype=np.float64)
        if values.size == 0:
            continue
        present = [
            int(scenario_id)
            for scenario_id in scenario_ids
            if (int(scenario_id), condition_id) in level and level[(int(scenario_id), condition_id)][metric] is not None
        ]
        worst = min(present, key=lambda scenario_id: float(level[(int(scenario_id), condition_id)][metric]))
        out[metric] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "lowest_scenario_id": int(worst),
        }
    pearson = [row["sinr_served_pearson"] for row in rows if row["sinr_served_pearson"] is not None]
    out["sinr_served_pearson_defined"] = int(len(pearson))
    out["sinr_served_pearson_median"] = float(np.median(pearson)) if pearson else None
    return out


def write_figures(level: dict[tuple[int, str], dict[str, float]], scenario_ids: tuple[int, ...], path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    labels = ["A", "B", "C", "D"]
    panels = (
        ("payload_bits", "Payload (bit)"),
        ("oldest_wait_mean", "Mean oldest wait (steps)"),
        ("positive_service_steps", "Positive service steps"),
        ("projector_trigger_rate", "Projector trigger rate"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for axis, (metric, title) in zip(axes.ravel(), panels):
        means = []
        lows = []
        highs = []
        for label in labels:
            values = np.asarray(
                [level[(int(scenario_id), label)][metric] for scenario_id in scenario_ids if (int(scenario_id), label) in level],
                dtype=np.float64,
            )
            if values.size == 0:
                means.append(0.0)
                lows.append(0.0)
                highs.append(0.0)
                continue
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            draws = rng.integers(0, int(values.size), size=(int(BOOTSTRAP_DRAWS), int(values.size)))
            sampled = values[draws].mean(axis=1)
            low, high = np.percentile(sampled, [2.5, 97.5])
            mean = float(np.mean(values))
            means.append(mean)
            lows.append(mean - float(low))
            highs.append(float(high) - mean)
        axis.bar(labels, means, color=["#1f4e79", "#5b8e7d", "#c4a35a", "#8c4a3a"])
        axis.errorbar(labels, means, yerr=np.vstack([lows, highs]), fmt="none", ecolor="black", capsize=3)
        axis.set_title(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return True


def _mean_defined(items: list[dict], metric: str) -> float | None:
    values = []
    for item in items:
        value = item.get(metric)
        if value is None or value == "":
            continue
        values.append(float(value))
    if not values:
        return None
    return float(np.mean(values))
