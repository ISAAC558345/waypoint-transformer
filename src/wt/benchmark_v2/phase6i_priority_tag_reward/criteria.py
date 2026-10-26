"""Apply the preregistered merge criteria. This module does not retune thresholds."""
from __future__ import annotations

import numpy as np

from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CONSTRAINT_DIFF_MAX,
    DROPPED_DIFF_MAX,
    INTERFERENCE_DIFF_MAX,
    MIN_PASSING_SEEDS,
    PAYLOAD_CI_LOW_MIN,
    PAYLOAD_MEAN_MIN,
    PAYLOAD_METRIC,
    SAFETY_DIFF_MAX,
    SINR_DIFF_MIN,
    TAGGED_CI_LOW_MIN,
    TAGGED_METRIC,
    TRAIN_SEEDS,
    URGENCY_CI_LOW_MIN,
    URGENCY_METRIC,
)


def paired_interval(diff: np.ndarray) -> dict[str, float]:
    values = np.asarray(diff, dtype=np.float64).reshape(-1)
    if int(values.size) < 2:
        raise RuntimeError("paired interval needs more than one scenario")
    generator = np.random.default_rng(int(BOOTSTRAP_SEED))
    draws = np.empty(int(BOOTSTRAP_DRAWS), dtype=np.float64)
    for index in range(int(BOOTSTRAP_DRAWS)):
        draws[index] = float(np.mean(values[generator.integers(0, int(values.size), int(values.size))]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(np.mean(values)), "ci_low": float(low), "ci_high": float(high), "scenarios": int(values.size)}


def _column(rows: list[dict], variant: str, seed: int, metric: str) -> np.ndarray:
    selected = [row for row in rows if row["variant"] == variant and int(row["model_seed"]) == int(seed)]
    selected.sort(key=lambda row: int(row["scenario_id"]))
    return np.asarray([float(row[metric]) for row in selected], dtype=np.float64)


def _baseline(rows: list[dict], metric: str) -> np.ndarray:
    return _column(rows, "L0", 0, metric)


def leave_one_out(diff: np.ndarray, *, floor: float, strict: bool) -> bool:
    values = np.asarray(diff, dtype=np.float64)
    for index in range(int(values.size)):
        value = float(np.mean(np.delete(values, index)))
        if value < float(floor) or (strict and value <= float(floor)):
            return False
    return True


def _seed_report(rows: list[dict], variant: str, seed: int) -> dict:
    urgency = _column(rows, variant, seed, URGENCY_METRIC) - _baseline(rows, URGENCY_METRIC)
    tagged = _column(rows, variant, seed, TAGGED_METRIC) - _baseline(rows, TAGGED_METRIC)
    payload = _column(rows, variant, seed, PAYLOAD_METRIC) - _baseline(rows, PAYLOAD_METRIC)
    safety = _column(rows, variant, seed, "safety_violations") - _baseline(rows, "safety_violations")
    constraints = _column(rows, variant, seed, "constraint_failures") - _baseline(rows, "constraint_failures")
    dropped = _column(rows, variant, seed, "dropped_bits") - _baseline(rows, "dropped_bits")
    sinr = _column(rows, variant, seed, "sinr_mean") - _baseline(rows, "sinr_mean")
    interference = _column(rows, variant, seed, "interference_mean") - _baseline(rows, "interference_mean")
    urgency_ci = paired_interval(urgency)
    tagged_ci = paired_interval(tagged)
    payload_ci = paired_interval(payload)
    payload_regressions = int(np.sum(payload < 0.0))
    passes = bool(
        urgency_ci["ci_low"] > float(URGENCY_CI_LOW_MIN)
        and tagged_ci["ci_low"] > float(TAGGED_CI_LOW_MIN)
        and payload_ci["mean"] >= float(PAYLOAD_MEAN_MIN)
        and payload_ci["ci_low"] >= float(PAYLOAD_CI_LOW_MIN)
        and float(np.sum(safety)) <= float(SAFETY_DIFF_MAX)
        and float(np.sum(constraints)) <= float(CONSTRAINT_DIFF_MAX)
        and float(np.mean(dropped)) <= float(DROPPED_DIFF_MAX)
        and float(np.mean(sinr)) >= float(SINR_DIFF_MIN)
        and float(np.mean(interference)) <= float(INTERFERENCE_DIFF_MAX)
        and leave_one_out(urgency, floor=0.0, strict=True)
        and leave_one_out(payload, floor=0.0, strict=False)
    )
    return {
        "variant": variant,
        "model_seed": int(seed),
        "passes_preregistered_criteria": passes,
        "urgency": urgency_ci,
        "tagged": tagged_ci,
        "payload": payload_ci,
        "payload_negative_scenarios": payload_regressions,
        "safety_diff_sum": float(np.sum(safety)),
        "constraint_diff_sum": float(np.sum(constraints)),
        "dropped_diff_mean": float(np.mean(dropped)),
        "sinr_diff_mean": float(np.mean(sinr)),
        "interference_diff_mean": float(np.mean(interference)),
        "urgency_not_single_scenario": bool(leave_one_out(urgency, floor=0.0, strict=True)),
        "payload_not_single_scenario": bool(leave_one_out(payload, floor=0.0, strict=False)),
    }


def merge_decision(rows: list[dict]) -> dict:
    l2 = [_seed_report(rows, "L2", int(seed)) for seed in TRAIN_SEEDS]
    l1 = [_seed_report(rows, "L1", int(seed)) for seed in TRAIN_SEEDS]
    l2_pass = int(sum(bool(item["passes_preregistered_criteria"]) for item in l2))
    l1_pass = int(sum(bool(item["passes_preregistered_criteria"]) for item in l1))
    return {
        "l2_seeds": l2,
        "l1_seeds": l1,
        "l2_passing_seeds": l2_pass,
        "l1_passing_seeds": l1_pass,
        "merge_l2": bool(l2_pass >= int(MIN_PASSING_SEEDS)),
        "l1_meets_same_criteria": bool(l1_pass >= int(MIN_PASSING_SEEDS)),
        "thresholds": {
            "min_passing_seeds": int(MIN_PASSING_SEEDS),
            "urgency_ci_low_min": float(URGENCY_CI_LOW_MIN),
            "tagged_ci_low_min": float(TAGGED_CI_LOW_MIN),
            "payload_mean_min": float(PAYLOAD_MEAN_MIN),
            "payload_ci_low_min": float(PAYLOAD_CI_LOW_MIN),
            "safety_diff_max": float(SAFETY_DIFF_MAX),
            "constraint_diff_max": float(CONSTRAINT_DIFF_MAX),
            "dropped_diff_max": float(DROPPED_DIFF_MAX),
            "sinr_diff_min": float(SINR_DIFF_MIN),
            "interference_diff_max": float(INTERFERENCE_DIFF_MAX),
        },
    }
