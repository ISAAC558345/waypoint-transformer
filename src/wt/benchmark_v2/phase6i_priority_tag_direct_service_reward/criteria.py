"""Section-6 merge rule. Thresholds are not revised after the benchmark."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag_reward.criteria import leave_one_out, paired_interval

from .protocol import (
    GATE_BLOCKED,
    GATE_FAIL,
    GATE_INCONCLUSIVE,
    GATE_PASS,
    MIN_PASSING_SEEDS,
    PAYLOAD_CI_FLOOR_FRACTION,
    TRAIN_SEEDS,
)


def _rows(rows: list[dict], variant: str, seed: int) -> list[dict]:
    selected = [row for row in rows if row["variant"] == variant and int(row["model_seed"]) == int(seed)]
    selected.sort(key=lambda row: int(row["scenario_id"]))
    return selected


def _values(rows: list[dict], metric: str) -> np.ndarray:
    return np.asarray([float(row[metric]) for row in rows], dtype=np.float64)


def _pair(current: list[dict], baseline: list[dict], metric: str) -> np.ndarray:
    if [int(row["scenario_id"]) for row in current] != [int(row["scenario_id"]) for row in baseline]:
        raise RuntimeError("paired scenarios are not aligned")
    return _values(current, metric) - _values(baseline, metric)


def leave_two_largest(diff: np.ndarray) -> bool:
    """True when the mean stays positive after removing the two largest scenario gains."""
    values = np.asarray(diff, dtype=np.float64).reshape(-1)
    if int(values.size) < 3:
        return False
    order = np.argsort(values)
    trimmed = np.delete(values, order[-2:])
    return bool(float(np.mean(trimmed)) > 0.0)


def score_l3(rows: list[dict]) -> dict:
    baseline = _rows(rows, "L0", 0)
    l0_payload_mean = float(np.mean(_values(baseline, "payload_bits")))
    payload_floor = -float(PAYLOAD_CI_FLOOR_FRACTION) * l0_payload_mean
    seeds = []
    for seed in TRAIN_SEEDS:
        current = _rows(rows, "L3", int(seed))
        reference = _rows(rows, "L2", int(seed))
        if len(current) != len(baseline) or len(reference) != len(baseline):
            raise RuntimeError(f"L3 seed {seed} does not match the paired scenario count")
        tagged = _pair(current, baseline, "served_tagged_bits")
        payload = _pair(current, baseline, "payload_bits")
        safety = _pair(current, baseline, "safety_violations")
        constraints = _pair(current, baseline, "constraint_failures")
        sinr = _pair(current, baseline, "sinr_mean")
        interference = _pair(current, baseline, "interference_mean")
        tagged_ci = paired_interval(tagged)
        payload_ci = paired_interval(payload)
        not_one = bool(leave_one_out(tagged, floor=0.0, strict=True))
        not_two = bool(leave_two_largest(tagged))
        safety_sum = float(np.sum(safety))
        constraint_sum = float(np.sum(constraints))
        passes = bool(
            float(tagged_ci["ci_low"]) > 0.0
            and float(payload_ci["mean"]) >= 0.0
            and float(payload_ci["ci_low"]) >= payload_floor
            and safety_sum <= 0.0
            and constraint_sum <= 0.0
            and float(np.mean(sinr)) >= 0.0
            and float(np.mean(interference)) <= 0.0
            and not_one
            and not_two
        )
        negative = [int(current[index]["scenario_id"]) for index, value in enumerate(payload) if float(value) < 0.0]
        seeds.append(
            {
                "variant": "L3",
                "model_seed": int(seed),
                "passes_preregistered_criteria": passes,
                "tagged_versus_l0": tagged_ci,
                "payload_versus_l0": payload_ci,
                "payload_ci_floor": payload_floor,
                "l0_payload_mean": l0_payload_mean,
                "safety_diff_sum": safety_sum,
                "constraint_diff_sum": constraint_sum,
                "sinr_versus_l0_mean": float(np.mean(sinr)),
                "interference_versus_l0_mean": float(np.mean(interference)),
                "payload_negative_scenarios": negative,
                "tagged_not_single_scenario": not_one,
                "tagged_not_two_scenarios": not_two,
                "age_versus_l0": paired_interval(_pair(current, baseline, "age_service_bits")),
                "tagged_versus_l2": paired_interval(_pair(current, reference, "served_tagged_bits")),
                "payload_versus_l2": paired_interval(_pair(current, reference, "payload_bits")),
                "ordinary_versus_l0": paired_interval(_pair(current, baseline, "served_ordinary_bits")),
                "dropped_versus_l0": paired_interval(_pair(current, baseline, "dropped_bits")),
                "queue_versus_l0_mean": float(np.mean(_pair(current, baseline, "final_queue_bits"))),
            }
        )
    passing = int(sum(bool(item["passes_preregistered_criteria"]) for item in seeds))
    return {
        "seeds": seeds,
        "passing_seeds": passing,
        "merge": bool(passing >= int(MIN_PASSING_SEEDS)),
        "payload_ci_floor": payload_floor,
    }


def publish_gate(scientific: str, isolated: bool) -> str:
    """A non-isolated branch cannot be published as a benchmark pass."""
    if scientific == GATE_PASS and not bool(isolated):
        return GATE_INCONCLUSIVE
    return scientific


def choose_gate(*, smoke_ok: bool, train_ok: bool, replay_ok: bool, hashes_ok: bool, score: dict | None) -> str:
    if not smoke_ok or not train_ok or not replay_ok or not hashes_ok or score is None:
        return GATE_BLOCKED
    seeds = score["seeds"]
    safety_fail = any(float(item["safety_diff_sum"]) > 0.0 or float(item["constraint_diff_sum"]) > 0.0 for item in seeds)
    negative_tagged = sum(float(item["tagged_versus_l0"]["mean"]) < 0.0 for item in seeds)
    negative_payload = sum(float(item["payload_versus_l0"]["mean"]) < 0.0 for item in seeds)
    if safety_fail or negative_tagged >= 2 or negative_payload >= 2:
        return GATE_FAIL
    if int(score["passing_seeds"]) >= int(MIN_PASSING_SEEDS):
        return GATE_PASS
    return GATE_INCONCLUSIVE
