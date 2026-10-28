"""Apply the section-6 merge rule. This module does not retune thresholds."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag_reward.criteria import leave_one_out, paired_interval
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import BOOTSTRAP_DRAWS as SOURCE_DRAWS
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import BOOTSTRAP_SEED as SOURCE_SEED

from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CONSTRAINT_DIFF_MAX,
    DROPPED_DIFF_MAX,
    GATE_BLOCKED,
    GATE_INCONCLUSIVE,
    GATE_PASS,
    INTERFERENCE_DIFF_MAX,
    MIN_PASSING_SEEDS,
    PAYLOAD_CI_LOW_MIN,
    PAYLOAD_MEAN_MIN,
    QUEUE_DIFF_MAX,
    SAFETY_DIFF_MAX,
    SINR_DIFF_MIN,
    TAGGED_CI_LOW_MIN,
    TAGGED_MEAN_MIN,
    TRAIN_SEEDS,
)

if int(SOURCE_DRAWS) != int(BOOTSTRAP_DRAWS) or int(SOURCE_SEED) != int(BOOTSTRAP_SEED):
    raise RuntimeError("bootstrap draw count or seed drifted from the preregistered interval")


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


def _negative_scenarios(current: list[dict], baseline: list[dict], metric: str) -> list[int]:
    diff = _pair(current, baseline, metric)
    return [int(current[index]["scenario_id"]) for index, value in enumerate(diff) if float(value) < 0.0]


def score_l3(rows: list[dict]) -> dict:
    baseline = _rows(rows, "L0", 0)
    seeds = []
    for seed in TRAIN_SEEDS:
        current = _rows(rows, "L3", int(seed))
        reference = _rows(rows, "L1", int(seed))
        if len(current) != len(baseline) or len(reference) != len(baseline):
            raise RuntimeError(f"L3 seed {seed} does not match the paired scenario count")
        tagged = _pair(current, baseline, "served_tagged_bits")
        payload = _pair(current, baseline, "payload_bits")
        dropped = _pair(current, baseline, "dropped_bits")
        sinr = _pair(current, baseline, "sinr_mean")
        interference = _pair(current, baseline, "interference_mean")
        queue = _pair(current, baseline, "final_queue_bits")
        safety = _pair(current, baseline, "safety_violations")
        constraints = _pair(current, baseline, "constraint_failures")
        versus_l1 = _pair(current, reference, "served_tagged_bits")
        tagged_ci = paired_interval(tagged)
        payload_ci = paired_interval(payload)
        versus_l1_ci = paired_interval(versus_l1)
        not_single = bool(leave_one_out(tagged, floor=0.0, strict=True))
        passes = bool(
            float(tagged_ci["mean"]) > float(TAGGED_MEAN_MIN)
            and float(tagged_ci["ci_low"]) > float(TAGGED_CI_LOW_MIN)
            and float(payload_ci["mean"]) >= float(PAYLOAD_MEAN_MIN)
            and float(payload_ci["ci_low"]) >= float(PAYLOAD_CI_LOW_MIN)
            and float(np.sum(safety)) <= float(SAFETY_DIFF_MAX)
            and float(np.sum(constraints)) <= float(CONSTRAINT_DIFF_MAX)
            and float(np.mean(dropped)) <= float(DROPPED_DIFF_MAX)
            and float(np.mean(sinr)) >= float(SINR_DIFF_MIN)
            and float(np.mean(interference)) <= float(INTERFERENCE_DIFF_MAX)
            and float(np.mean(queue)) <= float(QUEUE_DIFF_MAX)
            and not_single
            and float(versus_l1_ci["mean"]) > float(TAGGED_MEAN_MIN)
            and float(versus_l1_ci["ci_low"]) > float(TAGGED_CI_LOW_MIN)
        )
        seeds.append(
            {
                "variant": "L3",
                "model_seed": int(seed),
                "passes_preregistered_criteria": passes,
                "tagged_versus_l0": tagged_ci,
                "payload_versus_l0": payload_ci,
                "dropped_versus_l0_mean": float(np.mean(dropped)),
                "sinr_versus_l0_mean": float(np.mean(sinr)),
                "interference_versus_l0_mean": float(np.mean(interference)),
                "queue_versus_l0_mean": float(np.mean(queue)),
                "safety_diff_sum": float(np.sum(safety)),
                "constraint_diff_sum": float(np.sum(constraints)),
                "payload_negative_scenarios": _negative_scenarios(current, baseline, "payload_bits"),
                "tagged_not_single_scenario": not_single,
                "tagged_versus_l1": versus_l1_ci,
                "age_versus_l0": paired_interval(_pair(current, baseline, "age_service_bits")),
                "tag_increment_versus_l0": paired_interval(_pair(current, baseline, "tag_increment_service_bits")),
                "tagged_versus_l2": paired_interval(_pair(current, _rows(rows, "L2", int(seed)), "served_tagged_bits")),
            }
        )
    passing = int(sum(bool(item["passes_preregistered_criteria"]) for item in seeds))
    return {"seeds": seeds, "passing_seeds": passing, "merge": bool(passing >= int(MIN_PASSING_SEEDS))}


def choose_gate(*, smoke_ok: bool, train_ok: bool, replay_ok: bool, hashes_ok: bool, passing_seeds: int) -> str:
    if not smoke_ok or not train_ok or not replay_ok or not hashes_ok:
        return GATE_BLOCKED
    if int(passing_seeds) >= int(MIN_PASSING_SEEDS):
        return GATE_PASS
    return GATE_INCONCLUSIVE
