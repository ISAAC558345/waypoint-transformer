"""Preregistered performance gate. Safety increases fail the comparison."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.criteria import leave_two_largest
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


def _pair(current: list[dict], baseline: list[dict], metric: str) -> np.ndarray:
    if [int(row["scenario_id"]) for row in current] != [int(row["scenario_id"]) for row in baseline]:
        raise RuntimeError("paired scenarios are not aligned")
    return np.asarray([float(row[metric]) for row in current], dtype=np.float64) - np.asarray(
        [float(row[metric]) for row in baseline], dtype=np.float64
    )


def _reference_checks(current: list[dict], reference: list[dict], reference_name: str) -> dict:
    tagged = _pair(current, reference, "served_tagged_bits")
    payload = _pair(current, reference, "payload_bits")
    sinr = _pair(current, reference, "sinr_mean")
    interference = _pair(current, reference, "interference_mean")
    safety = _pair(current, reference, "safety_violations")
    constraints = _pair(current, reference, "constraint_failures")
    payload_mean = float(np.mean([float(row["payload_bits"]) for row in reference]))
    floor = -float(PAYLOAD_CI_FLOOR_FRACTION) * payload_mean
    return {
        "reference": reference_name,
        "tagged": paired_interval(tagged),
        "payload": paired_interval(payload),
        "payload_ci_floor": floor,
        "reference_payload_mean": payload_mean,
        "safety_diff_sum": float(np.sum(safety)),
        "constraint_diff_sum": float(np.sum(constraints)),
        "sinr_mean": float(np.mean(sinr)),
        "interference_mean": float(np.mean(interference)),
        "leave_one_tagged": bool(leave_one_out(tagged, floor=0.0, strict=True)),
        "leave_two_tagged": bool(leave_two_largest(tagged)),
        "payload_negative_scenarios": [int(current[index]["scenario_id"]) for index, value in enumerate(payload) if float(value) < 0.0],
        "tagged_negative_scenarios": [int(current[index]["scenario_id"]) for index, value in enumerate(tagged) if float(value) < 0.0],
    }


def _passes(versus_j: dict, versus_l0: dict) -> bool:
    return bool(
        float(versus_j["tagged"]["ci_low"]) > 0.0
        and float(versus_l0["tagged"]["ci_low"]) > 0.0
        and float(versus_j["payload"]["mean"]) >= 0.0
        and float(versus_l0["payload"]["mean"]) >= 0.0
        and float(versus_j["payload"]["ci_low"]) >= float(versus_j["payload_ci_floor"])
        and float(versus_l0["payload"]["ci_low"]) >= float(versus_l0["payload_ci_floor"])
        and float(versus_j["sinr_mean"]) >= 0.0
        and float(versus_l0["sinr_mean"]) >= 0.0
        and float(versus_j["interference_mean"]) <= 0.0
        and float(versus_l0["interference_mean"]) <= 0.0
        and versus_j["leave_one_tagged"]
        and versus_l0["leave_one_tagged"]
        and versus_j["leave_two_tagged"]
        and versus_l0["leave_two_tagged"]
    )


def score_full(rows: list[dict]) -> dict:
    l0 = _rows(rows, "L0", 0)
    seeds = []
    for seed in TRAIN_SEEDS:
        current = _rows(rows, "FULL_S", int(seed))
        control = _rows(rows, "FULL_J", int(seed))
        if len(current) != len(l0) or len(control) != len(l0) or len(l0) == 0:
            raise RuntimeError(f"FULL_S seed {seed} is not paired with FULL_J and L0")
        versus_j = _reference_checks(current, control, "FULL_J")
        versus_l0 = _reference_checks(current, l0, "L0")
        seeds.append(
            {
                "model_seed": int(seed),
                "passes_preregistered_criteria": _passes(versus_j, versus_l0),
                "versus_full_j": versus_j,
                "versus_l0": versus_l0,
            }
        )
    passing = int(sum(bool(item["passes_preregistered_criteria"]) for item in seeds))
    return {"seeds": seeds, "passing_seeds": passing}


def choose_gate(*, tests_ok: bool, train_ok: bool, hashes_ok: bool, preflight_ok: bool, score: dict | None) -> str:
    if not tests_ok or not train_ok or not hashes_ok or not preflight_ok or score is None:
        return GATE_BLOCKED
    seeds = score["seeds"]
    safety_increase = any(
        float(item["versus_l0"]["safety_diff_sum"]) > 0.0
        or float(item["versus_l0"]["constraint_diff_sum"]) > 0.0
        or float(item["versus_full_j"]["safety_diff_sum"]) > 0.0
        or float(item["versus_full_j"]["constraint_diff_sum"]) > 0.0
        for item in seeds
    )
    negative_tagged = sum(float(item["versus_l0"]["tagged"]["mean"]) < 0.0 for item in seeds)
    negative_payload = sum(float(item["versus_l0"]["payload"]["mean"]) < 0.0 for item in seeds)
    if safety_increase or negative_tagged >= 2 or negative_payload >= 2:
        return GATE_FAIL
    if int(score["passing_seeds"]) >= int(MIN_PASSING_SEEDS):
        return GATE_PASS
    return GATE_INCONCLUSIVE
