"""Offline comparison of two 48-step traces. This module does not run the environment."""
from __future__ import annotations

import numpy as np

from .protocol import ACCOUNTING_TOLERANCE_BITS, GATE_BLOCKED, GATE_INCONCLUSIVE, GATE_LOCALIZED, PATH_EQUAL_METERS


DIFF_FIELDS = (
    "mu",
    "desired",
    "projected",
    "executed",
    "position",
    "q_bits",
    "served",
    "dropped",
    "snap",
    "payload",
    "path_increment",
)
ACTION_FIELDS = {"mu", "desired", "projected", "executed"}
QUEUE_FIELDS = {"q_bits", "served", "dropped", "snap", "payload"}


def _max_abs(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def step_diffs(left: dict, right: dict) -> dict[str, float]:
    return {field: _max_abs(left[field], right[field]) for field in DIFF_FIELDS}


def first_difference(left_rows: list[dict], right_rows: list[dict]) -> dict | None:
    if len(left_rows) != len(right_rows):
        return {"step_index": None, "reason": "trace lengths differ", "diffs": {}}
    for index, (left, right) in enumerate(zip(left_rows, right_rows)):
        diffs = step_diffs(left, right)
        nonzero = {key: value for key, value in diffs.items() if value > 0.0}
        if nonzero or left.get("obs_sha256") != right.get("obs_sha256"):
            if left.get("obs_sha256") != right.get("obs_sha256"):
                nonzero["observation"] = 1.0
            field = next(name for name in ("observation",) + DIFF_FIELDS if name in nonzero)
            return {"step_index": int(index), "field": field, "diffs": diffs, "nonzero": nonzero}
    return None


def episode_bits(rows: list[dict]) -> dict[str, float]:
    last = rows[-1]
    return {
        "payload_bits": float(last["payload"]),
        "served_bits": float(sum(float(row["served"]) for row in rows)),
        "dropped_bits": float(sum(float(row["dropped"]) for row in rows)),
        "snap_bits": float(sum(float(row["snap"]) for row in rows)),
        "path_length_m": float(sum(float(row["path_increment"]) for row in rows)),
    }


def saved_gaps(episode: dict, saved: dict) -> dict[str, float]:
    return {
        "payload_bits": abs(float(episode["payload_bits"]) - float(saved["payload_bits"])),
        "served_ordinary_bits": abs(float(episode["served_ordinary_bits"]) - float(saved["served_ordinary_bits"])),
        "served_tagged_bits": abs(float(episode["served_tagged_bits"]) - float(saved["served_tagged_bits"])),
        "dropped_ordinary_bits": abs(float(episode["dropped_ordinary_bits"]) - float(saved["dropped_ordinary_bits"])),
        "dropped_tagged_bits": abs(float(episode["dropped_tagged_bits"]) - float(saved["dropped_tagged_bits"])),
        "urgency_served_bits": abs(float(episode["urgency_served_bits"]) - float(saved["urgency_served_bits"])),
        "path_length_m": abs(float(episode["path_length_m"]) - float(saved["path_length_m"])),
    }


def bits_within_tolerance(gaps: dict) -> bool:
    keys = (
        "payload_bits",
        "served_ordinary_bits",
        "served_tagged_bits",
        "dropped_ordinary_bits",
        "dropped_tagged_bits",
        "urgency_served_bits",
    )
    return all(float(gaps[key]) <= float(ACCOUNTING_TOLERANCE_BITS) for key in keys)


def path_matches(gaps: dict) -> bool:
    return float(gaps["path_length_m"]) <= float(PATH_EQUAL_METERS)


def cause_letter(
    *,
    hashes_ok: bool,
    split_ok: bool,
    reset_equal: bool,
    inherited_matches: bool,
    first_field: str | None,
) -> str:
    if not hashes_ok or not split_ok:
        return "F"
    if not reset_equal:
        return "B"
    if inherited_matches and first_field in ACTION_FIELDS:
        return "A"
    if inherited_matches and first_field in QUEUE_FIELDS:
        return "D"
    return "F"


def choose_gate(
    *,
    hashes_ok: bool,
    split_ok: bool,
    instrumented_same: bool,
    inherited_matches: bool,
    first_step_found: bool,
    fresh_matches: bool,
) -> str:
    if not hashes_ok or not split_ok:
        return GATE_BLOCKED
    if fresh_matches:
        return GATE_INCONCLUSIVE
    if instrumented_same and inherited_matches and first_step_found:
        return GATE_LOCALIZED
    return GATE_INCONCLUSIVE
