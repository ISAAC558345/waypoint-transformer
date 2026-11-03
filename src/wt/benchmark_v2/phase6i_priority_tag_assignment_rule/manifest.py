"""Outer record of a finished selection. The scorer never receives a scenario id."""
from __future__ import annotations

from .protocol import DECISION_TIME, HOLD_STEPS
from .rule import choice_sha256


def freeze_record(result: dict, scenario_id: int, delta_t: float) -> dict:
    scores = result["eligible"]
    return {
        "scenario_id": int(scenario_id),
        "decision_time": int(DECISION_TIME),
        "horizon": int(HOLD_STEPS),
        "delta_t": repr(float(delta_t)),
        "weight": float(result["weight"]),
        "pool_sha256": result["pool_sha256"],
        "pool_size": int(result["pool_size"]),
        "unique_count": int(result["unique_count"]),
        "eligible_count": int(result["eligible_count"]),
        "baseline_pool_index": int(result["baseline_pool_index"]),
        "baseline_assignment": list(result["baseline_assignment"]),
        "baseline_mask": list(result["baseline_mask"]),
        "natural": result["natural"],
        "masked": result["masked"],
        "tagged": result["tagged"],
        "tagged_differs_from_masked": bool(result["tagged_differs_from_masked"]),
        "scores_sha256": choice_sha256(scores),
    }


def selection_sha256(records: list[dict]) -> str:
    return choice_sha256(records)
