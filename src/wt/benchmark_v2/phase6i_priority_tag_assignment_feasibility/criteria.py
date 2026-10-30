"""Descriptive labels for the registered candidate cap. They are not a performance pass."""
from __future__ import annotations

from .protocol import (
    KNOWN_TAG_GAIN_BITS,
    OPPORTUNITY_ABSENT,
    OPPORTUNITY_LOCAL,
    OPPORTUNITY_REPLICATED,
    OPPORTUNITY_THIN,
    PAYLOAD_FLOOR_BITS,
)


def service_levels(known_tagged_delta: float, payload_delta: float, safety_delta: float, constraint_delta: float, sinr_delta: float, interference_delta: float) -> dict[str, bool]:
    s0 = float(known_tagged_delta) > float(KNOWN_TAG_GAIN_BITS) and float(payload_delta) >= float(PAYLOAD_FLOOR_BITS)
    s1 = bool(s0 and float(safety_delta) <= 0.0 and float(constraint_delta) <= 0.0)
    s2 = bool(s1 and float(sinr_delta) >= 0.0 and float(interference_delta) <= 0.0)
    return {"s0": bool(s0), "s1": s1, "s2": s2}


def pareto_slots(rows: list[dict]) -> list[int]:
    """Non-dominated slots inside one snapshot. Higher service and SINR are better."""
    keep = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            better_or_equal = (
                float(other["known_tagged_delta_bits"]) >= float(row["known_tagged_delta_bits"])
                and float(other["payload_delta_bits"]) >= float(row["payload_delta_bits"])
                and float(other["safety_delta"]) <= float(row["safety_delta"])
                and float(other["constraint_delta"]) <= float(row["constraint_delta"])
                and float(other["mean_sinr_delta"]) >= float(row["mean_sinr_delta"])
                and float(other["mean_interference_delta"]) <= float(row["mean_interference_delta"])
            )
            strict = (
                float(other["known_tagged_delta_bits"]) > float(row["known_tagged_delta_bits"])
                or float(other["payload_delta_bits"]) > float(row["payload_delta_bits"])
                or float(other["safety_delta"]) < float(row["safety_delta"])
                or float(other["constraint_delta"]) < float(row["constraint_delta"])
                or float(other["mean_sinr_delta"]) > float(row["mean_sinr_delta"])
                or float(other["mean_interference_delta"]) < float(row["mean_interference_delta"])
            )
            if better_or_equal and strict:
                dominated = True
                break
        if not dominated:
            keep.append(int(row["candidate_slot"]))
    return keep


def posthoc_best(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["known_tagged_delta_bits"]),
            float(row["payload_delta_bits"]),
            -float(row["safety_delta"]),
            -float(row["constraint_delta"]),
            -int(row["candidate_slot"]),
        ),
        reverse=True,
    )
    chosen = ordered[0]
    return {
        "candidate_slot": int(chosen["candidate_slot"]),
        "known_tagged_delta_bits": float(chosen["known_tagged_delta_bits"]),
        "hindsight_diagnostic": True,
        "deployable_policy": False,
    }


def opportunity_evidence(rows: list[dict], *, expected_scenarios: int, expected_snapshots: int) -> str:
    """Label only interpretable rows. Partial coverage cannot become a replicated claim."""
    valid = [row for row in rows if row.get("interpretable")]
    valid_snapshots = {(int(row["scenario_id"]), int(row["decision_time"])) for row in valid}
    valid_scenarios = {scenario for scenario, _time in valid_snapshots}
    s0_scenarios = {int(row["scenario_id"]) for row in valid if row.get("s0")}
    s2_scenarios = {int(row["scenario_id"]) for row in valid if row.get("s2")}
    complete = len(valid_scenarios) >= int(expected_scenarios) and len(valid_snapshots) >= int(expected_snapshots)
    if len(s2_scenarios) >= 2 and complete:
        return OPPORTUNITY_REPLICATED
    if s0_scenarios:
        return OPPORTUNITY_LOCAL
    if not complete:
        return OPPORTUNITY_THIN
    return OPPORTUNITY_ABSENT
