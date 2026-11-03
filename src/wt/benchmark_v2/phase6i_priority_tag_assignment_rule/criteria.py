"""Mechanism labels for one fixed rule. They are not an adoption standard."""
from __future__ import annotations

from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.criteria import service_levels
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.protocol import PAYLOAD_FLOOR_BITS

from .protocol import (
    MECHANISM_ABSENT,
    MECHANISM_LOCAL,
    MECHANISM_NO_EFFECT,
    MECHANISM_REPLICATED,
    SCENARIOS,
)


def mechanism_evidence(rows: list[dict]) -> dict:
    """Classify only a complete 16-scenario sample. Integrity failures stay unlabeled."""
    expected = len(SCENARIOS)
    if len(rows) != expected or any(not row.get("interpretable") for row in rows):
        return {"label": None, "published": False, "failed_checks": ["sample is incomplete"]}
    if float(PAYLOAD_FLOOR_BITS) != -1.0:
        return {"label": None, "published": False, "failed_checks": ["payload floor drifted"]}
    if all(not row.get("tagged_differs_from_masked") for row in rows):
        return {"label": MECHANISM_NO_EFFECT, "published": True, "failed_checks": []}
    both = [row for row in rows if row.get("s2_tm") and row.get("s2_tn")]
    safety_ok = all(
        float(row["safety_t"]) <= float(row["safety_m"]) and float(row["safety_t"]) <= float(row["safety_n"])
        for row in rows
    )
    constraint_ok = all(
        float(row["constraint_t"]) <= float(row["constraint_m"]) and float(row["constraint_t"]) <= float(row["constraint_n"])
        for row in rows
    )
    payload_tm = sum(float(row["payload_tm"]) for row in rows) / float(expected)
    payload_tn = sum(float(row["payload_tn"]) for row in rows) / float(expected)
    payload_ok = payload_tm >= float(PAYLOAD_FLOOR_BITS) and payload_tn >= float(PAYLOAD_FLOOR_BITS)
    if len(both) >= 2 and safety_ok and constraint_ok and payload_ok:
        return {"label": MECHANISM_REPLICATED, "published": True, "failed_checks": []}
    if any(row.get("s0_tm") or row.get("s2_tm") for row in rows):
        failed = []
        if len(both) < 2:
            failed.append("fewer than two scenarios reach S2 against both the masked control and the natural baseline")
        if not safety_ok:
            failed.append("raw safety count rose against the masked control or the natural baseline")
        if not constraint_ok:
            failed.append("constraint count rose against the masked control or the natural baseline")
        if not payload_ok:
            failed.append("scenario-mean payload delta is below -1 bit")
        return {"label": MECHANISM_LOCAL, "published": True, "failed_checks": failed}
    return {"label": MECHANISM_ABSENT, "published": True, "failed_checks": ["no scenario reached S0 for the tagged rule against the masked control"]}


def contrast_levels(left: dict, right: dict) -> dict:
    """S0, S1, and S2 for one paired hold. Thresholds stay in the feasibility module."""
    return service_levels(
        float(left["known_tagged_served_bits"]) - float(right["known_tagged_served_bits"]),
        float(left["payload_bits"]) - float(right["payload_bits"]),
        float(left["safety_count"]) - float(right["safety_count"]),
        float(left["constraint_count"]) - float(right["constraint_count"]),
        float(left["mean_sinr"]) - float(right["mean_sinr"]),
        float(left["mean_interference"]) - float(right["mean_interference"]),
    )
