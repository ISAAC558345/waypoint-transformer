"""Match rules and the E9 gate. Thresholds are not adjusted after a result."""
from __future__ import annotations

from .protocol import (
    ACCOUNTING_TOLERANCE_BITS,
    BIT_FIELDS,
    DECOMPOSITION_TOLERANCE_BITS,
    GATE_BLOCKED,
    GATE_COMPLETE,
    GATE_INCONCLUSIVE,
    PATH_TOLERANCE_M,
)


def episode_status(row: dict, saved: dict) -> dict:
    bit_gaps = {key: abs(float(row[key]) - float(saved[key])) for key in BIT_FIELDS}
    path_gap = abs(float(row["path_length_m"]) - float(saved["path_length_m"]))
    age = float(row["age_service_bits"])
    increment = float(row["tag_increment_service_bits"])
    combined = float(row["combined_urgency_served_bits"])
    recorded = float(row["recorded_urgency_served_bits"])
    identity_gap = abs(age + increment - combined)
    ledger_gap = abs(combined - recorded)
    queue_identity = abs(
        float(row["final_ordinary_queue_bits"]) + float(row["final_tagged_queue_bits"]) - float(row["final_queue_bits"])
    )
    bit_ok = all(gap <= float(ACCOUNTING_TOLERANCE_BITS) for gap in bit_gaps.values())
    ok = bool(
        bit_ok
        and path_gap <= float(PATH_TOLERANCE_M)
        and identity_gap <= float(DECOMPOSITION_TOLERANCE_BITS)
        and ledger_gap <= float(DECOMPOSITION_TOLERANCE_BITS)
        and queue_identity <= float(ACCOUNTING_TOLERANCE_BITS)
        and float(row["max_ledger_gap_bits"]) <= float(ACCOUNTING_TOLERANCE_BITS)
        and int(row["steps"]) == int(saved["steps"])
    )
    return {
        "matches_saved": ok,
        "bit_gaps": bit_gaps,
        "max_bit_gap": float(max(bit_gaps.values()) if bit_gaps else 0.0),
        "path_gap_m": float(path_gap),
        "identity_gap_bits": float(identity_gap),
        "ledger_urgency_gap_bits": float(ledger_gap),
        "queue_identity_gap_bits": float(queue_identity),
    }


def choose_gate(
    *,
    integrity_ok: bool,
    budget_ok: bool,
    reproduced: bool,
    decomposition_ok: bool,
    l2_groups: list[dict] | None,
) -> str:
    if not integrity_ok or not budget_ok or not reproduced or not decomposition_ok:
        return GATE_BLOCKED
    if l2_groups is None or len(l2_groups) != 3:
        return GATE_BLOCKED
    signs = []
    dominants = []
    for group in l2_groups:
        combined = float(group["combined_delta_mean"])
        age = float(group["age_delta_mean"])
        tag = float(group["tag_increment_delta_mean"])
        if combined == 0.0 or age == tag:
            return GATE_INCONCLUSIVE
        signs.append(combined > 0.0)
        dominants.append("age" if age > tag else "tag_increment")
    if len(set(signs)) != 1 or len(set(dominants)) != 1:
        return GATE_INCONCLUSIVE
    return GATE_COMPLETE
