"""Two sampled-pool rules. Urgency may break a 1-bit payload tie and nothing else."""
from __future__ import annotations

from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import EPS_BITS


def _best_payload(rows: list[dict]) -> float:
    if not rows:
        raise RuntimeError("rule received an empty candidate subset")
    return max(float(row["payload_bits"]) for row in rows)


def _payload_group(rows: list[dict], eps_bits: float = EPS_BITS) -> list[dict]:
    best = _best_payload(rows)
    group = [row for row in rows if abs(float(row["payload_bits"]) - best) <= float(eps_bits)]
    if not group:
        raise RuntimeError("payload tie group is empty")
    return group


def payload_first(rows: list[dict], eps_bits: float = EPS_BITS) -> dict:
    """Maximum payload label. The smaller candidate id breaks a 1-bit tie."""
    group = _payload_group(rows, eps_bits)
    return min(group, key=lambda row: int(row["candidate_id"]))


def urgency_tiebreak(rows: list[dict], eps_bits: float = EPS_BITS) -> dict:
    """Maximum urgent service inside the payload tie. A lower payload outside that tie is refused."""
    best = _best_payload(rows)
    group = _payload_group(rows, eps_bits)
    best_urgent = max(float(row["urgent_service_bits"]) for row in group)
    finalists = [row for row in group if abs(float(row["urgent_service_bits"]) - best_urgent) <= float(eps_bits)]
    chosen = min(finalists, key=lambda row: int(row["candidate_id"]))
    if abs(float(chosen["payload_bits"]) - best) > float(eps_bits):
        raise RuntimeError("urgency tie-break selected a candidate outside the 1-bit payload group")
    return chosen


def choose_executed(rows: list[dict], arm: str, eps_bits: float = EPS_BITS) -> tuple[dict, dict, dict]:
    payload_choice = payload_first(rows, eps_bits)
    tie_choice = urgency_tiebreak(rows, eps_bits)
    gap = float(tie_choice["payload_bits"]) - float(payload_choice["payload_bits"])
    if gap < -float(eps_bits):
        raise RuntimeError("tie-break payload is more than 1 bit below payload-first")
    if arm == "payload_first":
        executed = payload_choice
    elif arm == "payload_first_urgency_tiebreak":
        executed = tie_choice
    else:
        raise RuntimeError(f"unknown arm {arm}")
    if abs(float(executed["payload_bits"]) - _best_payload(rows)) > float(eps_bits):
        raise RuntimeError("executed candidate is outside the 1-bit payload group")
    return payload_choice, tie_choice, executed
