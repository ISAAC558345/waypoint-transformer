"""Per-state payload and urgent rules. A 1-bit gap is a tie."""
from __future__ import annotations

from .protocol import EPS_BITS


def payload_loss(payload_bits: float, best_payload_bits: float, eps_bits: float = EPS_BITS) -> float:
    """Loss below one bit is zero. Larger losses keep their bit value."""
    gap = float(best_payload_bits) - float(payload_bits)
    if gap <= float(eps_bits):
        return 0.0
    return float(gap)


def _best_payload(rows: list[dict]) -> float:
    if not rows:
        raise RuntimeError("rule received an empty candidate subset")
    return max(float(row["payload_bits"]) for row in rows)


def _top(rows: list[dict], key: str, eps_bits: float = EPS_BITS) -> dict:
    best = max(float(row[key]) for row in rows)
    group = [row for row in rows if abs(float(row[key]) - best) <= float(eps_bits)]
    if not group:
        raise RuntimeError("top group is empty")
    return min(group, key=lambda row: int(row["candidate_id"]))


def pareto_front(rows: list[dict], eps_bits: float = EPS_BITS) -> list[dict]:
    """A point is dominated when another is no worse and strictly better beyond one bit."""
    front = []
    for row in rows:
        dominated = False
        for other in rows:
            if int(other["candidate_id"]) == int(row["candidate_id"]):
                continue
            payload_gap = float(other["payload_bits"]) - float(row["payload_bits"])
            urgent_gap = float(other["urgent_bits"]) - float(row["urgent_bits"])
            not_worse = payload_gap >= -float(eps_bits) and urgent_gap >= -float(eps_bits)
            strictly_better = payload_gap > float(eps_bits) or urgent_gap > float(eps_bits)
            if not_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda row: (-float(row["payload_bits"]), -float(row["urgent_bits"]), int(row["candidate_id"])))


def payload_id_only(rows: list[dict], eps_bits: float = EPS_BITS) -> dict:
    """Maximum payload. Urgent service is ignored and the smaller id breaks the tie."""
    return _top(rows, "payload_bits", eps_bits)


def rule_p(rows: list[dict], eps_bits: float = EPS_BITS) -> dict:
    """Payload-first. Urgent service breaks ties only inside the 1-bit payload group."""
    best = _best_payload(rows)
    group = [row for row in rows if payload_loss(row["payload_bits"], best, eps_bits) == 0.0]
    return _top(group, "urgent_bits", eps_bits)


def rule_c(rows: list[dict], delta_payload_max: float, eps_bits: float = EPS_BITS) -> dict:
    """Keep candidates inside the payload-loss budget, then maximize urgent service."""
    if float(delta_payload_max) < 0.0:
        raise RuntimeError("payload loss budget is negative")
    best = _best_payload(rows)
    eligible = [row for row in rows if payload_loss(row["payload_bits"], best, eps_bits) <= float(delta_payload_max)]
    if not eligible:
        raise RuntimeError("payload budget excluded every candidate")
    return _top(eligible, "urgent_bits", eps_bits)


def urgent_first(rows: list[dict], eps_bits: float = EPS_BITS) -> dict:
    """Maximum urgent service on the sampled subset. This is not a selected training rule."""
    return _top(rows, "urgent_bits", eps_bits)


def eligible_count(rows: list[dict], delta_payload_max: float, eps_bits: float = EPS_BITS) -> int:
    best = _best_payload(rows)
    return int(sum(payload_loss(row["payload_bits"], best, eps_bits) <= float(delta_payload_max) for row in rows))
