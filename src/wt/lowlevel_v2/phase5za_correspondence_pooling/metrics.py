"""Phase-5ZA gate. Thresholds stay in protocol.py."""
from __future__ import annotations


def choose_gate(
    *,
    structural_ok: bool,
    collision_fixed: bool,
    improved_v: bool,
    ci_above_zero: bool,
    improved_y: bool,
    scenarios_ok: bool,
    seeds_ok: bool,
    top3_ok: bool,
    regret_ok: bool,
    case_ok: bool,
) -> str:
    if not bool(structural_ok) or not bool(collision_fixed):
        return "POOL-D"
    confirmed = all(
        (
            bool(improved_v),
            bool(ci_above_zero),
            bool(improved_y),
            bool(scenarios_ok),
            bool(seeds_ok),
            bool(top3_ok),
            bool(regret_ok),
            bool(case_ok),
        )
    )
    if confirmed:
        return "POOL-A"
    if bool(improved_v):
        return "POOL-B"
    return "POOL-C"


__all__ = ["choose_gate"]
