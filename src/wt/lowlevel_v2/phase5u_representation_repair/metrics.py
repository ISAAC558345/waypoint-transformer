"""Gate for the representation repair. Thresholds stay in protocol.py."""
from __future__ import annotations


def choose_gate(
    *,
    collision_down: bool,
    collision_worse: bool,
    regret_down: bool,
    j_up: bool,
    j_stable: bool,
    no_reversal: bool,
    rank_clear: bool,
    rank_improved: bool,
    t15_still_high: bool,
    payload_regression: bool,
) -> str:
    if bool(payload_regression) or (bool(collision_worse) and not bool(j_up)):
        return "U-D"
    if bool(collision_down) and bool(regret_down) and bool(j_up) and bool(rank_clear) and bool(no_reversal) and bool(j_stable) and not bool(t15_still_high):
        return "U-A"
    if bool(collision_down) and bool(rank_improved) and (bool(t15_still_high) or not bool(j_stable)):
        return "U-B"
    if (not bool(collision_down)) and not (bool(regret_down) and bool(j_up)):
        return "U-C"
    if bool(collision_down) and not bool(rank_improved):
        return "U-C"
    return "U-B"


__all__ = ["choose_gate"]
