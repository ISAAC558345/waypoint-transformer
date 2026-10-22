"""Phase-5Z gate. Thresholds stay in protocol.py."""
from __future__ import annotations


def choose_gate(
    *,
    head: bool,
    pooling: bool,
    pair: bool,
    phy: bool,
    temp: bool,
    state_insufficient: bool,
) -> str:
    if bool(head):
        return "Z-E"
    if bool(pooling):
        return "Z-A"
    if bool(pair):
        return "Z-B"
    if bool(phy):
        return "Z-C"
    if bool(temp):
        return "Z-D"
    if bool(state_insufficient):
        return "Z-F"
    return "Z-F"


__all__ = ["choose_gate"]
