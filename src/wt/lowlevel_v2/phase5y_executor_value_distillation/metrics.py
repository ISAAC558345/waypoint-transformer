"""Phase-5Y gate. Thresholds stay in protocol.py."""
from __future__ import annotations


def choose_gate(
    *,
    confirmed: bool,
    promising: bool,
    regression_without_decision: bool,
    representation_limit: bool,
) -> str:
    if bool(confirmed):
        return "Y-A"
    if bool(representation_limit):
        return "Y-D"
    if bool(regression_without_decision):
        return "Y-C"
    if bool(promising):
        return "Y-B"
    return "Y-E"


__all__ = ["choose_gate"]
