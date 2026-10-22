"""Horizon gate. Thresholds stay in protocol.py."""
from __future__ import annotations


def choose_gate(*, short_confirmed: bool, long_confirmed: bool, promising: bool, only_full_execution: bool) -> str:
    if bool(short_confirmed):
        return "X-A"
    if bool(long_confirmed):
        return "X-C"
    if bool(promising):
        return "X-B"
    if bool(only_full_execution):
        return "X-D"
    return "X-D"


__all__ = ["choose_gate"]
