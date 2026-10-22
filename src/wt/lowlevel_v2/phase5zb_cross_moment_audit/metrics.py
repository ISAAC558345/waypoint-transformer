"""Phase-5ZB gate. Thresholds stay in protocol.py."""
from __future__ import annotations


def choose_gate(
    *,
    permutation_ok: bool,
    cm_a: bool,
    cm_all_passes: bool,
    helped: bool,
    no_reduction: bool,
) -> str:
    if not bool(permutation_ok):
        return "CM-D"
    if bool(cm_a):
        return "CM-A"
    if bool(cm_all_passes):
        return "CM-B"
    if bool(no_reduction):
        return "CM-C"
    if bool(helped):
        return "CM-B"
    return "CM-C"


__all__ = ["choose_gate"]
