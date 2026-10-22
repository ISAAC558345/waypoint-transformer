"""Locked route gate. Thresholds stay in protocol.py."""
from __future__ import annotations


def choose_gate(
    *,
    implementation_error: bool,
    teacher_mismatch: bool,
    easy_region: bool,
    true_h1_stronger: bool,
) -> str:
    """ZI-A, then ZI-C, then ZI-B, then ZI-D, otherwise ZI-E."""
    if bool(implementation_error):
        return "ZI-A"
    if bool(teacher_mismatch):
        return "ZI-C"
    if bool(easy_region):
        return "ZI-B"
    if bool(true_h1_stronger):
        return "ZI-D"
    return "ZI-E"


__all__ = ["choose_gate"]
