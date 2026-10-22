"""Distillation gate. Thresholds stay in protocol.py."""
from __future__ import annotations

import numpy as np


def choose_gate(*, r1_reproduced: bool, confirmed: bool, mean_improved: bool) -> str:
    """DISTILL-D, then DISTILL-A, then DISTILL-B, otherwise DISTILL-C."""
    if not bool(r1_reproduced):
        return "DISTILL-D"
    if bool(confirmed):
        return "DISTILL-A"
    if bool(mean_improved):
        return "DISTILL-B"
    return "DISTILL-C"


def retention_ratio(gain_predicted: float, gain_true: float, tie_bits: float) -> float:
    if abs(float(gain_true)) <= float(tie_bits):
        return float("nan")
    return float(gain_predicted) / float(gain_true)


__all__ = ["choose_gate", "retention_ratio"]
