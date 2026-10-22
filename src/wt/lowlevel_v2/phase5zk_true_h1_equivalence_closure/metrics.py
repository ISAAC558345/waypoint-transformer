"""Locked closure gate. A batch-only numerical miss does not replace the shipped path."""
from __future__ import annotations


def combined_limit(reference: float, atol: float, rtol: float) -> float:
    return float(atol) + float(rtol) * abs(float(reference))


def within_combined(left: float, right: float, atol: float, rtol: float) -> bool:
    """right is the original value. Failures are not decided by relative error alone."""
    return abs(float(left) - float(right)) <= combined_limit(right, atol, rtol)


def choose_gate(
    *,
    baseline_ok: bool,
    behavioral_mismatch: bool,
    numerical_exception: bool,
    speedup_ok: bool,
    latency_ci_positive: bool,
) -> str:
    """ZK-E, then ZK-D, then ZK-B, then ZK-C, otherwise ZK-A."""
    if not bool(baseline_ok):
        return "ZK-E"
    if bool(behavioral_mismatch):
        return "ZK-D"
    if bool(numerical_exception):
        return "ZK-B"
    if not bool(speedup_ok) or not bool(latency_ci_positive):
        return "ZK-C"
    return "ZK-A"


__all__ = ["choose_gate", "combined_limit", "within_combined"]
