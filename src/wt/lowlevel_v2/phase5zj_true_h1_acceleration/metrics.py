"""Locked acceleration gate."""
from __future__ import annotations


def choose_gate(
    *,
    baseline_ok: bool,
    batch_safe: bool,
    equivalent: bool,
    speedup_ok: bool,
    latency_ci_positive: bool,
) -> str:
    """FAST-E, then FAST-D, then FAST-C, then FAST-A, otherwise FAST-B."""
    if not bool(baseline_ok):
        return "FAST-E"
    if not bool(batch_safe):
        return "FAST-D"
    if not bool(equivalent):
        return "FAST-C"
    if bool(speedup_ok) and bool(latency_ci_positive):
        return "FAST-A"
    return "FAST-B"


__all__ = ["choose_gate"]
