"""Locked ranking-distillation gate."""
from __future__ import annotations


def choose_gate(
    *,
    aligned: bool,
    j15_mean_up: bool,
    ci_low_positive: bool,
    seeds_ok: bool,
    retention_ok: bool,
    ranking_ok: bool,
    physical_ok: bool,
    ranking_up: bool,
    j15_degraded: bool,
) -> str:
    """RANK-D, then degradation, then RANK-A, then RANK-B, otherwise RANK-C."""
    if not bool(aligned):
        return "RANK-D"
    if bool(j15_degraded) or not bool(physical_ok):
        return "RANK-C"
    if bool(j15_mean_up) and bool(ci_low_positive) and bool(seeds_ok) and bool(retention_ok) and bool(ranking_ok):
        return "RANK-A"
    if bool(ranking_up):
        return "RANK-B"
    return "RANK-C"


__all__ = ["choose_gate"]
