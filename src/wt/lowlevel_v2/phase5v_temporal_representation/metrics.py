"""Gate for the temporal-state test. Thresholds stay in protocol.py."""
from __future__ import annotations


def choose_gate(
    *,
    regression: bool,
    j_up: bool,
    j_ci: bool,
    t15_down: bool,
    majority_scenario: bool,
    majority_seed: bool,
    ablation_used: bool,
    case_clear: bool,
    promising: bool,
) -> str:
    if bool(regression):
        return "V-D"
    stable = bool(j_up) and bool(j_ci) and bool(t15_down) and bool(majority_scenario) and bool(majority_seed)
    if stable and bool(ablation_used) and bool(case_clear):
        return "V-A"
    if bool(promising):
        return "V-B"
    return "V-C"


__all__ = ["choose_gate"]
