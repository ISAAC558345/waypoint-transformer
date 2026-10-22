"""Failure-attribution rules. They do not reopen the fresh-confirmation gate."""
from __future__ import annotations

from pathlib import Path

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZQ_FreshConfirmationFailureAttribution_CN.md"
FRESH_POOL = tuple(range(310118, 310126))
VALUE_SEEDS = (0, 1, 2, 3, 4)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
PHASE5ZP_GATE = "CONF-C"
REPRODUCTION_LABEL = "REPRODUCTION_ON_CONSUMED_CONFIRMATION_DATA"


def conservation_error(initial: float, arrivals: float, completed: float, dropped: float, residual: float) -> float:
    return float(initial + arrivals - completed - dropped - residual)


def choose_gate(
    *,
    accounting_bug: bool,
    reproduction_mismatch: bool,
    sinr_or_rate_gain: bool,
    progress_or_completion_gain: bool,
    intermediates_differ: bool,
    timing_or_unfinished_differ: bool,
    completion_saturated: bool,
) -> dict:
    """Accounting failure stops first. CONF-C is not revised by any other letter."""
    if accounting_bug or reproduction_mismatch:
        gate, name = "ATTR-E", "ACCOUNTING_OR_IMPLEMENTATION_BUG"
    else:
        not_converted = bool(sinr_or_rate_gain) and not bool(progress_or_completion_gain)
        saturated = bool(completion_saturated) and bool(intermediates_differ) and bool(timing_or_unfinished_differ)
        no_advantage = not bool(sinr_or_rate_gain) and not bool(intermediates_differ) and not bool(progress_or_completion_gain)
        if not_converted and saturated:
            gate, name = "ATTR-D", "MIXED_FAILURE"
        elif saturated:
            gate, name = "ATTR-C", "ENDPOINT_SATURATION_CONFIRMED"
        elif not_converted:
            gate, name = "ATTR-B", "INTERMEDIATE_GAIN_NOT_CONVERTED"
        elif no_advantage:
            gate, name = "ATTR-A", "ALGORITHM_NO_ADVANTAGE"
        else:
            gate, name = "ATTR-D", "MIXED_FAILURE"
    return {
        "gate": gate,
        "gate_name": name,
        "phase5zp_gate": PHASE5ZP_GATE,
        "phase5zp_gate_revised": False,
        "independent_algorithm_confirmation": False,
        "planner_ready": False,
        "deployment_verified": False,
    }
