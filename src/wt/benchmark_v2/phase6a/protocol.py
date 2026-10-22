"""Benchmark-V2 audit rules. Reserved confirmation ids cannot be opened."""
from __future__ import annotations

from pathlib import Path

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6A_BenchmarkV2_CommunicationUtilityAndContinuity_CN.md"
DEVELOPMENT_POOL = tuple(range(320000, 320064))
RESERVED_POOL = tuple(range(320100, 320132))
HISTORICAL_CONSUMED = tuple(range(310100, 310126))
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
PHASE5ZP_GATE = "CONF-C"
PHASE5ZQ_GATE = "ATTR-C"
SATURATION_MAX = 0.50
PAIRWISE_TIE_MAX = 0.40
DISTINGUISH_MIN = 0.75
SWITCH_NEGATIVE_SPEARMAN = -0.20
CONSERVATION_ABS = 1.0e-6


def assert_allowed(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_POOL or value in HISTORICAL_CONSUMED:
        raise RuntimeError(f"phase-6a refuses scenario {value}")
    if value not in DEVELOPMENT_POOL:
        raise RuntimeError(f"phase-6a development pool does not include {value}")


def choose_gate(
    *,
    bit_queue_exists: bool,
    e1_matches_environment_transition: bool,
    e1_e2_conserved: bool,
    e0_saturated: bool,
    endpoint_separates: bool,
    switch_negatively_related: bool,
    continuity_pattern: bool,
) -> dict:
    """A bit metric that is not the environment transition cannot be adopted."""
    if (not bit_queue_exists) or (not e1_matches_environment_transition) or (not e1_e2_conserved):
        gate, name = "BV2-E", "ACCOUNTING_MODEL_INVALID"
    elif e0_saturated and endpoint_separates and switch_negatively_related:
        gate, name = "BV2-C", "BOTH_ENDPOINT_AND_CONTINUITY"
    elif endpoint_separates and not switch_negatively_related:
        gate, name = "BV2-A", "OLD_ENDPOINT_SATURATION_ONLY"
    elif continuity_pattern:
        gate, name = "BV2-B", "CONTINUITY_BOTTLENECK_CONFIRMED"
    else:
        gate, name = "BV2-D", "BENCHMARK_V2_STILL_NOT_DISCRIMINATIVE"
    return {
        "gate": gate,
        "gate_name": name,
        "phase5zp_gate": PHASE5ZP_GATE,
        "phase5zq_gate": PHASE5ZQ_GATE,
        "phase5_revised": False,
        "planner_ready": False,
        "deployment_verified": False,
        "lambda_selected": False,
    }
