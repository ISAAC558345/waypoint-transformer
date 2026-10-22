"""Phase-6B bit-queue rules. Reserved confirmation ids stay closed."""
from __future__ import annotations

from pathlib import Path

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6B_BitLevelQueueStateTransition_CN.md"
DEVELOPMENT_POOL = tuple(range(320000, 320064))
RESERVED_POOL = tuple(range(320100, 320132))
HISTORICAL_CONSUMED = tuple(range(310100, 310126))
SANITY_SCENARIOS = (320000, 320001, 320002, 320003)


def assert_sanity_scenario(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_POOL or value in HISTORICAL_CONSUMED:
        raise RuntimeError(f"phase-6b refuses scenario {value}")
    if value not in SANITY_SCENARIOS:
        raise RuntimeError(f"phase-6b sanity check does not open {value}")


def choose_gate(
    *,
    rate_unit_known: bool,
    legacy_coupling_blocks: bool,
    transition_complete: bool,
    conservation_ok: bool,
    tests_ok: bool,
    switch_persists: bool,
    cross_uav_ok: bool,
    no_double_decrement: bool,
    payload_matches_service: bool,
) -> dict:
    if not rate_unit_known:
        gate, name = "BITQ-D", "RATE_UNIT_AMBIGUITY"
    elif legacy_coupling_blocks:
        gate, name = "BITQ-E", "LEGACY_COUPLING_BLOCKS_VALID_IMPLEMENTATION"
    elif not conservation_ok:
        gate, name = "BITQ-C", "CONSERVATION_FAILURE"
    elif not (
        transition_complete
        and tests_ok
        and switch_persists
        and cross_uav_ok
        and no_double_decrement
        and payload_matches_service
    ):
        gate, name = "BITQ-B", "ACCOUNTING_IMPLEMENTATION_INCOMPLETE"
    else:
        gate, name = "BITQ-A", "BIT_LEVEL_ACCOUNTING_VALID"
    return {
        "gate": gate,
        "gate_name": name,
        "phase5zp_gate": "CONF-C",
        "phase5zq_gate": "ATTR-C",
        "phase6a_gate": "BV2-E",
        "phase5_revised": False,
        "planner_ready": False,
        "deployment_verified": False,
        "deployment_pass": False,
        "lambda_switch": "NOT_DEFINED",
        "lambda_progress": "NOT_DEFINED",
        "trained": False,
    }
