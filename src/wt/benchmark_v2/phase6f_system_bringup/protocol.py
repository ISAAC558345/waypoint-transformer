"""Phase-6F bring-up rules. The reserved pool stays closed."""
from __future__ import annotations

from pathlib import Path

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6F_BenchmarkV2_EndToEndSystemBringUp_CN.md"
SMOKE_POOL = tuple(range(320000, 320004))
INTEGRATION_POOL = tuple(range(320008, 320016))
RESERVED_POOL = tuple(range(320100, 320132))
HISTORICAL_CONSUMED = tuple(range(310100, 310126))
HOLD = 15
GENERATOR = "G1_RATE"
CANDIDATE_K = 5


def assert_allowed(scenario_id: int, *, split: str) -> None:
    value = int(scenario_id)
    if value in RESERVED_POOL or value in HISTORICAL_CONSUMED:
        raise RuntimeError(f"phase-6f refuses scenario {value}")
    pool = SMOKE_POOL if split == "smoke" else INTEGRATION_POOL
    if split == "maxrate":
        pool = SMOKE_POOL
    if value not in pool:
        raise RuntimeError(f"phase-6f {split} split does not include {value}")


def choose_gate(
    *,
    contamination: int,
    accounting_failures: int,
    crashed: bool,
    smoke_ran: int,
    smoke_complete: int,
    integration_complete: int,
    illegal: int,
    nan_inf: int,
    hold_ok: bool,
    chain_ok: bool,
    logs_ok: bool,
) -> dict:
    interface_ok = int(illegal) == 0 and int(nan_inf) == 0 and bool(hold_ok) and bool(chain_ok) and bool(logs_ok)
    if int(contamination) > 0:
        gate, name = "RUN-C", "SHADOW_STATE_CONTAMINATION"
    elif int(accounting_failures) > 0:
        gate, name = "RUN-D", "ACCOUNTING_REGRESSION"
    elif crashed or int(smoke_complete) < int(smoke_ran):
        gate, name = "RUN-E", "SYSTEM_RUNTIME_FAILURE"
    elif not interface_ok:
        gate, name = "RUN-B", "INTERFACE_FAILURE"
    elif int(smoke_ran) < len(SMOKE_POOL) or int(integration_complete) < len(INTEGRATION_POOL):
        gate, name = "RUN-E", "SYSTEM_RUNTIME_FAILURE"
    else:
        gate, name = "RUN-A", "END_TO_END_SYSTEM_OPERATIONAL"
    return {
        "gate": gate,
        "gate_name": name,
        "phase6b_gate": "BITQ-A",
        "phase6c_gate": "DYN-A",
        "phase6d_gate": "TEMP-F",
        "phase6e_gate": "CAND-D",
        "planner_ready": False,
        "deployment_verified": False,
        "deployment_pass": False,
        "trained": False,
        "fresh_confirmation_untouched": True,
        "parameters_modified_for_speed": False,
        "train_hgnn_now": False,
        "next_phase_trains_low_level_mappo_only": gate == "RUN-A",
        "optimize_j15_runtime": False,
    }
