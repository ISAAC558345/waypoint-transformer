"""Phase-5ZO applies the human specification. It does not open the fresh pool."""
from __future__ import annotations

import json
from pathlib import Path

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZO_HumanSpecificationApplicationAndConfirmationFreeze_CN.md"
PHASE4E_REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase4E_DeploymentVerification_CN.md"
MANIFEST = (
    ROOT
    / "src"
    / "wt"
    / "lowlevel_v2"
    / "phase5zn_human_deployment_specification"
    / "phase5zn_algorithm_freeze_manifest.json"
)
FRESH_POOL = tuple(range(310118, 310126))
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
PHASE4E_STATUS = "DEPLOYMENT_VERIFICATION_FAILED"


def assert_not_fresh(scenario_id: int) -> None:
    if int(scenario_id) in FRESH_POOL:
        raise RuntimeError(f"phase-5zo refuses fresh scenario {scenario_id}")


def choose_gate(
    *,
    hash_match: bool,
    integrity_pass: bool,
    contamination_count: int,
    human_spec_applied: bool,
    protocol_complete: bool,
) -> dict:
    """Hash mismatch stops first. Only FREEZE-A authorizes the next confirmation."""
    if not hash_match:
        gate, name = "FREEZE-B", "ALGORITHM_HASH_MISMATCH"
    elif not integrity_pass:
        gate, name = "FREEZE-C", "FRESH_POOL_INTEGRITY_FAILED"
    elif not human_spec_applied or int(contamination_count) > 0:
        gate, name = "FREEZE-D", "HUMAN_SPECIFICATION_APPLICATION_FAILED"
    elif not protocol_complete:
        gate, name = "FREEZE-E", "PROTOCOL_INCOMPLETE"
    else:
        gate, name = "FREEZE-A", "HUMAN_SPECIFICATION_APPLIED_AND_CONFIRMATION_READY"
    ready = gate == "FREEZE-A"
    return {
        "gate": gate,
        "gate_name": name,
        "section3_stop_name": "FREEZE_MISMATCH" if gate == "FREEZE-B" else None,
        "frozen_for_fresh_confirmation": ready,
        "fresh_confirmation_authorized": ready,
        "planner_ready": False,
        "deployment_verified": False,
    }


def dump_protocol(effective: bool) -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    (PHASE / "phase5zo_protocol.json").write_text(
        json.dumps(
            {
                "human_decisions_are_fixed": True,
                "model_r1_aggregation_rule": "RANK_MEAN_CONSENSUS_V1",
                "model_r1_aggregation_approved": True,
                "deployment_latency_requirement": "NOT_YET_SPECIFIED",
                "latency_gate_active": False,
                "latency_diagnostic_only": True,
                "timeout_policy": "KEEP_CURRENT_TASKOPTION_NO_HOLD_RESET",
                "runtime_fallback_primary": "KEEP_CURRENT_VALID_TASKOPTION",
                "runtime_fallback_secondary": "ABORT_DECISION_IF_NO_VALID_TASKOPTION",
                "fresh_confirmation_pool": "310118-310125",
                "fresh_confirmation_authorized": effective,
                "frozen_for_fresh_confirmation": effective,
                "planner_ready": PLANNER_READY,
                "deployment_verified": DEPLOYMENT_VERIFIED,
                "phase4e_status": PHASE4E_STATUS,
                "opens_fresh_pool_in_this_phase": False,
                "runs_fresh_confirmation_in_this_phase": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


__all__ = [
    "DEPLOYMENT_VERIFIED",
    "FRESH_POOL",
    "MANIFEST",
    "PHASE",
    "PHASE4E_REPORT",
    "PHASE4E_STATUS",
    "PLANNER_READY",
    "REPORT",
    "ROOT",
    "assert_not_fresh",
    "choose_gate",
    "dump_protocol",
]
