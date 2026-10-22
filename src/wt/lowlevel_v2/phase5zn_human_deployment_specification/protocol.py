"""Phase-5ZN human specification. No algorithm change and no fresh-pool access."""
from __future__ import annotations

import json
from pathlib import Path

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZN_HumanDeploymentSpecificationFreeze_CN.md"
PHASE4E_REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase4E_DeploymentVerification_CN.md"
FRESH_POOL = tuple(range(310118, 310126))
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
FROZEN_FOR_FRESH_CONFIRMATION = False
FRESH_CONFIRMATION_AUTHORIZED = False
MODEL_R1_AGGREGATION_APPROVED = False
AGGREGATION_RULE = "RANK_MEAN_CONSENSUS_V1"
AGGREGATION_APPROVAL = "HUMAN_APPROVAL_REQUIRED"
DEPLOYMENT_LATENCY_REQUIREMENT = "HUMAN_SPECIFICATION_REQUIRED"
TIMEOUT_POLICY = "HUMAN_SPECIFICATION_REQUIRED"
RUNTIME_FALLBACK_POLICY = "HUMAN_SPECIFICATION_REQUIRED"
PHASE4E_STATUS = "DEPLOYMENT_VERIFICATION_FAILED"
RECOMMEND_APPROVAL = True


def assert_not_fresh(scenario_id: int) -> None:
    if int(scenario_id) in FRESH_POOL:
        raise RuntimeError(f"phase-5zn refuses fresh scenario {scenario_id}")


def choose_gate(*, integrity_pass: bool, freeze_consistent: bool, human_approval_pending: bool) -> dict:
    """SPEC-B and SPEC-C stop the phase. SPEC-A is the waiting state."""
    if not integrity_pass:
        gate = "SPEC-B"
        name = "FRESH_POOL_INTEGRITY_FAILED"
    elif not freeze_consistent:
        gate = "SPEC-C"
        name = "ALGORITHM_FREEZE_INCONSISTENT"
    elif human_approval_pending:
        gate = "SPEC-A"
        name = "ALGORITHM_FROZEN_WAITING_HUMAN_APPROVAL"
    else:
        raise RuntimeError("phase-5zn cannot emit a deployment pass")
    return {
        "gate": gate,
        "gate_name": name,
        "planner_ready": False,
        "deployment_verified": False,
        "fresh_confirmation_authorized": False,
    }


def dump_protocol() -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    (PHASE / "phase5zn_protocol.json").write_text(
        json.dumps(
            {
                "aggregation_rule": AGGREGATION_RULE,
                "model_r1_aggregation_approved": MODEL_R1_AGGREGATION_APPROVED,
                "model_r1_aggregation_approval": AGGREGATION_APPROVAL,
                "recommend_approval": RECOMMEND_APPROVAL,
                "deployment_latency_requirement": DEPLOYMENT_LATENCY_REQUIREMENT,
                "timeout_policy": TIMEOUT_POLICY,
                "runtime_fallback_policy": RUNTIME_FALLBACK_POLICY,
                "fresh_confirmation_pool": "310118-310125",
                "fresh_confirmation_authorized": FRESH_CONFIRMATION_AUTHORIZED,
                "planner_ready": PLANNER_READY,
                "deployment_verified": DEPLOYMENT_VERIFIED,
                "frozen_for_fresh_confirmation": FROZEN_FOR_FRESH_CONFIRMATION,
                "phase4e_status": PHASE4E_STATUS,
                "opens_fresh_pool": False,
                "runs_fresh_confirmation": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


__all__ = [
    "AGGREGATION_APPROVAL",
    "AGGREGATION_RULE",
    "DEPLOYMENT_LATENCY_REQUIREMENT",
    "DEPLOYMENT_VERIFIED",
    "FRESH_CONFIRMATION_AUTHORIZED",
    "FRESH_POOL",
    "FROZEN_FOR_FRESH_CONFIRMATION",
    "MODEL_R1_AGGREGATION_APPROVED",
    "PHASE",
    "PHASE4E_REPORT",
    "PHASE4E_STATUS",
    "PLANNER_READY",
    "RECOMMEND_APPROVAL",
    "REPORT",
    "RUNTIME_FALLBACK_POLICY",
    "TIMEOUT_POLICY",
    "assert_not_fresh",
    "choose_gate",
    "dump_protocol",
]
