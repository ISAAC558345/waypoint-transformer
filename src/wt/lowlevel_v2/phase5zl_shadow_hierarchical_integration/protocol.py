"""Phase-5ZL shadow integration contract. Thresholds are fixed before replay."""
from __future__ import annotations

import json
from pathlib import Path

from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import CONFIRM_SCENARIOS, TIMES
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import SEEDS
from wt.lowlevel_v2.phase5zk_true_h1_equivalence_closure.protocol import ATOL, RTOL

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZL_TrueH1HierarchicalShadowIntegration_CN.md"
PHASE4E_REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase4E_DeploymentVerification_CN.md"
FORBIDDEN = frozenset(range(310118, 310126)) | frozenset(range(230000, 230128))
EVAL_SCENARIOS = tuple(int(item) for item in CONFIRM_SCENARIOS)
EVAL_TIMES = tuple(int(item) for item in TIMES)
VALUE_SEEDS = tuple(int(item) for item in SEEDS)
HOLD_STEPS = 15
WARMUP = 1
REPEATS = 3
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
NEAR_TIE_ABS = 1.0e-6
BASELINE_VALUE_ABS = 1.0e-3
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
FROZEN_FOR_FRESH_CONFIRMATION = False
DEPLOYMENT_LATENCY_REQUIREMENT = "NOT_SPECIFIED"
PHASE4E_STATUS = "DEPLOYMENT_VERIFICATION_FAILED"
AGGREGATION = "per_seed_only"
TIE_BREAK = "stable_argsort_descending"


def assert_consumed(scenario_id: int) -> None:
    if int(scenario_id) in FORBIDDEN or int(scenario_id) not in EVAL_SCENARIOS:
        raise RuntimeError(f"phase-5zl refuses scenario {scenario_id}")


def within_combined(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= float(ATOL) + float(RTOL) * abs(float(right))


def choose_gate(
    *,
    reproduced: bool,
    isolated: bool,
    mapping_ok: bool,
    decisions_match: bool,
    seeds_agree: bool,
    latency_specified: bool,
    fallback_approved: bool,
) -> dict:
    """Primary letter follows E, B, C, D. Every blocker is retained."""
    blockers: list[str] = []
    if not reproduced or not decisions_match:
        blockers.append("INT-E")
    if not isolated or not mapping_ok:
        blockers.append("INT-B")
    if not seeds_agree:
        blockers.append("INT-C")
    if not latency_specified or not fallback_approved:
        blockers.append("INT-D")
    order = ("INT-E", "INT-B", "INT-C", "INT-D")
    primary = next((item for item in order if item in blockers), "INT-A")
    names = {
        "INT-A": "SHADOW_INTEGRATION_CONTRACT_VERIFIED",
        "INT-B": "INTERFACE_OR_STATE_ISOLATION_ISSUE",
        "INT-C": "DECISION_OR_MODEL_SELECTION_UNRESOLVED",
        "INT-D": "LATENCY_OR_FALLBACK_SPEC_REQUIRED",
        "INT-E": "REPRODUCTION_FAILED",
    }
    return {
        "gate": primary,
        "gate_name": names[primary],
        "blockers": blockers,
        "contract_technically_clean": primary not in {"INT-B", "INT-E"} and "INT-B" not in blockers and "INT-E" not in blockers,
        "planner_ready": False,
    }


def dump_protocol() -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    (PHASE / "phase5zl_protocol.json").write_text(
        json.dumps(
            {
                "atol": ATOL,
                "rtol": RTOL,
                "baseline_value_abs": BASELINE_VALUE_ABS,
                "hold_steps": HOLD_STEPS,
                "warmup": WARMUP,
                "repeats": REPEATS,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "near_tie_abs": NEAR_TIE_ABS,
                "aggregation": AGGREGATION,
                "tie_break": TIE_BREAK,
                "deployment_latency_requirement": DEPLOYMENT_LATENCY_REQUIREMENT,
                "phase4e_status": PHASE4E_STATUS,
                "planner_ready": PLANNER_READY,
                "deployment_verified": DEPLOYMENT_VERIFIED,
                "frozen_for_fresh_confirmation": FROZEN_FOR_FRESH_CONFIRMATION,
                "scenarios": list(EVAL_SCENARIOS),
                "times": list(EVAL_TIMES),
                "seeds": list(VALUE_SEEDS),
                "gate_priority": ["INT-E", "INT-B", "INT-C", "INT-D", "INT-A"],
                "record_all_blockers": True,
                "calls_deployment_decision": False,
                "prior_unified_median_ms": 357.55,
                "prior_median_includes_taskoption": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


__all__ = [
    "AGGREGATION",
    "BASELINE_VALUE_ABS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "DEPLOYMENT_LATENCY_REQUIREMENT",
    "DEPLOYMENT_VERIFIED",
    "EVAL_SCENARIOS",
    "EVAL_TIMES",
    "FORBIDDEN",
    "FROZEN_FOR_FRESH_CONFIRMATION",
    "HOLD_STEPS",
    "NEAR_TIE_ABS",
    "PHASE",
    "PHASE4E_REPORT",
    "PHASE4E_STATUS",
    "PLANNER_READY",
    "REPEATS",
    "REPORT",
    "ROOT",
    "TIE_BREAK",
    "VALUE_SEEDS",
    "WARMUP",
    "assert_consumed",
    "choose_gate",
    "dump_protocol",
    "within_combined",
]
