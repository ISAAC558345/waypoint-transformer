"""Phase-5ZP gate rules. Locked to the Phase-5ZO preregistration."""
from __future__ import annotations

from pathlib import Path

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZP_OneShotFreshConfirmation_CN.md"
PHASE4E_REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase4E_DeploymentVerification_CN.md"
FRESH_POOL = tuple(range(310118, 310126))
VALUE_SEEDS = (0, 1, 2, 3, 4)
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
HOLD_STEPS = 15
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
PHASE4E_STATUS = "DEPLOYMENT_VERIFICATION_FAILED"
PRIMARY_BASELINE = "Frozen Model-S"
PRIMARY_SYSTEM = "Frozen Rank-Mean True-H1 Hierarchical Controller"
MODEL_S_PAYLOAD = "mean_of_five_frozen_static_seed_episode_payloads"
MODEL_S_SELECTION = "per_seed_stable_argmax_on_the_same_120_pool_with_zero_response"


def confirmation_may_start(pre_access_match: bool) -> bool:
    return bool(pre_access_match)


def model_s_payload(seed_payloads: list[float]) -> float:
    """Five frozen static seeds. This is their mean, not the best seed."""
    if len(seed_payloads) != len(VALUE_SEEDS):
        raise ValueError("Model-S payload needs the five frozen static seeds")
    return float(sum(seed_payloads) / float(len(seed_payloads)))


def choose_gate(
    *,
    protocol_violation: bool,
    illegal_assignment: int,
    state_contamination: int,
    nan_final_decision: int,
    inf_final_decision: int,
    unhandled_runtime_exception: int,
    mean_delta: float,
    ci_low: float,
) -> dict:
    """Safety and protocol outrank a positive mean. Latency is not an input."""
    safety = (
        int(illegal_assignment) > 0
        or int(state_contamination) > 0
        or int(nan_final_decision) > 0
        or int(inf_final_decision) > 0
        or int(unhandled_runtime_exception) > 0
    )
    if protocol_violation:
        gate, name = "CONF-E", "PROTOCOL_VIOLATION"
    elif safety or mean_delta != mean_delta or ci_low != ci_low:
        gate, name = "CONF-D", "SAFETY_OR_EXECUTION_FAILURE"
    elif float(mean_delta) <= 0.0:
        gate, name = "CONF-C", "NO_INDEPENDENT_GAIN"
    elif float(ci_low) <= 0.0:
        gate, name = "CONF-B", "POSITIVE_MEAN_BUT_NOT_STATISTICALLY_CONFIRMED"
    else:
        gate, name = "CONF-A", "INDEPENDENT_PERFORMANCE_CONFIRMED"
    return {
        "gate": gate,
        "gate_name": name,
        "independent_algorithm_confirmation": gate == "CONF-A",
        "planner_ready": False,
        "deployment_verified": False,
        "latency_gate_active": False,
    }
