"""Phase-4D-5 candidates. Values come from design principles, not the 64-set."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase4d_deployment_gate_audit.protocol import TAU_SOURCE, TAU_UHAT
from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
REPORT_NAME = "HTW_Phase4D5_DeploymentCriterionCandidates_CN.md"

# Frozen Gate-1. Not modified. Not a deployment criterion.
FROZEN_GATE1 = {
    "role": "per-scenario B2 vs LOCAL switch",
    "margin": "U_hat(argmax U_hat) - U_hat(argmax U_single)",
    "tau": TAU_UHAT,
    "tau_display": 1.535e6,
    "tau_source": TAU_SOURCE,
    "rule": "if margin >= tau: B2 else LOCAL",
}

# Observed reference only. Forbidden as a threshold source.
OBSERVED_REFERENCE = {
    "mean_delta_R": 0.111,
    "degrade_rate": 0.047,
    "degrade": "3/64",
    "worst_case_delta_R": -0.214,
    "mean_R": 0.874,
    "n_B2": "26/64",
    "disclaimer": (
        "These are observed evaluation results and are NOT a source of candidate thresholds."
    ),
}

# R is defined on [0, 1]. Decimal fractions of that unit interval are
# design constants, not statistics of the current 64 scenarios.
R_UNIT = 1.0

CANDIDATES = (
    {
        "candidate_id": "A",
        "name": "Strict Safety",
        "principle": "Single-scenario safety first. Any regression versus LOCAL is a deployment defect.",
        "max_degradation_rate": 0.0,
        "max_worst_case_delta_R": 0.0,
        "min_mean_delta_R": 0.0,
        "min_mean_U_gain": 0.0,
        "scenario_degradation_policy": "forbid_all",
        "require_all_scenarios_noninferior": True,
        "independent_holdout_required": True,
        "required_independent_batches": 1,
        "rationale": {
            "C1": "Zero allowed degrade rate: a planner change must not make any scenario worse than LOCAL.",
            "C2": "Zero allowed worst-case ΔR: the worst scenario must stay non-negative versus LOCAL.",
            "C3": "Mean ΔR >= 0 is implied by strict non-inferiority; recorded explicitly as a floor of 0.",
            "C4": "Mean U_traj must not fall below LOCAL. No positive bit-margin is invented from current gain.",
            "C5": "forbid_all: fallback to LOCAL is not an excuse to accept a harmful B2 substitution.",
            "C6": "strict non-inferiority: for every scenario, R_selected >= R_LOCAL.",
            "C7": "A second unseen batch is required before a strict-safety claim can be treated as deployable.",
            "C8": "One additional independently generated batch beyond 220000-220063.",
            "not_from_observed_64": True,
        },
    },
    {
        "candidate_id": "B",
        "name": "Balanced Deployment",
        "principle": "Allow only rare, bounded losses; require non-negative average gain and one extra unseen batch.",
        "max_degradation_rate": 0.01,
        "max_worst_case_delta_R": -0.10,
        "min_mean_delta_R": 0.0,
        "min_mean_U_gain": 0.0,
        "scenario_degradation_policy": "allow_only_if_worst_case_bound_holds",
        "require_all_scenarios_noninferior": False,
        "independent_holdout_required": True,
        "required_independent_batches": 1,
        "rationale": {
            "C1": "1-in-100 rare-event cap. Decimal 0.01 is a conventional reliability quota, not 3/64 or 0.047.",
            "C2": "At most 0.10 of the R unit interval [0, 1]. A one-tenth score drop is the design loss bound, not -0.214.",
            "C3": "Average rank-normalized score must not decline. Floor is 0, not 0.111.",
            "C4": "Average payload must not decline versus LOCAL. Floor is 0 bits, not the current 1.34e6 gain.",
            "C5": "A rare degradation is allowed only if it also respects the C2 loss bound.",
            "C6": "aggregate non-inferiority: summary statistics may pass even if a rare scenario does not.",
            "C7": "One new unseen batch is required so Gate-1 safety is not judged only on 220000-220063.",
            "C8": "One additional independently generated batch.",
            "not_from_observed_64": True,
        },
    },
    {
        "candidate_id": "C",
        "name": "Conservative-with-Fallback",
        "principle": "Use LOCAL fallback as a risk control; allow limited degradation if average gain, bounded loss, and repeated unseen batches all hold.",
        "max_degradation_rate": 0.02,
        "max_worst_case_delta_R": -0.125,
        "min_mean_delta_R": 0.02,
        "min_mean_U_gain": 0.0,
        "scenario_degradation_policy": "allow_some",
        "require_all_scenarios_noninferior": False,
        "independent_holdout_required": True,
        "required_independent_batches": 2,
        "rationale": {
            "C1": "2-in-100 limited-tolerance cap. Decimal 0.02 is a conventional minority quota, not 3/64.",
            "C2": "At most 1/8 of the R unit interval. Dyadic 0.125 is a design constant, not a pad around -0.214.",
            "C3": "At least 0.02 of the R unit interval as a minimum average improvement that can justify changing planner defaults.",
            "C4": "Average payload must still not fall below LOCAL. No bit-margin is copied from current U gain.",
            "C5": "allow_some: fallback already withholds B2 when margin < tau; residual degradations are tolerated only inside C1/C2.",
            "C6": "aggregate non-inferiority; strict per-scenario non-inferiority is not required because fallback is part of the system.",
            "C7": "Independent hold-out is mandatory for this candidate by design.",
            "C8": "Two additional independently generated batches, because fallback-plus-B2 is a system claim, not a single-table claim.",
            "not_from_observed_64": True,
        },
    },
)

__all__ = [
    "CANDIDATES",
    "FROZEN_GATE1",
    "OBSERVED_REFERENCE",
    "OUTPUT",
    "PHASE",
    "REPORT_NAME",
    "ROOT",
    "R_UNIT",
    "TAU_SOURCE",
    "TAU_UHAT",
]
