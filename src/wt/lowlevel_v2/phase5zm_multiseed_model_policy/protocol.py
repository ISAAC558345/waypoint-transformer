"""Phase-5ZM thresholds. Locked before any consensus measurement."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import CONFIRM_SCENARIOS, TIMES
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import SEEDS

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZM_ModelR1MultiSeedConsensusAndDeploymentPolicy_CN.md"
PHASE4E_REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase4E_DeploymentVerification_CN.md"
FORBIDDEN = frozenset(range(310118, 310126)) | frozenset(range(230000, 230128))
EVAL_SCENARIOS = tuple(int(item) for item in CONFIRM_SCENARIOS)
EVAL_TIMES = tuple(int(item) for item in TIMES)
VALUE_SEEDS = tuple(int(item) for item in SEEDS)
AGGREGATION = "RANK_MEAN_CONSENSUS_V1"
HOLD_STEPS = 15
WARMUP = 1
REPEATS = 3
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
SEED_PERM_SEED = 0
CANDIDATE_PERM_SEED = 1
CANDIDATE_PERM_COUNT = 10
SEED_PERM_RANDOM_COUNT = 10
J15_RETAIN_ABS = 1.0
LARGE_CONFLICT_FRACTION = 0.5
LOW_STRENGTH = 0.4
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
FROZEN_FOR_FRESH_CONFIRMATION = False
FRESH_CONFIRMATION_AUTHORIZED = False
MODEL_R1_AGGREGATION_APPROVED = False
DEPLOYMENT_LATENCY_REQUIREMENT = "NOT_SPECIFIED"
PHASE4E_STATUS = "DEPLOYMENT_VERIFICATION_FAILED"
CASE_SCENARIO = 310111
CASE_TIME = 15


def _permutations() -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    seed_rng = np.random.default_rng(SEED_PERM_SEED)
    seed_perms = [tuple(range(5)), tuple(range(4, -1, -1))]
    for _ in range(SEED_PERM_RANDOM_COUNT):
        seed_perms.append(tuple(int(item) for item in seed_rng.permutation(5)))
    candidate_rng = np.random.default_rng(CANDIDATE_PERM_SEED)
    candidate_perms = [
        tuple(int(item) for item in candidate_rng.permutation(120)) for _ in range(CANDIDATE_PERM_COUNT)
    ]
    return seed_perms, candidate_perms


SEED_PERMUTATIONS, CANDIDATE_PERMUTATIONS = _permutations()


def assert_consumed(scenario_id: int) -> None:
    if int(scenario_id) in FORBIDDEN or int(scenario_id) not in EVAL_SCENARIOS:
        raise RuntimeError(f"phase-5zm refuses scenario {scenario_id}")


def choose_gate(
    *,
    permutation_ok: bool,
    j15_retained: bool,
    illegal: bool,
    isolation_ok: bool,
    large_conflict: bool,
    live_matches_saved: bool,
    human_spec_approved: bool,
) -> dict:
    """MS-C outranks MS-B. MS-D stays beside MS-A when only human approval is missing."""
    blockers: list[str] = []
    if not j15_retained:
        blockers.append("MS-C")
    if not permutation_ok or illegal or not isolation_ok or large_conflict or not live_matches_saved:
        blockers.append("MS-B")
    if not human_spec_approved:
        blockers.append("MS-D")
    names = {
        "MS-A": "SEED_NEUTRAL_AGGREGATION_VERIFIED",
        "MS-B": "AGGREGATION_CHANGES_DECISION_UNSTABLY",
        "MS-C": "AGGREGATION_DEGRADES_EXISTING_R1",
        "MS-D": "HUMAN_DEPLOYMENT_SPEC_REQUIRED",
    }
    if "MS-C" in blockers:
        primary = "MS-C"
    elif "MS-B" in blockers:
        primary = "MS-B"
    else:
        primary = "MS-A"
    return {
        "gate": primary,
        "gate_name": names[primary],
        "blockers": blockers,
        "aggregation_technically_verified": primary == "MS-A",
        "planner_ready": False,
        "fresh_confirmation_authorized": False,
    }


def dump_protocol() -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    (PHASE / "phase5zm_protocol.json").write_text(
        json.dumps(
            {
                "aggregation": AGGREGATION,
                "rank_definition": "competition rank, 1 is best, ties share the minimum rank",
                "tie_break_1": "higher majority_top1_count",
                "tie_break_2": "lexicographic assignment tuple",
                "uses_j15_in_selection": False,
                "uses_scenario_id_in_selection": False,
                "trainable_parameters": 0,
                "seed_permutations": [list(item) for item in SEED_PERMUTATIONS],
                "candidate_permutation_count": len(CANDIDATE_PERMUTATIONS),
                "candidate_permutation_seed": CANDIDATE_PERM_SEED,
                "j15_retain_abs_bit": J15_RETAIN_ABS,
                "large_conflict_fraction": LARGE_CONFLICT_FRACTION,
                "low_strength": LOW_STRENGTH,
                "warmup": WARMUP,
                "repeats": REPEATS,
                "deployment_latency_requirement": DEPLOYMENT_LATENCY_REQUIREMENT,
                "model_r1_aggregation_approved": MODEL_R1_AGGREGATION_APPROVED,
                "fresh_confirmation_authorized": FRESH_CONFIRMATION_AUTHORIZED,
                "fresh_confirmation_pool": "310118-310125",
                "planner_ready": PLANNER_READY,
                "deployment_verified": DEPLOYMENT_VERIFIED,
                "frozen_for_fresh_confirmation": FROZEN_FOR_FRESH_CONFIRMATION,
                "phase4e_status": PHASE4E_STATUS,
                "scenarios": list(EVAL_SCENARIOS),
                "times": list(EVAL_TIMES),
                "seeds": list(VALUE_SEEDS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


__all__ = [
    "AGGREGATION",
    "CANDIDATE_PERMUTATIONS",
    "CASE_SCENARIO",
    "CASE_TIME",
    "DEPLOYMENT_LATENCY_REQUIREMENT",
    "DEPLOYMENT_VERIFIED",
    "EVAL_SCENARIOS",
    "EVAL_TIMES",
    "FORBIDDEN",
    "FRESH_CONFIRMATION_AUTHORIZED",
    "FROZEN_FOR_FRESH_CONFIRMATION",
    "HOLD_STEPS",
    "J15_RETAIN_ABS",
    "LARGE_CONFLICT_FRACTION",
    "LOW_STRENGTH",
    "MODEL_R1_AGGREGATION_APPROVED",
    "PHASE",
    "PHASE4E_REPORT",
    "PHASE4E_STATUS",
    "PLANNER_READY",
    "REPORT",
    "REPEATS",
    "SEED_PERMUTATIONS",
    "VALUE_SEEDS",
    "WARMUP",
    "assert_consumed",
    "choose_gate",
    "dump_protocol",
]
