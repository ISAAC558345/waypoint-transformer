"""Registered E18 limits. Unused E17 steps are not extra room, and these values stay fixed."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.protocol import (
    ASSIGNMENT_SEED_BASE,
    BIT_TOL,
    BUCKET_TOL,
    CANDIDATE_LIMIT,
    DECISION_TIMES,
    E15_GATE,
    E16_GATE,
    EXCLUDED_TAIL_TIME,
    FORMAL_ACTOR_SHA256,
    GENERATOR,
    HOLD_STEPS,
    HORIZON,
    KNOWN_TAG_GAIN_BITS,
    MAX_CANDIDATE_STEPS,
    MAX_REFERENCE_STEPS,
    MAX_REPEAT_STEPS,
    MAX_SIMULATOR_STEPS,
    PATH_TOL_M,
    PAYLOAD_FLOOR_BITS,
    POOL_K,
    PROCESS_SEED,
    PROCESS_SEED_ROLE,
    SCENARIOS,
)

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_assignment_coverage"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Assignment_Coverage_Diagnosis_Report_CN.md"
E17_OUTPUT = ROOT / "results" / "phase6i_priority_tag_assignment_feasibility"
INSTRUCTION_COMMIT = "2f2b72880df80183ac77d6875f9d69fbb761a02d"
SOURCE_COMMIT = "87509424b6b2119b422146ff0a64ec98c0d2656b"
E17_RESULT_COMMIT = "3465a926512fd0bdb485455ee73240ddd010cd1f"
E17_GATE = "ASSIGNMENT_FEASIBILITY_COMPLETE"
E17_OPPORTUNITY = "NOT_OBSERVED_IN_REGISTERED_POOL"

SCHEMA_VERSION = "assignment_coverage_diag_v1"
GATE_COMPLETE = "ASSIGNMENT_COVERAGE_DIAG_COMPLETE"
GATE_BLOCKED = "ASSIGNMENT_COVERAGE_DIAG_BLOCKED"
OPPORTUNITY_REPLICATED = "REPLICATED_LOCAL_OPPORTUNITY"
OPPORTUNITY_LOCAL = "LOCAL_OR_CONSTRAINED_OPPORTUNITY"
OPPORTUNITY_ABSENT = "NOT_OBSERVED_UNDER_MATCHED_MASK"
OPPORTUNITY_THIN = "INSUFFICIENT_MATCHED_ALTERNATIVES"

ALTERNATIVE_CAP = 4
CANDIDATE_SEED_BASE = 9_300_000
EXPECTED_SNAPSHOTS = len(SCENARIOS) * len(DECISION_TIMES)


def snapshot_file(scenario_id: int, decision_time: int) -> Path:
    return OUTPUT / "snapshots" / f"scenario_{int(scenario_id)}_t_{int(decision_time)}.json"


def candidate_seed(scenario_id: int, decision_time: int) -> int:
    """Per-snapshot selector seed. It is not an environment or assignment seed."""
    return int(CANDIDATE_SEED_BASE) + 100 * int(scenario_id) + int(decision_time)


def registered_flags() -> dict:
    return {
        "release_eligible": False,
        "merged": False,
        "adopted": False,
        "new_formal_loader": False,
        "training_updates": 0,
        "parameter_updates": 0,
        "adam_updates": 0,
        "deployment_verified": False,
        "e17_rewritten": False,
        "not_a_scheme_c_result": True,
        "not_an_e17_rewrite": True,
        "hindsight_diagnostic": True,
        "deployable_policy": False,
    }


class StepBudget:
    """Own 2544-step ceiling. A smaller sample does not release the unused steps."""

    def __init__(self, reference: int = 0, candidate: int = 0, repeat: int = 0) -> None:
        self.reference = int(reference)
        self.candidate = int(candidate)
        self.repeat = int(repeat)
        self.fixture = 0

    def add(self, kind: str, count: int = 1) -> None:
        amount = int(count)
        if amount < 0:
            raise RuntimeError("step count cannot be negative")
        if kind == "reference":
            self.reference += amount
        elif kind == "candidate":
            self.candidate += amount
        elif kind == "repeat":
            self.repeat += amount
        elif kind == "fixture":
            self.fixture += amount
        else:
            raise RuntimeError(f"unknown step kind {kind}")
        if self.reference > int(MAX_REFERENCE_STEPS):
            raise RuntimeError("reference step cap was exceeded")
        if self.candidate > int(MAX_CANDIDATE_STEPS):
            raise RuntimeError("candidate step cap was exceeded")
        if self.repeat > int(MAX_REPEAT_STEPS):
            raise RuntimeError("repeat step cap was exceeded")
        if self.total > int(MAX_SIMULATOR_STEPS):
            raise RuntimeError("registered simulator step budget was exceeded")

    @property
    def total(self) -> int:
        return int(self.reference + self.candidate + self.repeat + self.fixture)

    def as_dict(self) -> dict:
        return {
            "reference_steps": int(self.reference),
            "candidate_steps": int(self.candidate),
            "repeat_steps": int(self.repeat),
            "fixture_steps": int(self.fixture),
            "total_steps": int(self.total),
            "max_reference_steps": int(MAX_REFERENCE_STEPS),
            "max_candidate_steps": int(MAX_CANDIDATE_STEPS),
            "max_repeat_steps": int(MAX_REPEAT_STEPS),
            "max_steps": int(MAX_SIMULATOR_STEPS),
            "unused_steps_are_not_extra_room": True,
            "training_updates": 0,
            "parameter_updates": 0,
        }
