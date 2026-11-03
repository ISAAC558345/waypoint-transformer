"""Registered E19 limits. Earlier-stage steps are not available here."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import HOLD, TRAIN_POOL
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.protocol import (
    FORMAL_ACTOR_SHA256,
    HORIZON,
    PROCESS_SEED,
    PROCESS_SEED_ROLE,
)
from wt.l3b.config import L3BConfig

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_assignment_rule"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Fixed_Tag_Rule_Diagnosis_Report_CN.md"
E17_OUTPUT = ROOT / "results" / "phase6i_priority_tag_assignment_feasibility"
E18_OUTPUT = ROOT / "results" / "phase6i_priority_tag_assignment_coverage"
INSTRUCTION_COMMIT = "26e35e9abcf2d79a7e5d3b932387af8d32d88431"
SOURCE_COMMIT = "c6aac5b71f23a33653e21db1f2d558f212167be3"
E18_IMPLEMENTATION_COMMIT = "f79ad10139eacbb298b4a306a2b917057b32ef6b"
E18_SELECTION_COMMIT = "83fe1476f0dd109e49425db67dfa50ead5638445"

E17_GATE = "ASSIGNMENT_FEASIBILITY_COMPLETE"
E17_OPPORTUNITY = "NOT_OBSERVED_IN_REGISTERED_POOL"
E18_GATE = "ASSIGNMENT_COVERAGE_DIAG_COMPLETE"
E18_OPPORTUNITY = "REPLICATED_LOCAL_OPPORTUNITY"
E15_GATE = "CLIP_GROUP_BENCHMARK_INCONCLUSIVE"
E16_GATE = "CLIP_GROUP_LINE_CLOSED_UNADOPTED"

SCHEMA_VERSION = "fixed_tag_rule_diag_v1"
GATE_COMPLETE = "FIXED_TAG_RULE_DIAG_COMPLETE"
GATE_BLOCKED = "FIXED_TAG_RULE_DIAG_BLOCKED"
MECHANISM_NO_EFFECT = "NO_TAG_DECISION_EFFECT"
MECHANISM_REPLICATED = "REPLICATED_GUARDED_TAG_SIGNAL"
MECHANISM_LOCAL = "LOCAL_OR_UNGUARDED_TAG_EFFECT"
MECHANISM_ABSENT = "NO_GUARDED_TAG_SIGNAL"

SCENARIOS = tuple(range(321008, 321024))
PRIOR_DIAGNOSIS_SCENARIOS = tuple(range(321000, 321008))
DECISION_TIME = 15
HOLD_STEPS = int(HOLD)
TAG_WEIGHT = 0.001
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 9_400_019
DUST = float(DUST_BITS)
POOL_RULE = (("G1_RATE", 5),)

MAX_REFERENCE_STEPS = len(SCENARIOS) * int(HORIZON)
MAX_CANDIDATE_STEPS = len(SCENARIOS) * 3 * HOLD_STEPS
MAX_REPEAT_STEPS = len(SCENARIOS) * HOLD_STEPS
MAX_SIMULATOR_STEPS = MAX_REFERENCE_STEPS + MAX_CANDIDATE_STEPS + MAX_REPEAT_STEPS


def snapshot_file(scenario_id: int) -> Path:
    return OUTPUT / "snapshots" / f"scenario_{int(scenario_id)}_t_{int(DECISION_TIME)}.json"


def registered_flags() -> dict:
    return {
        "release_eligible": False,
        "merged": False,
        "adopted": False,
        "performance_pass": False,
        "new_formal_loader": False,
        "training_updates": 0,
        "parameter_updates": 0,
        "adam_updates": 0,
        "deployment_verified": False,
        "not_a_scheme_c_result": True,
        "not_a_closed_loop_policy": True,
        "e17_rewritten": False,
        "e18_rewritten": False,
    }


def scenario_contract() -> list[str]:
    problems = []
    if SCENARIOS != tuple(range(321008, 321024)) or len(SCENARIOS) != 16:
        problems.append("scenario list is not the registered sixteen training ids")
    if set(SCENARIOS) & set(PRIOR_DIAGNOSIS_SCENARIOS):
        problems.append("scenario list overlaps the E17/E18 diagnosis ids")
    if any(int(item) not in TRAIN_POOL for item in SCENARIOS):
        problems.append("a registered scenario is outside the training pool")
    if int(DECISION_TIME) != 15 or int(HOLD_STEPS) != 15 or int(HORIZON) != 48:
        problems.append("window is not the registered t=15 hold of 15 inside a 48-step episode")
    if float(TAG_WEIGHT) != 0.001:
        problems.append("tag weight is not the registered 0.001")
    if int(MAX_SIMULATOR_STEPS) != 1728:
        problems.append("simulator ceiling is not 1728 steps")
    if int(BOOTSTRAP_DRAWS) != 10000 or int(BOOTSTRAP_SEED) != 9400019:
        problems.append("bootstrap contract changed")
    if float(L3BConfig.episode_horizon) != float(HORIZON):
        problems.append("environment horizon does not match the registered episode length")
    return problems


class StepBudget:
    """Own 1728-step ceiling. Unused steps do not buy more scenarios or weights."""

    def __init__(self, reference: int = 0, candidate: int = 0, repeat: int = 0) -> None:
        self.reference = int(reference)
        self.candidate = int(candidate)
        self.repeat = int(repeat)

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
        return int(self.reference + self.candidate + self.repeat)

    def as_dict(self) -> dict:
        return {
            "reference_steps": int(self.reference),
            "candidate_steps": int(self.candidate),
            "repeat_steps": int(self.repeat),
            "fixture_steps": 0,
            "total_steps": int(self.total),
            "max_reference_steps": int(MAX_REFERENCE_STEPS),
            "max_candidate_steps": int(MAX_CANDIDATE_STEPS),
            "max_repeat_steps": int(MAX_REPEAT_STEPS),
            "max_steps": int(MAX_SIMULATOR_STEPS),
            "unused_steps_are_not_extra_room": True,
            "training_updates": 0,
            "parameter_updates": 0,
        }
