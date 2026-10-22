"""Phase-5O audit constants. No network is trained."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5o_executor_joint_utility_audit"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5O_ExecutorConditionedJointUtilityAudit_CN.md"
PHASE5N_LABELS = ROOT / "results/phase5n_joint_planner" / "assignment_labels.pt"
PHASE5N_SUMMARY = ROOT / "results/phase5n_joint_planner" / "phase5n_summary.json"
CACHE = OUTPUT / "phase5o_rollout_cache.pt"
SCENARIOS = (310100, 310101)
DECISION_TIMES = (0, 15, 30)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
POOL_SIZE = 120
HOLD_STEPS = 15
PAYLOAD_BLOCK_BITS = 1.0e6
NOISE_BITS = 1.0
TRACE_STEPS = (1, 3, 5, 10, 15)
PAIR_LIMIT = 10
CACHE_VERSION = 1
# Fixed before the sweep. A gap below one delivered block is not discrimination.
MIN_DISCRIMINATING_STATES = 3
POOR_SPEARMAN = 0.5
POOR_REGRET = 0.5
POOR_TOP1 = 0.5
COVERAGE_STATES = 4


def assert_phase5o_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5o refused deployment holdout")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-5o refused seed {value}")


__all__ = [
    "CACHE",
    "CACHE_VERSION",
    "COVERAGE_STATES",
    "DECISION_TIMES",
    "DEPLOYMENT_VERIFIED",
    "HOLD_STEPS",
    "MIN_DISCRIMINATING_STATES",
    "NOISE_BITS",
    "OUTPUT",
    "PAIR_LIMIT",
    "PAYLOAD_BLOCK_BITS",
    "PHASE5N_LABELS",
    "PHASE5N_SUMMARY",
    "PLANNER_READY",
    "POOR_REGRET",
    "POOR_SPEARMAN",
    "POOR_TOP1",
    "POOL_SIZE",
    "REPORT",
    "SCENARIOS",
    "TRACE_STEPS",
    "assert_phase5o_seed",
]
