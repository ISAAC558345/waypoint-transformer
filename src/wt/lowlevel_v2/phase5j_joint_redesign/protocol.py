"""Phase-5J constants. Existing Phase-5E scenarios only."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5j_joint_redesign"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5J_JointUtilityHGNNRedesignAudit_CN.md"
SCENARIOS = (310100, 310101)
DECISION_TIMES = (0, 15, 30)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
SEARCH_LIMIT = 500
LAMBDA_CONFLICT = 1.0


def assert_phase5j_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5j refused deployment holdout")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-5j refused seed {value}")


__all__ = [
    "DECISION_TIMES",
    "DEPLOYMENT_VERIFIED",
    "LAMBDA_CONFLICT",
    "OUTPUT",
    "PLANNER_READY",
    "REPORT",
    "SCENARIOS",
    "SEARCH_LIMIT",
    "assert_phase5j_seed",
]
