"""Phase-5M constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5m_joint_planner"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5M_HierarchicalJointPlanner_CN.md"
SCENARIOS = (310100, 310101)
DECISION_TIMES = (0, 15, 30)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
TOP_K = 3
SEARCH_LIMIT = 500


def assert_phase5m_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5m refused deployment holdout")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-5m refused seed {value}")


__all__ = [
    "DECISION_TIMES",
    "DEPLOYMENT_VERIFIED",
    "OUTPUT",
    "PLANNER_READY",
    "REPORT",
    "SCENARIOS",
    "SEARCH_LIMIT",
    "TOP_K",
    "assert_phase5m_seed",
]
