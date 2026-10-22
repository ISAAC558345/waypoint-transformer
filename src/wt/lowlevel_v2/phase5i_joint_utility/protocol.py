"""Phase-5I audit constants. No training."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5i_joint_utility"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5I_JointAssignmentUtilityAudit_CN.md"
SCENARIOS = (310102, 310103)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
SEARCH_LIMIT = 500


def assert_phase5i_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5i refused deployment holdout")
    if value == 310100:
        raise RuntimeError("phase-5i refused the utility-HGNN training scenario")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-5i refused seed {value}")


__all__ = [
    "DEPLOYMENT_VERIFIED",
    "OUTPUT",
    "PLANNER_READY",
    "REPORT",
    "SCENARIOS",
    "SEARCH_LIMIT",
    "assert_phase5i_seed",
]
