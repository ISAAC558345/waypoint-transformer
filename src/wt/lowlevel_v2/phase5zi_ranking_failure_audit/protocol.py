"""Phase-5ZI thresholds. Locked before the saved scores are summarized."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZI_RankingLossFailureAndTrueH1Feasibility_CN.md"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
FROZEN_FOR_FRESH_CONFIRMATION = False
DIAGNOSTIC_SEED = 51015
TEACHER_SPEARMAN_MIN = 0.70
TEACHER_TOP1_MIN = 0.70
EASY_TEACHER_GAP = 1.0
EASY_LOSS_SHARE = 0.80
HARD_LOSS_SHARE = 0.20
VALUE_STD_LOW = 0.1
VALUE_STD_HIGH = 10.0
SATURATION_LOW = 0.05
SATURATION_HIGH = 0.95


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127 or 310118 <= value <= 310125:
        raise RuntimeError(f"scenario {value} is outside this phase")


__all__ = [
    "DEPLOYMENT_VERIFIED",
    "DIAGNOSTIC_SEED",
    "EASY_LOSS_SHARE",
    "EASY_TEACHER_GAP",
    "FROZEN_FOR_FRESH_CONFIRMATION",
    "HARD_LOSS_SHARE",
    "OUTPUT",
    "PLANNER_READY",
    "REPORT",
    "ROOT",
    "SATURATION_HIGH",
    "SATURATION_LOW",
    "TEACHER_SPEARMAN_MIN",
    "TEACHER_TOP1_MIN",
    "VALUE_STD_HIGH",
    "VALUE_STD_LOW",
    "assert_consumed",
]
