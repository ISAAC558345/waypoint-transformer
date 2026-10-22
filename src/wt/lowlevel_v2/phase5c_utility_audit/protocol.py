"""Phase-5C audit constants. No training."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5c_utility_audit"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5C_HighLevelUtilityRedesignAudit_CN.md"
AUDIT_SEEDS = (310100, 310101)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False


def assert_audit_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5c refused deployment holdout")
    protected = (
        (220000, 220064),
        (221000, 221016),
        (222000, 222064),
        (223000, 223064),
        (104000, 104064),
        (105000, 105064),
        (106000, 106064),
    )
    for low, high in protected:
        if low <= value < high:
            raise RuntimeError(f"phase-5c refused protected seed {value}")


__all__ = [
    "AUDIT_SEEDS",
    "DEPLOYMENT_VERIFIED",
    "OUTPUT",
    "PLANNER_READY",
    "REPORT",
    "assert_audit_seed",
]
