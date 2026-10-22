"""Phase-5H audit constants. No training."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5h_execution_audit"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5H_LowLevelExecutionCompatibilityAudit_CN.md"
SCENARIOS = (310102, 310103)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
BLOCK_BITS = 1.0e6


def assert_phase5h_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5h refused deployment holdout")
    if value == 310100:
        raise RuntimeError("phase-5h refused the utility-HGNN training scenario")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-5h refused seed {value}")


__all__ = [
    "BLOCK_BITS",
    "DEPLOYMENT_VERIFIED",
    "OUTPUT",
    "PLANNER_READY",
    "REPORT",
    "SCENARIOS",
    "assert_phase5h_seed",
]
