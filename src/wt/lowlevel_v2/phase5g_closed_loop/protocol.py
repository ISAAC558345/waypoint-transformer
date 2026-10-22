"""Phase-5G uses scenarios that were not in the utility-HGNN training split."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5g_closed_loop"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5G_HGNN_TrajectoryUtilityClosedLoopValidation_CN.md"
UTILITY_CKPT = ROOT / "results/phase5f_utility_rank" / "hgnn_utility_rank.pt"
MAPPO_CKPT = ROOT / "results/phase5_hgnn_urgency" / "checkpoints" / "mappo_original.pt"

# 310100 trained the utility HGNN. 310101 was its test split.
# 310102 and 310103 already exist as isolated resets and were not used for that training.
SCENARIOS = (310102, 310103)
DECISION_TIMES = (0, 15, 30)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False


def assert_phase5g_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5g refused deployment holdout")
    if value == 310100:
        raise RuntimeError("phase-5g refused the utility-HGNN training scenario")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-5g refused seed {value}")


__all__ = [
    "DECISION_TIMES",
    "DEPLOYMENT_VERIFIED",
    "MAPPO_CKPT",
    "OUTPUT",
    "PLANNER_READY",
    "REPORT",
    "SCENARIOS",
    "UTILITY_CKPT",
    "assert_phase5g_seed",
]
