"""Phase-5ZB thresholds. Locked before any cross-moment statistic is scored."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import PAYLOAD_BLOCK_BITS, TRAIN_SCENARIOS

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5zb_cross_moment_audit"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZB_SecondOrderUAVTargetCorrespondenceAudit_CN.md"
PHASE5Z_SWAPS = ROOT / "results/phase5z_representation_collision_audit" / "phase5z_swap_pair_collision.csv"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
CASE_SCENARIO = 310114
CASE_TIME = 15
CASE_PREDICTED = "2;5;-1;4;-1;-1"
CASE_OPTIMUM = "-1;4;2;5;-1;-1"
PERMUTATION_EPS = 1.0e-5
COSINE_LIMIT = 0.99
COLLISION_1 = 0.10
COLLISION_2 = 0.05
NN_MARGIN = 0.05
HELP_DROP = 0.20
NO_REDUCTION_EPS = 0.02
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
EPS = 1.0e-8


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127 or 310118 <= value <= 310125:
        raise RuntimeError("phase-5zb refused held-out scenarios")
    if value not in TRAIN_SCENARIOS and value != CASE_SCENARIO:
        raise RuntimeError(f"phase-5zb refused seed {value}")


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CASE_OPTIMUM",
    "CASE_PREDICTED",
    "CASE_SCENARIO",
    "CASE_TIME",
    "COLLISION_1",
    "COLLISION_2",
    "COSINE_LIMIT",
    "DEPLOYMENT_VERIFIED",
    "EPS",
    "HELP_DROP",
    "NN_MARGIN",
    "NO_REDUCTION_EPS",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PERMUTATION_EPS",
    "PHASE5Z_SWAPS",
    "PLANNER_READY",
    "REPORT",
    "TRAIN_SCENARIOS",
    "assert_consumed",
]
