"""Phase-5P constants. Only the supervision values change."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5p_executor_joint_planner"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5P_ExecutorConditionedJointPlannerRetraining_CN.md"
PHASE5O_LABELS = ROOT / "results/phase5o_executor_joint_utility_audit" / "phase5o_executor_conditioned_joint_utility.csv"
SCENARIOS = (310100, 310101)
TRAIN_SCENARIO = 310100
DEV_SCENARIO = 310101
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 10
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP = 1.0
PAYLOAD_BLOCK_BITS = 1.0e6
POOL_SIZE = 120


def assert_phase5p_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5p refused deployment holdout")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-5p refused seed {value}")


__all__ = [
    "DEV_SCENARIO",
    "EPOCHS",
    "GRAD_CLIP",
    "LEARNING_RATE",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PHASE5O_LABELS",
    "PLANNER_READY",
    "POOL_SIZE",
    "REPORT",
    "SCENARIOS",
    "SEEDS",
    "TRAIN_SCENARIO",
    "WEIGHT_DECAY",
    "assert_phase5p_seed",
]
