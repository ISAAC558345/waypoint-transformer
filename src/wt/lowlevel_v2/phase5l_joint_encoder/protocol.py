"""Phase-5L constants. The encoder is trained; the matching decoder is not."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5l_joint_encoder"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5L_JointUtilityHGNNTraining_CN.md"
SCENARIOS = (310100, 310101)
TRAIN_SCENARIO = 310100
TEST_SCENARIO = 310101
DECISION_TIMES = (0, 15, 30)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
EPOCHS = 40
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
PATIENCE = 8
WIDTH = 128


def assert_phase5l_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5l refused deployment holdout")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-5l refused seed {value}")


__all__ = [
    "DECISION_TIMES",
    "DEPLOYMENT_VERIFIED",
    "EPOCHS",
    "LEARNING_RATE",
    "OUTPUT",
    "PATIENCE",
    "PLANNER_READY",
    "REPORT",
    "SCENARIOS",
    "TEST_SCENARIO",
    "TRAIN_SCENARIO",
    "WEIGHT_DECAY",
    "WIDTH",
    "assert_phase5l_seed",
]
