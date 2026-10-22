"""Phase-5N constants. Only the joint assignment head is trained."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5n_joint_planner"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5N_LearnableJointPlanner_CN.md"
LABELS = OUTPUT / "assignment_labels.pt"
SCENARIOS = (310100, 310101)
TRAIN_SCENARIO = 310100
TEST_SCENARIO = 310101
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
EPOCHS = 40
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
PATIENCE = 8
LABEL_VERSION = 2


def assert_phase5n_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5n refused deployment holdout")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-5n refused seed {value}")


__all__ = [
    "DEPLOYMENT_VERIFIED",
    "EPOCHS",
    "LABELS",
    "LABEL_VERSION",
    "LEARNING_RATE",
    "OUTPUT",
    "PATIENCE",
    "PLANNER_READY",
    "REPORT",
    "SCENARIOS",
    "TEST_SCENARIO",
    "TRAIN_SCENARIO",
    "WEIGHT_DECAY",
    "assert_phase5n_seed",
]
