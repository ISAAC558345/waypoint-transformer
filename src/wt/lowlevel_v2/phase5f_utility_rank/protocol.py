"""Phase-5F constants. The test scenario is isolated from training."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5f_utility_rank"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5F_HGNN_TrajectoryUtilityRetraining_CN.md"
LABELS = ROOT / "results/phase5e_utility_dataset" / "trajectory_utility_dataset.csv"
OLD_HGNN = ROOT / "results/phase5_hgnn_urgency" / "checkpoints" / "hgnn_orderfree.pt"

TRAIN_SCENARIO = 310100
TEST_SCENARIO = 310101
TRAIN_TIMES = (0, 15)
VAL_TIMES = (30,)
TEST_TIMES = (0, 15, 30)
EPOCHS = 20
LEARNING_RATE = 1.0e-3
PATIENCE = 6
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False

MODEL_INPUT_KEYS = ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu")
FORBIDDEN_INPUTS = ("payload_gain", "utility_score", "utility_rank", "trajectory")


def assert_phase5f_seed(seed: int) -> None:
    value = int(seed)
    if value not in {TRAIN_SCENARIO, TEST_SCENARIO}:
        raise RuntimeError(f"phase-5f refused seed {value}")
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5f refused deployment holdout")


__all__ = [
    "DEPLOYMENT_VERIFIED",
    "EPOCHS",
    "FORBIDDEN_INPUTS",
    "LABELS",
    "LEARNING_RATE",
    "MODEL_INPUT_KEYS",
    "OLD_HGNN",
    "OUTPUT",
    "PATIENCE",
    "PLANNER_READY",
    "REPORT",
    "TEST_SCENARIO",
    "TEST_TIMES",
    "TRAIN_SCENARIO",
    "TRAIN_TIMES",
    "VAL_TIMES",
    "assert_phase5f_seed",
]
