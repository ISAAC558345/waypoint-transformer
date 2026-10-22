"""Phase-5K constants. Decoder heads only; the encoder stays frozen."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5k_joint_decoder"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5K_JointAssignmentDecoderRedesign_CN.md"
CACHE = OUTPUT / "assignment_cache.pt"
SCENARIOS = (310100, 310101)
TRAIN_SCENARIO = 310100
TEST_SCENARIO = 310101
TRAIN_TIMES = (0, 15)
VAL_TIMES = (30,)
TEST_TIMES = (0, 15, 30)
DECISION_TIMES = (0, 15, 30)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
SEARCH_LIMIT = 500
EPOCHS = 40
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
PATIENCE = 8
SINKHORN_STEPS = 20
CACHE_VERSION = 1


def assert_phase5k_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5k refused deployment holdout")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-5k refused seed {value}")


def split_name(scenario: int, moment: int) -> str:
    if int(scenario) == int(TEST_SCENARIO):
        if int(moment) not in TEST_TIMES:
            raise RuntimeError(f"phase-5k refused test time {moment}")
        return "test"
    if int(scenario) != int(TRAIN_SCENARIO):
        raise RuntimeError(f"phase-5k refused seed {scenario}")
    if int(moment) in TRAIN_TIMES:
        return "train"
    if int(moment) in VAL_TIMES:
        return "val"
    raise RuntimeError(f"phase-5k refused time {moment}")


__all__ = [
    "CACHE",
    "CACHE_VERSION",
    "DECISION_TIMES",
    "DEPLOYMENT_VERIFIED",
    "EPOCHS",
    "LEARNING_RATE",
    "OUTPUT",
    "PATIENCE",
    "PLANNER_READY",
    "REPORT",
    "SCENARIOS",
    "SEARCH_LIMIT",
    "SINKHORN_STEPS",
    "TEST_SCENARIO",
    "TEST_TIMES",
    "TRAIN_SCENARIO",
    "TRAIN_TIMES",
    "VAL_TIMES",
    "WEIGHT_DECAY",
    "assert_phase5k_seed",
    "split_name",
]
