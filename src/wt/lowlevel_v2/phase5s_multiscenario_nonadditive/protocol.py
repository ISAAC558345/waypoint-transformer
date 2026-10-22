"""Phase-5S constants. These are fixed before confirmation scenarios are opened."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5s_multiscenario_nonadditive"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5S_MultiScenarioNonAdditiveJointScorer_CN.md"
CHECKPOINTS = OUTPUT / "checkpoints"
PHASE5O_LABELS = ROOT / "results/phase5o_executor_joint_utility_audit" / "phase5o_executor_conditioned_joint_utility.csv"
PHASE5R_LABELS = ROOT / "results/phase5r_nonadditive_generalization" / "phase5r_ground_truth_jmappo.csv"
PHASE5R_CHECKPOINTS = ROOT / "results/phase5r_nonadditive_generalization" / "checkpoints"
TRAIN_SCENARIOS = tuple(range(310100, 310110))
CONFIRM_SCENARIOS = tuple(range(310110, 310118))
TIMES = (0, 15, 30)
SEEDS = (0, 1, 2, 3, 4)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
EPOCHS = 10
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP = 1.0
PAYLOAD_BLOCK_BITS = 1.0e6
MODEL_POOL_WIDTH = 3
EXPANDED_WIDTH = 5
COVERAGE_EPS = 1.0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
WORKERS = 4
TIE_ABS_BITS = PAYLOAD_BLOCK_BITS
CLOSE_ORACLE_BITS = PAYLOAD_BLOCK_BITS


def assert_not_holdout(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5s refused deployment holdout")


def assert_train_seed(seed: int) -> None:
    assert_not_holdout(seed)
    if int(seed) not in TRAIN_SCENARIOS:
        raise RuntimeError(f"phase-5s training refused seed {seed}")


def assert_confirm_seed(seed: int) -> None:
    assert_not_holdout(seed)
    if int(seed) not in CONFIRM_SCENARIOS:
        raise RuntimeError(f"phase-5s confirmation refused seed {seed}")


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CHECKPOINTS",
    "CLOSE_ORACLE_BITS",
    "CONFIRM_SCENARIOS",
    "COVERAGE_EPS",
    "DEPLOYMENT_VERIFIED",
    "EPOCHS",
    "EXPANDED_WIDTH",
    "GRAD_CLIP",
    "LEARNING_RATE",
    "MODEL_POOL_WIDTH",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PHASE5O_LABELS",
    "PHASE5R_CHECKPOINTS",
    "PHASE5R_LABELS",
    "PLANNER_READY",
    "REPORT",
    "SEEDS",
    "TIE_ABS_BITS",
    "TIMES",
    "TRAIN_SCENARIOS",
    "WEIGHT_DECAY",
    "WORKERS",
    "assert_confirm_seed",
    "assert_not_holdout",
    "assert_train_seed",
]
