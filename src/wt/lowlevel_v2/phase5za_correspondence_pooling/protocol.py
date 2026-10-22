"""Phase-5ZA thresholds. Locked before 310110-310117 are scored."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase5l_joint_encoder.protocol import WIDTH
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import (
    CONFIRM_SCENARIOS,
    EPOCHS,
    GRAD_CLIP,
    LEARNING_RATE,
    PAYLOAD_BLOCK_BITS,
    SEEDS,
    TIMES,
    TRAIN_SCENARIOS,
    WEIGHT_DECAY,
)
from wt.lowlevel_v2.phase5u_representation_repair.protocol import REL_DIM
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import FINAL_REL_DIM, TEMP_INPUT_DIM

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5za_correspondence_pooling"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZA_CorrespondencePreservingAssignmentPooling_CN.md"
CHECKPOINTS = OUTPUT / "checkpoints"
PHASE5Z_SWAPS = ROOT / "results/phase5z_representation_collision_audit" / "phase5z_swap_pair_collision.csv"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
HUBER_DELTA = 1.0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
CASE_SCENARIO = 310114
CASE_TIME = 15
CASE_PREDICTED = "2;5;-1;4;-1;-1"
CASE_OPTIMUM = "-1;4;2;5;-1;-1"
PERMUTATION_EPS = 1.0e-5
SWAP_COLLISION_CEILING = 0.10
COSINE_LIMIT = 0.99
TOP3_FLOOR = 0.75
REGRET_CEILING = 0.10
MIN_SCENARIO_UP = 5
MIN_SEED_UP = 4
CASE_TOP_RANK = 3
ASSIGNMENT_DIM = int(WIDTH) * 5
GLOBAL_DIM = int(WIDTH)
HEAD_INPUT_DIM = int(ASSIGNMENT_DIM) + int(GLOBAL_DIM) + int(FINAL_REL_DIM)


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5za refused deployment holdout")
    if value not in TRAIN_SCENARIOS and value not in CONFIRM_SCENARIOS:
        raise RuntimeError(f"phase-5za refused seed {value}")


__all__ = [
    "ASSIGNMENT_DIM",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CASE_OPTIMUM",
    "CASE_PREDICTED",
    "CASE_SCENARIO",
    "CASE_TIME",
    "CASE_TOP_RANK",
    "CHECKPOINTS",
    "CONFIRM_SCENARIOS",
    "COSINE_LIMIT",
    "DEPLOYMENT_VERIFIED",
    "EPOCHS",
    "FINAL_REL_DIM",
    "GLOBAL_DIM",
    "GRAD_CLIP",
    "HEAD_INPUT_DIM",
    "HUBER_DELTA",
    "LEARNING_RATE",
    "MIN_SCENARIO_UP",
    "MIN_SEED_UP",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PERMUTATION_EPS",
    "PHASE5Z_SWAPS",
    "PLANNER_READY",
    "REGRET_CEILING",
    "REL_DIM",
    "REPORT",
    "SEEDS",
    "SWAP_COLLISION_CEILING",
    "TEMP_INPUT_DIM",
    "TIMES",
    "TOP3_FLOOR",
    "TRAIN_SCENARIOS",
    "WEIGHT_DECAY",
    "WIDTH",
    "assert_consumed",
]
