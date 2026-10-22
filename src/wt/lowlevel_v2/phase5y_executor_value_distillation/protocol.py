"""Phase-5Y thresholds. Locked before 310110-310117 are scored."""
from __future__ import annotations

from pathlib import Path

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
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import FINAL_REL_DIM

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5y_executor_value_distillation"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5Y_ExecutorConditionedJointValueDistillation_CN.md"
CHECKPOINTS = OUTPUT / "checkpoints"
PHASE5X_ROLLOUTS = ROOT / "results/phase5x_executor_lookahead" / "phase5x_executor_rollouts.csv"
PHASE5X_RUNTIME = ROOT / "results/phase5x_executor_lookahead" / "phase5x_runtime.csv"
PHASE5W_RESIDUAL = ROOT / "results/phase5w_residual_mechanism_audit" / "phase5w_residual_case_hybrids.csv"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
HUBER_DELTA = 1.0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
CASE_SCENARIO = 310114
CASE_TIME = 15
CASE_PREDICTED = "2;5;-1;4;-1;-1"
CASE_OPTIMUM = "-1;4;2;5;-1;-1"
TOP3_FLOOR = 0.75
REGRET_CEILING = 0.10
MIN_SCENARIO_UP = 5
MIN_SEED_UP = 4
CASE_TOP_RANK = 3
LARGE_DROP_BITS = PAYLOAD_BLOCK_BITS
VALUE_GAP_BLOCKS = 0.1
COSINE_LIMIT = 0.99
COLLISION_FRACTION = 0.25
REGRESSION_R2 = 0.5
REGRESSION_MAE = 1.0
RECOVERY_EPS = 1.0
REFERENCE_HORIZONS = (5, 8)
FORBIDDEN_INPUTS = (
    "payload_h",
    "q_exec_h",
    "mean_sinr_h",
    "progress_end_sum",
    "completed_blocks_h",
    "j15",
    "future_position",
    "future_action",
)


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5y refused deployment holdout")
    if value not in TRAIN_SCENARIOS and value not in CONFIRM_SCENARIOS:
        raise RuntimeError(f"phase-5y refused seed {value}")


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CASE_OPTIMUM",
    "CASE_PREDICTED",
    "CASE_SCENARIO",
    "CASE_TIME",
    "CASE_TOP_RANK",
    "CHECKPOINTS",
    "COLLISION_FRACTION",
    "CONFIRM_SCENARIOS",
    "COSINE_LIMIT",
    "DEPLOYMENT_VERIFIED",
    "EPOCHS",
    "FINAL_REL_DIM",
    "FORBIDDEN_INPUTS",
    "GRAD_CLIP",
    "HUBER_DELTA",
    "LARGE_DROP_BITS",
    "LEARNING_RATE",
    "MIN_SCENARIO_UP",
    "MIN_SEED_UP",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PHASE5W_RESIDUAL",
    "PHASE5X_ROLLOUTS",
    "PHASE5X_RUNTIME",
    "PLANNER_READY",
    "RECOVERY_EPS",
    "REFERENCE_HORIZONS",
    "REGRESSION_MAE",
    "REGRESSION_R2",
    "REGRET_CEILING",
    "REPORT",
    "SEEDS",
    "TIMES",
    "TOP3_FLOOR",
    "TRAIN_SCENARIOS",
    "VALUE_GAP_BLOCKS",
    "WEIGHT_DECAY",
    "assert_consumed",
]
