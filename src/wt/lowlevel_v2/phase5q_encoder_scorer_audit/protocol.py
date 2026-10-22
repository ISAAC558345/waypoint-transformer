"""Phase-5Q constants. Thresholds are fixed before the run."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5q_encoder_scorer_audit"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5Q_EncoderVsAssignmentScorerAudit_CN.md"
PHASE5P_STATE = ROOT / "results/phase5p_executor_joint_planner" / "phase5p_state_metrics.csv"
PHASE5P_PREDICTIONS = ROOT / "results/phase5p_executor_joint_planner" / "phase5p_assignment_predictions.csv"
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
PHASE5P_RECORDED_LOSS = 0.6931
COLLAPSE_EPS = 1.0e-6
LOSS_DROP = 1.0e-3
CLEAR_PAYLOAD_BITS = PAYLOAD_BLOCK_BITS
R2_MARGIN = 0.05
SPEARMAN_MARGIN = 0.05
MODEL_A = "model_a"
MODEL_B = "model_b"
MODEL_C = "model_c"


def assert_phase5q_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5q refused deployment holdout")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-5q refused seed {value}")


__all__ = [
    "CLEAR_PAYLOAD_BITS",
    "COLLAPSE_EPS",
    "DEV_SCENARIO",
    "EPOCHS",
    "GRAD_CLIP",
    "LEARNING_RATE",
    "LOSS_DROP",
    "MODEL_A",
    "MODEL_B",
    "MODEL_C",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PHASE5P_PREDICTIONS",
    "PHASE5P_RECORDED_LOSS",
    "PHASE5P_STATE",
    "PLANNER_READY",
    "POOL_SIZE",
    "R2_MARGIN",
    "REPORT",
    "SCENARIOS",
    "SEEDS",
    "SPEARMAN_MARGIN",
    "TRAIN_SCENARIO",
    "WEIGHT_DECAY",
    "assert_phase5q_seed",
]
