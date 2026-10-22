"""Phase-5ZE thresholds. Locked before 310110-310117 are scored."""
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
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import RESPONSE_FEATURES, RESPONSE_GROUPS

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZE_OneStepExecutorResponseDistillation_CN.md"
CHECKPOINTS = OUTPUT / "checkpoints"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
HUBER_DELTA = 1.0
LOSS_WEIGHT_CONTINUOUS = 1.0
LOSS_WEIGHT_BINARY = 1.0
HIDDEN = 128
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
CASE_SCENARIO = 310114
CASE_TIME = 15
CASE_PREDICTED = "2;5;-1;4;-1;-1"
CASE_OPTIMAL = "-1;4;2;5;-1;-1"
TOP3_BAR = 0.85
REGRET_BAR = 0.12
MIN_SCENARIO_GE = 5
MAX_SCENARIO_DOWN = 1
MIN_SEED_UP = 4
RETENTION_BAR = 0.75
MAX_J15_DROP_VS_TRUE = 1.0e5
SPEEDUP_BAR = 2.0
PERMUTATION_ABS = 1.0e-5
TIE_BITS = 1.0
RUNTIME_REPEATS = 100
RUNTIME_WARMUP = 2
LABEL_MATCH_ABS = 1.0e-5
EDGE_MATCH_ABS = 1.0e-4
CONTINUOUS_FEATURES = (
    "sinr_1",
    "delta_sinr",
    "interference_1",
    "delta_interference",
    "service_progress_1",
    "delta_service_progress",
    "rate_1",
    "distance_to_target_1",
    "delta_distance_to_target",
    "radial_motion_1",
    "heading_alignment_1",
)
BINARY_FEATURES = (
    "assigned_in_range_1",
    "service_active_1",
    "same_target_after_step_1",
)


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127 or 310118 <= value <= 310125:
        raise RuntimeError(f"scenario {value} is outside this phase")
    if value not in TRAIN_SCENARIOS and value not in CONFIRM_SCENARIOS:
        raise RuntimeError(f"scenario {value} is outside this phase")


__all__ = [
    "BINARY_FEATURES",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CASE_OPTIMAL",
    "CASE_PREDICTED",
    "CASE_SCENARIO",
    "CASE_TIME",
    "EDGE_MATCH_ABS",
    "CHECKPOINTS",
    "CONFIRM_SCENARIOS",
    "CONTINUOUS_FEATURES",
    "EPOCHS",
    "GRAD_CLIP",
    "HIDDEN",
    "HUBER_DELTA",
    "LABEL_MATCH_ABS",
    "LEARNING_RATE",
    "LOSS_WEIGHT_BINARY",
    "LOSS_WEIGHT_CONTINUOUS",
    "MAX_J15_DROP_VS_TRUE",
    "MAX_SCENARIO_DOWN",
    "MIN_SCENARIO_GE",
    "MIN_SEED_UP",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PERMUTATION_ABS",
    "PLANNER_READY",
    "REGRET_BAR",
    "REPORT",
    "RESPONSE_FEATURES",
    "RESPONSE_GROUPS",
    "RETENTION_BAR",
    "ROOT",
    "RUNTIME_REPEATS",
    "RUNTIME_WARMUP",
    "SEEDS",
    "SPEEDUP_BAR",
    "TIE_BITS",
    "TIMES",
    "TOP3_BAR",
    "TRAIN_SCENARIOS",
    "WEIGHT_DECAY",
    "assert_consumed",
]
