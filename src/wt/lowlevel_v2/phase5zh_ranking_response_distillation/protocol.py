"""Phase-5ZH thresholds. Locked before confirmation scores are read."""
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
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import RESPONSE_FEATURES

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
CHECKPOINTS = OUTPUT / "checkpoints"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZH_StateConditionalValueRankingResponseDistillation_CN.md"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
FROZEN_FOR_FRESH_CONFIRMATION = False
LAMBDA_RANK = 0.2
# Model-R1 is trained on J15 / PAYLOAD_BLOCK_BITS, so one block is 1 value unit.
VALUE_TEMPERATURE = 1.0
VALUE_TIE = 1.0e-3
TOP_OUTSIDE = 5
TOP_INTERNAL = 10
RANDOM_PAIRS = 64
PAIR_RNG_SEED = 51014
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
RETENTION_BAR = 0.70
MIN_SEEDS_UP = 4
TIE_BITS = 1.0
ALIGN_ABS = 1.0e-3
TOKEN_NORM_ABS = 1.0e-4
MAE_RATIO_MAX = 3.0
SINR_RATE_SPEARMAN_MIN = 0.5
CHANNEL_STD_MIN = 1.0e-4
Z_EXTREME = 8.0
EXTREME_FRACTION = 0.01
SCORE_COLLAPSE_STD = 1.0e-3
CASE_SCENARIO = 310114
CASE_TIME = 15
CASE_PREDICTED = "2;5;-1;4;-1;-1"
CASE_OPTIMAL = "-1;4;2;5;-1;-1"
LATENCY_WARMUP = 2
LATENCY_REPEATS = 20


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127 or 310118 <= value <= 310125:
        raise RuntimeError(f"scenario {value} is outside this phase")
    if value not in TRAIN_SCENARIOS and value not in CONFIRM_SCENARIOS:
        raise RuntimeError(f"scenario {value} is outside this phase")


__all__ = [
    "ALIGN_ABS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CASE_OPTIMAL",
    "CASE_PREDICTED",
    "CASE_SCENARIO",
    "CASE_TIME",
    "CHANNEL_STD_MIN",
    "CHECKPOINTS",
    "CONFIRM_SCENARIOS",
    "DEPLOYMENT_VERIFIED",
    "EPOCHS",
    "EXTREME_FRACTION",
    "FROZEN_FOR_FRESH_CONFIRMATION",
    "GRAD_CLIP",
    "LAMBDA_RANK",
    "LATENCY_REPEATS",
    "LATENCY_WARMUP",
    "LEARNING_RATE",
    "MAE_RATIO_MAX",
    "MIN_SEEDS_UP",
    "OUTPUT",
    "PAIR_RNG_SEED",
    "PAYLOAD_BLOCK_BITS",
    "PLANNER_READY",
    "RANDOM_PAIRS",
    "REPORT",
    "RESPONSE_FEATURES",
    "RETENTION_BAR",
    "ROOT",
    "SCORE_COLLAPSE_STD",
    "SEEDS",
    "SINR_RATE_SPEARMAN_MIN",
    "TIE_BITS",
    "TIMES",
    "TOKEN_NORM_ABS",
    "TOP_INTERNAL",
    "TOP_OUTSIDE",
    "TRAIN_SCENARIOS",
    "VALUE_TEMPERATURE",
    "VALUE_TIE",
    "WEIGHT_DECAY",
    "Z_EXTREME",
    "assert_consumed",
]
