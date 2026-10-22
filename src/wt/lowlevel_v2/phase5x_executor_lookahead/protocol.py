"""Phase-5X audit thresholds. No network is trained."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import (
    CONFIRM_SCENARIOS,
    PAYLOAD_BLOCK_BITS,
    SEEDS,
    TIMES,
    TRAIN_SCENARIOS,
)

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5x_executor_lookahead"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5X_FrozenExecutorShortHorizonLookahead_CN.md"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
HORIZONS = (1, 3, 5, 8, 10, 15)
SHORT_HORIZONS = (1, 3, 5, 8, 10)
CANDIDATE_HORIZONS = (1, 3, 5, 8, 10)
PROMISING_HORIZONS = (3, 5, 8)
LONG_HORIZONS = (8, 10)
CASE_SCENARIO = 310114
CASE_TIME = 15
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
TOP3_GAIN = 0.05
REGRET_DROP = 0.05
SATURATION_TOP3 = 0.02
SATURATION_REGRET = 0.02
RESIDUAL_FRACTION = 0.70
CASE_TOP_RANK = 3
SINR_GAP = 1.0
PROGRESS_GAP = 0.05


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5x refused deployment holdout")
    if value not in TRAIN_SCENARIOS and value not in CONFIRM_SCENARIOS:
        raise RuntimeError(f"phase-5x refused seed {value}")


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CANDIDATE_HORIZONS",
    "CASE_SCENARIO",
    "CASE_TIME",
    "CASE_TOP_RANK",
    "CONFIRM_SCENARIOS",
    "DEPLOYMENT_VERIFIED",
    "HORIZONS",
    "LONG_HORIZONS",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PLANNER_READY",
    "PROGRESS_GAP",
    "PROMISING_HORIZONS",
    "REGRET_DROP",
    "REPORT",
    "RESIDUAL_FRACTION",
    "SATURATION_REGRET",
    "SATURATION_TOP3",
    "SEEDS",
    "SHORT_HORIZONS",
    "SINR_GAP",
    "TIMES",
    "TOP3_GAIN",
    "TRAIN_SCENARIOS",
    "assert_consumed",
]
