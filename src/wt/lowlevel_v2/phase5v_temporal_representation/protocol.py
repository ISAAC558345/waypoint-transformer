"""Phase-5V thresholds. Fixed before 310110-310117 are scored."""
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
from wt.lowlevel_v2.phase5u_representation_repair.protocol import REL_DIM

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5v_temporal_representation"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5V_CandidateConditionedTemporalState_CN.md"
CHECKPOINTS = OUTPUT / "checkpoints"
PHASE5U_CHECKPOINTS = ROOT / "results/phase5u_representation_repair" / "checkpoints"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
MODEL_POOL_WIDTH = 3
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
CASE_SCENARIO = 310114
CASE_TIME = 15
PHASE5U_CASE_RANK = 47.0
T15_CLEAR_DROP = 2.0e5
T15_REGRESSION = PAYLOAD_BLOCK_BITS
PAYLOAD_REGRESSION = PAYLOAD_BLOCK_BITS
RANK_CLEAR = 20.0
RANK_PROMISING = 40.0
ABLATION_DISAGREEMENT = 0.25
ABLATION_REGRET_BITS = 1.0e5
EDGE_FEATURES = (
    "same_target",
    "switch",
    "prev_sinr_if_same",
    "continue_progress",
    "buoy_progress",
    "closing_speed",
    "heading_cos",
    "velocity_valid",
    "distance_delta",
)
GLOBAL_FEATURES = ("episode_fraction", "remaining_fraction", "decision_index")
GROUP_FEATURES = {
    "continuation_switch": ("same_target", "switch", "prev_sinr_if_same"),
    "service_continuity": ("continue_progress", "buoy_progress"),
    "motion_trend": ("closing_speed", "heading_cos", "velocity_valid", "distance_delta"),
    "global_context": GLOBAL_FEATURES,
}


def _slice(names: tuple[str, ...], start: int) -> tuple[int, int]:
    return start, start + 4 * len(names)


CONTINUATION_SLICE = _slice(GROUP_FEATURES["continuation_switch"], 0)
SERVICE_SLICE = _slice(GROUP_FEATURES["service_continuity"], CONTINUATION_SLICE[1])
MOTION_SLICE = _slice(GROUP_FEATURES["motion_trend"], SERVICE_SLICE[1])
TEMP_AGG_DIM = MOTION_SLICE[1]
GLOBAL_SLICE = (TEMP_AGG_DIM, TEMP_AGG_DIM + len(GLOBAL_FEATURES))
TEMP_INPUT_DIM = GLOBAL_SLICE[1]
FINAL_REL_DIM = int(REL_DIM) + int(TEMP_INPUT_DIM)
GROUPS = {
    "continuation_switch": CONTINUATION_SLICE,
    "service_continuity": SERVICE_SLICE,
    "motion_trend": MOTION_SLICE,
    "global_context": GLOBAL_SLICE,
}


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5v refused deployment holdout")
    if value not in TRAIN_SCENARIOS and value not in CONFIRM_SCENARIOS:
        raise RuntimeError(f"phase-5v refused seed {value}")


__all__ = [
    "ABLATION_DISAGREEMENT",
    "ABLATION_REGRET_BITS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CASE_SCENARIO",
    "CASE_TIME",
    "CHECKPOINTS",
    "CONFIRM_SCENARIOS",
    "CONTINUATION_SLICE",
    "DEPLOYMENT_VERIFIED",
    "EDGE_FEATURES",
    "EPOCHS",
    "FINAL_REL_DIM",
    "GLOBAL_FEATURES",
    "GLOBAL_SLICE",
    "GRAD_CLIP",
    "GROUPS",
    "LEARNING_RATE",
    "MODEL_POOL_WIDTH",
    "MOTION_SLICE",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PAYLOAD_REGRESSION",
    "PHASE5U_CASE_RANK",
    "PHASE5U_CHECKPOINTS",
    "PLANNER_READY",
    "RANK_CLEAR",
    "RANK_PROMISING",
    "REL_DIM",
    "REPORT",
    "SEEDS",
    "SERVICE_SLICE",
    "T15_CLEAR_DROP",
    "T15_REGRESSION",
    "TEMP_AGG_DIM",
    "TEMP_INPUT_DIM",
    "TIMES",
    "TRAIN_SCENARIOS",
    "WEIGHT_DECAY",
    "assert_consumed",
]
