"""Phase-5U thresholds. Fixed before REP-AUG is scored on 310110-310117."""
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

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5u_representation_repair"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5U_AssignmentRepresentationRepair_CN.md"
CHECKPOINTS = OUTPUT / "checkpoints"
PHASE5S_CHECKPOINTS = ROOT / "results/phase5s_multiscenario_nonadditive" / "checkpoints"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
MODEL_POOL_WIDTH = 3
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
CASE_SCENARIO = 310114
CASE_TIME = 15
COLLISION_ABS_DROP = 0.05
COLLISION_REL_DROP = 0.15
COLLISION_WORSE = 0.02
RANK_CLEAR = 20.0
RANK_IMPROVED = 50.0
T15_STILL_HIGH = PAYLOAD_BLOCK_BITS
PAYLOAD_REGRESSION = PAYLOAD_BLOCK_BITS
ABLATION_DISAGREEMENT = 0.25
ABLATION_REGRET_BITS = 1.0e5
EDGE_FEATURES = (
    "distance_norm",
    "eta_norm",
    "dx_norm",
    "dy_norm",
    "rate_norm",
    "log_path_gain",
    "queue_ratio",
    "urgency",
    "service_progress",
    "in_service_radius",
    "same_current_target",
    "target_occupied",
)
PAIR_FEATURES = (
    "uav_distance_norm",
    "target_distance_norm",
    "direction_cosine",
    "destination_offset_norm",
    "cross_distance_min_norm",
    "cross_distance_max_norm",
)
STATS = ("mean", "max", "min", "std")


def _block(names: tuple[str, ...], start: int) -> tuple[int, int]:
    return start, start + 4 * len(names)


RAW_SLICE = _block(EDGE_FEATURES, 0)
GEOMETRY_NAMES = ("uav_distance_norm", "target_distance_norm", "direction_cosine", "destination_offset_norm")
CROSS_NAMES = ("cross_distance_min_norm", "cross_distance_max_norm")
GEOMETRY_SLICE = _block(GEOMETRY_NAMES, RAW_SLICE[1])
CROSS_SLICE = _block(CROSS_NAMES, GEOMETRY_SLICE[1])
INTERFERENCE_DIM = 5
SINR_DIM = 3
INTERFERENCE_SLICE = (CROSS_SLICE[1], CROSS_SLICE[1] + INTERFERENCE_DIM + SINR_DIM)
REL_DIM = INTERFERENCE_SLICE[1]
GROUPS = {
    "selected_edge_raw": RAW_SLICE,
    "joint_geometry": GEOMETRY_SLICE,
    "cross_geometry": CROSS_SLICE,
    "interference_summary": INTERFERENCE_SLICE,
}


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5u refused deployment holdout")
    if value not in TRAIN_SCENARIOS and value not in CONFIRM_SCENARIOS:
        raise RuntimeError(f"phase-5u refused seed {value}")


__all__ = [
    "ABLATION_DISAGREEMENT",
    "ABLATION_REGRET_BITS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CASE_SCENARIO",
    "CASE_TIME",
    "CHECKPOINTS",
    "COLLISION_ABS_DROP",
    "COLLISION_REL_DROP",
    "COLLISION_WORSE",
    "CONFIRM_SCENARIOS",
    "CROSS_SLICE",
    "DEPLOYMENT_VERIFIED",
    "EDGE_FEATURES",
    "EPOCHS",
    "GEOMETRY_SLICE",
    "GRAD_CLIP",
    "GROUPS",
    "INTERFERENCE_SLICE",
    "LEARNING_RATE",
    "MODEL_POOL_WIDTH",
    "OUTPUT",
    "PAIR_FEATURES",
    "PAYLOAD_BLOCK_BITS",
    "PAYLOAD_REGRESSION",
    "PHASE5S_CHECKPOINTS",
    "PLANNER_READY",
    "RANK_CLEAR",
    "RANK_IMPROVED",
    "RAW_SLICE",
    "REL_DIM",
    "REPORT",
    "SEEDS",
    "T15_STILL_HIGH",
    "TIMES",
    "TRAIN_SCENARIOS",
    "WEIGHT_DECAY",
    "assert_consumed",
]
