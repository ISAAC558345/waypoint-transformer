"""Phase-5ZD thresholds. Locked before 310110-310117 are scored."""
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
OUTPUT = Path(__file__).resolve().parent
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZD_ExecutorConditionedJointValue_CN.md"
CHECKPOINTS = OUTPUT / "checkpoints"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
HUBER_DELTA = 1.0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
INTEGRITY_SAMPLES = 100
INTEGRITY_SEED = 0
CASE_SCENARIO = 310114
CASE_TIME = 15
CASE_PREDICTED = "2;5;-1;4;-1;-1"
CASE_OPTIMAL = "-1;4;2;5;-1;-1"
TOP3_BAR = 0.85
REGRET_BAR = 0.12
SPEARMAN_BAR = 0.70
MIN_SCENARIO_UP = 5
MIN_SEED_UP = 4
CASE_TOP_RANK = 3
DISAGREEMENT_BAR = 0.10
J15_DROP_BITS = PAYLOAD_BLOCK_BITS
SINR_MATCH_ABS = 1.0e-3
PROGRESS_MATCH_ABS = 1.0e-4
PAYLOAD_MATCH_ABS = 1.0
RESPONSE_FEATURES = (
    "sinr_1",
    "delta_sinr",
    "interference_1",
    "delta_interference",
    "service_progress_1",
    "delta_service_progress",
    "rate_1",
    "distance_to_target_1",
    "delta_distance_to_target",
    "assigned_in_range_1",
    "service_active_1",
    "same_target_after_step_1",
    "radial_motion_1",
    "heading_alignment_1",
)
RESPONSE_GROUPS = {
    "sinr": (0, 1),
    "interference": (2, 3),
    "progress": (4, 5),
    "motion": (7, 8, 12, 13),
}
FORBIDDEN_INPUTS = (
    "j15",
    "payload_h",
    "q_exec_h",
    "completed_blocks",
    "future_sinr",
    "future_progress",
    "future_queue",
    "oracle_rank",
)


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127 or 310118 <= value <= 310125:
        raise RuntimeError(f"scenario {value} is outside this phase")
    if value not in TRAIN_SCENARIOS and value not in CONFIRM_SCENARIOS:
        raise RuntimeError(f"scenario {value} is outside this phase")


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CASE_OPTIMAL",
    "CASE_PREDICTED",
    "CASE_SCENARIO",
    "CASE_TIME",
    "CASE_TOP_RANK",
    "CHECKPOINTS",
    "CONFIRM_SCENARIOS",
    "DISAGREEMENT_BAR",
    "EPOCHS",
    "FINAL_REL_DIM",
    "FORBIDDEN_INPUTS",
    "GRAD_CLIP",
    "HUBER_DELTA",
    "INTEGRITY_SAMPLES",
    "INTEGRITY_SEED",
    "J15_DROP_BITS",
    "LEARNING_RATE",
    "MIN_SCENARIO_UP",
    "MIN_SEED_UP",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PAYLOAD_MATCH_ABS",
    "PLANNER_READY",
    "PROGRESS_MATCH_ABS",
    "REGRET_BAR",
    "REPORT",
    "RESPONSE_FEATURES",
    "RESPONSE_GROUPS",
    "ROOT",
    "SEEDS",
    "SINR_MATCH_ABS",
    "SPEARMAN_BAR",
    "TIMES",
    "TOP3_BAR",
    "TRAIN_SCENARIOS",
    "WEIGHT_DECAY",
    "assert_consumed",
]
