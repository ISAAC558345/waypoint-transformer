"""Phase-5Z thresholds. Locked before the collision audit is scored."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import (
    CONFIRM_SCENARIOS,
    PAYLOAD_BLOCK_BITS,
    TIMES,
    TRAIN_SCENARIOS,
)
from wt.lowlevel_v2.phase5u_representation_repair.protocol import REL_DIM
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import FINAL_REL_DIM, TEMP_INPUT_DIM

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5z_representation_collision_audit"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5Z_AssignmentConditionedRepresentationCollisionAudit_CN.md"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
CASE_SCENARIO = 310114
CASE_TIME = 15
CASE_PREDICTED = "2;5;-1;4;-1;-1"
CASE_OPTIMUM = "-1;4;2;5;-1;-1"
TRACE_SEED = 0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
COSINE_LIMIT = 0.99
VALUE_GAP_BLOCKS = 0.1
POOL_FRACTION = 0.25
PRE_SEPARATE_FRACTION = 0.50
SWAP_WIPE_FRACTION = 0.50
PAIR_SPEARMAN_GAIN = 0.05
PAIR_SPEARMAN_FLOOR = 0.25
PHY_SPEARMAN = 0.40
PHY_GAIN = 0.05
TEMP_DISAGREEMENT = 0.10
TEMP_COSINE_FLAT = 0.99
EMBED_COLLISION_HIGH = 0.25
HEAD_GAP_FRACTION = 0.25
LARGE_GAP_COLLISION = 0.30
CURRENT_SPEARMAN_WEAK = 0.25


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5z refused deployment holdout")
    if value not in TRAIN_SCENARIOS and value not in CONFIRM_SCENARIOS:
        raise RuntimeError(f"phase-5z refused seed {value}")


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CASE_OPTIMUM",
    "CASE_PREDICTED",
    "CASE_SCENARIO",
    "CASE_TIME",
    "CONFIRM_SCENARIOS",
    "COSINE_LIMIT",
    "CURRENT_SPEARMAN_WEAK",
    "DEPLOYMENT_VERIFIED",
    "EMBED_COLLISION_HIGH",
    "FINAL_REL_DIM",
    "HEAD_GAP_FRACTION",
    "LARGE_GAP_COLLISION",
    "OUTPUT",
    "PAIR_SPEARMAN_FLOOR",
    "PAIR_SPEARMAN_GAIN",
    "PAYLOAD_BLOCK_BITS",
    "PHY_GAIN",
    "PHY_SPEARMAN",
    "PLANNER_READY",
    "POOL_FRACTION",
    "PRE_SEPARATE_FRACTION",
    "REL_DIM",
    "REPORT",
    "SWAP_WIPE_FRACTION",
    "TEMP_COSINE_FLAT",
    "TEMP_DISAGREEMENT",
    "TEMP_INPUT_DIM",
    "TIMES",
    "TRACE_SEED",
    "TRAIN_SCENARIOS",
    "assert_consumed",
]
