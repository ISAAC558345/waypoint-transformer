"""Phase-5ZC thresholds. Locked before any observability number is computed."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import (
    CONFIRM_SCENARIOS,
    PAYLOAD_BLOCK_BITS,
    TIMES,
    TRAIN_SCENARIOS,
)

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZC_ExecutorResponseDynamicObservabilityAudit_CN.md"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
HORIZONS = (1, 3, 5, 8)
ORACLE_HORIZON = 15
CASE_SCENARIO = 310114
CASE_TIME = 15
CASE_PREDICTED = "2;5;-1;4;-1;-1"
CASE_OPTIMAL = "-1;4;2;5;-1;-1"
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
SPEARMAN_BAR = 0.70
TOP3_BAR = 0.85
REGRET_BAR = 0.12
RESIDUAL_BAR = 0.40
SPEARMAN_GAIN = 0.10
CROSS_UAV_FRACTION = 0.20
COSINE_LIMIT = 0.99
BLOCK_GAPS = (1, 2, 3, 4)
# ACT0 ranking uses this one pre-registered scalar. It is not chosen after the audit.
ACT0_SCORE = "mean_radial_toward_assigned_buoy"
# Horizon ranking uses the already registered Score-B, not a new combination.
HORIZON_SCORE = "q_exec_h"
# L0 ranking uses the marginal rate already stored on the decision snapshot.
L0_SCORE = "sum_marginal_rate_if_assigned"
PHY_SCORE = "sum_joint_sinr_rate"
# A feature that is lower-is-better is negated before the residual test.
RESIDUAL_ORIENTATION = {
    "mean_sinr": 1.0,
    "service_progress": 1.0,
    "mean_interference": -1.0,
    "distance_reduction": 1.0,
    "motion_efficiency": 1.0,
}
MOTION_EFFICIENCY = "distance_reduction_h / (low_level_action_norm + 1e-8)"
# The stored shadow has an in-range count at the checkpoint, not an integrated duration.
IN_RANGE_FIELD = "number_in_service_radius_h"


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127 or 310118 <= value <= 310125:
        raise RuntimeError(f"scenario {value} is outside this audit")
    allowed = set(TRAIN_SCENARIOS) | set(CONFIRM_SCENARIOS) | {CASE_SCENARIO}
    if value not in allowed:
        raise RuntimeError(f"scenario {value} is outside this audit")


__all__ = [
    "ACT0_SCORE",
    "BLOCK_GAPS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CASE_OPTIMAL",
    "CASE_PREDICTED",
    "CASE_SCENARIO",
    "CASE_TIME",
    "CONFIRM_SCENARIOS",
    "COSINE_LIMIT",
    "CROSS_UAV_FRACTION",
    "DEPLOYMENT_VERIFIED",
    "HORIZON_SCORE",
    "HORIZONS",
    "IN_RANGE_FIELD",
    "L0_SCORE",
    "MOTION_EFFICIENCY",
    "ORACLE_HORIZON",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PHY_SCORE",
    "PLANNER_READY",
    "REGRET_BAR",
    "REPORT",
    "RESIDUAL_BAR",
    "RESIDUAL_ORIENTATION",
    "ROOT",
    "SPEARMAN_BAR",
    "SPEARMAN_GAIN",
    "TIMES",
    "TOP3_BAR",
    "TRAIN_SCENARIOS",
    "assert_consumed",
]
