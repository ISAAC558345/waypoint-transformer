"""Phase-5ZF thresholds. Locked before substitution scores are read."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import CONFIRM_SCENARIOS, SEEDS, TIMES
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import RESPONSE_FEATURES

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZF_ResponseDistillationDecisionErrorAudit_CN.md"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
FROZEN_FOR_FRESH_CONFIRMATION = False
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
CASE_SCENARIO = 310114
CASE_TIME = 15
CASE_PREDICTED = "2;5;-1;4;-1;-1"
CASE_OPTIMAL = "-1;4;2;5;-1;-1"
CASE_TOP_RANK = 3
TIE_BITS = 1.0
ALIGN_ABS = 1.0e-3
RECOVERY_BAR = 0.70
COMBINED_RECOVERY_BAR = 0.75
NEAR_PROGRESS = 0.8
PAIR_EPS = 1.0e-8
GROUPS = {
    "sinr": (0, 1),
    "interference": (2, 3),
    "progress": (4, 5),
    "rate": (6,),
    "motion": (7, 8, 12, 13),
    "binary": (9, 10, 11),
}
COMBINATIONS = {
    "C1": ("sinr", "interference"),
    "C2": ("sinr", "progress"),
    "C3": ("sinr", "interference", "progress"),
}
PRIMARY_SCALAR = {
    "sinr": "mean_sinr_1",
    "interference": "mean_interference_1",
    "progress": "sum_service_progress_1",
    "rate": "mean_rate_1",
    "motion": "mean_delta_distance_to_target",
    "binary": "mean_service_active_1",
}


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127 or 310118 <= value <= 310125:
        raise RuntimeError(f"scenario {value} is outside this phase")
    if value not in CONFIRM_SCENARIOS:
        raise RuntimeError(f"scenario {value} is outside this phase")


__all__ = [
    "ALIGN_ABS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CASE_OPTIMAL",
    "CASE_PREDICTED",
    "CASE_SCENARIO",
    "CASE_TIME",
    "CASE_TOP_RANK",
    "COMBINATIONS",
    "COMBINED_RECOVERY_BAR",
    "CONFIRM_SCENARIOS",
    "FROZEN_FOR_FRESH_CONFIRMATION",
    "GROUPS",
    "NEAR_PROGRESS",
    "OUTPUT",
    "PAIR_EPS",
    "PLANNER_READY",
    "PRIMARY_SCALAR",
    "RECOVERY_BAR",
    "REPORT",
    "RESPONSE_FEATURES",
    "ROOT",
    "SEEDS",
    "TIE_BITS",
    "TIMES",
    "assert_consumed",
]
