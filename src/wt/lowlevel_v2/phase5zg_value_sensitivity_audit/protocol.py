"""Phase-5ZG thresholds. Locked before sensitivity numbers are read."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import CONFIRM_SCENARIOS, SEEDS, TIMES
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import RESPONSE_FEATURES

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZG_ResponseSpaceValueSensitivityAudit_CN.md"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
FROZEN_FOR_FRESH_CONFIRMATION = False
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
ALIGN_ABS = 1.0e-3
PATH_POINTS = 16
EXPLAIN_SIGN = 0.70
EXPLAIN_SPEARMAN = 0.70
MARGIN_SIGN = 0.70
FINITE_FRACTION = 0.90
STABLE_SCENARIOS = 6
STABLE_SEEDS = 4
UNIQUE_RELATIVE_GAP = 0.05
INTERACTION_RATIO = 0.25
ORDERING_PAIRWISE_MAX = 0.80
VALUE_EPS = 1.0e-6
TIE_BITS = 1.0
CASE_SCENARIO = 310114
CASE_TIME = 15
CASE_PREDICTED = "2;5;-1;4;-1;-1"
CASE_OPTIMAL = "-1;4;2;5;-1;-1"
GROUPS = {
    "sinr": (0, 1),
    "interference": (2, 3),
    "progress": (4, 5),
    "rate": (6,),
    "motion": (7, 8, 12, 13),
    "binary": (9, 10, 11),
}
PRIMARY_FEATURE = {
    "sinr": 0,
    "interference": 2,
    "progress": 4,
    "rate": 6,
    "motion": 8,
    "binary": 10,
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
    "CONFIRM_SCENARIOS",
    "DEPLOYMENT_VERIFIED",
    "EXPLAIN_SIGN",
    "EXPLAIN_SPEARMAN",
    "FINITE_FRACTION",
    "FROZEN_FOR_FRESH_CONFIRMATION",
    "GROUPS",
    "INTERACTION_RATIO",
    "MARGIN_SIGN",
    "ORDERING_PAIRWISE_MAX",
    "OUTPUT",
    "PATH_POINTS",
    "PLANNER_READY",
    "PRIMARY_FEATURE",
    "REPORT",
    "RESPONSE_FEATURES",
    "ROOT",
    "SEEDS",
    "STABLE_SCENARIOS",
    "STABLE_SEEDS",
    "TIE_BITS",
    "TIMES",
    "UNIQUE_RELATIVE_GAP",
    "VALUE_EPS",
    "assert_consumed",
]
