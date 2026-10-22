"""Phase-5ZK thresholds. Locked before the closure measurement."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import CONFIRM_SCENARIOS, TIMES

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5ZK_TrueH1NumericalEquivalenceAndLatencyClosure_CN.md"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
FROZEN_FOR_FRESH_CONFIRMATION = False
ATOL = 1.0e-6
RTOL = 1.0e-5
BASELINE_VALUE_ABS = 1.0e-3
MEDIAN_SPEEDUP = 0.20
WARMUP = 1
REPEATS = 5
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
VALUE_SEED = 0
EXCEPTION_SCENARIO = 310110
EXCEPTION_TIME = 30
CASE_SCENARIO = 310114
CASE_TIME = 15
CASE_PREDICTED = "2;5;-1;4;-1;-1"
CASE_OPTIMAL = "-1;4;2;5;-1;-1"
# End-to-end wall time includes encoder, candidate preparation, H1, one Model-R1, and argmax.
# The median is the median of per-state means. Repeats are not scenarios.
# The scenario-cluster CI is the 95% interval of the mean paired absolute improvement.


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127 or 310118 <= value <= 310125:
        raise RuntimeError(f"scenario {value} is outside this phase")
    if value not in CONFIRM_SCENARIOS:
        raise RuntimeError(f"scenario {value} is outside this phase")


__all__ = [
    "ATOL",
    "BASELINE_VALUE_ABS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CASE_OPTIMAL",
    "CASE_PREDICTED",
    "CASE_SCENARIO",
    "CASE_TIME",
    "CONFIRM_SCENARIOS",
    "DEPLOYMENT_VERIFIED",
    "EXCEPTION_SCENARIO",
    "EXCEPTION_TIME",
    "FROZEN_FOR_FRESH_CONFIRMATION",
    "MEDIAN_SPEEDUP",
    "OUTPUT",
    "PLANNER_READY",
    "REPEATS",
    "REPORT",
    "ROOT",
    "RTOL",
    "TIMES",
    "VALUE_SEED",
    "WARMUP",
    "assert_consumed",
]
