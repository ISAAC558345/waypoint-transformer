"""Diagnostic cutoffs for Phase-5W. This phase does not train."""
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
OUTPUT = ROOT / "results/phase5w_residual_mechanism_audit"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5W_ResidualTemporalJointExecutionMechanismAudit_CN.md"
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
CASE_SCENARIO = 310114
CASE_TIME = 15
MATCH_Z_DISTANCE = 1.0
SINR_GAP = 1.0
TRAJECTORY_GAP_M = 2.0
INTERFERENCE_GAP = 0.05
INVERSION_FRACTION = 0.30
CORRELATION_GAP = 0.10
SWAP_SEED_MAJORITY = 3
SYNERGY_BLOCK = PAYLOAD_BLOCK_BITS
SYNERGY_RATIO_MEDIAN = 0.25
MOTION_CLOSING_GAP = 0.15
MOTION_HEADING_GAP = 0.15
MOTION_DISTANCE_GAP = 0.20


def assert_consumed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5w refused deployment holdout")
    if value not in TRAIN_SCENARIOS and value not in CONFIRM_SCENARIOS:
        raise RuntimeError(f"phase-5w refused seed {value}")


__all__ = [
    "CASE_SCENARIO",
    "CASE_TIME",
    "CONFIRM_SCENARIOS",
    "CORRELATION_GAP",
    "DEPLOYMENT_VERIFIED",
    "INTERFERENCE_GAP",
    "INVERSION_FRACTION",
    "MATCH_Z_DISTANCE",
    "MOTION_CLOSING_GAP",
    "MOTION_DISTANCE_GAP",
    "MOTION_HEADING_GAP",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PLANNER_READY",
    "REPORT",
    "SEEDS",
    "SINR_GAP",
    "SWAP_SEED_MAJORITY",
    "SYNERGY_BLOCK",
    "SYNERGY_RATIO_MEDIAN",
    "TIMES",
    "TRAIN_SCENARIOS",
    "TRAJECTORY_GAP_M",
    "assert_consumed",
]
