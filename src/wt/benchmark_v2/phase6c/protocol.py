"""Phase-6C dynamic-range rules. The reserved confirmation pool stays closed."""
from __future__ import annotations

from pathlib import Path

import numpy as np

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6C_BenchmarkV2_DynamicRangeAndRelationalPlanning_CN.md"
DEVELOPMENT_POOL = tuple(range(320000, 320064))
RESERVED_POOL = tuple(range(320100, 320132))
HISTORICAL_CONSUMED = tuple(range(310100, 310126))
SATURATION_MAX = 0.50
PAIRWISE_TIE_MAX = 0.40
DISTINGUISH_MIN = 0.75
POSITIVE_MIN = 0.60
COVERAGE_MIN = 0.90
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
HOLD_STEPS = 15
PRIMARY_RELATIONAL_BASELINE = "max_rate_independent"
SECONDARY_RELATIONAL_BASELINE = "queue_rate"
JOINT_CONTROLLER = "joint_true_bit"


def assert_development(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_POOL or value in HISTORICAL_CONSUMED:
        raise RuntimeError(f"phase-6c refuses scenario {value}")
    if value not in DEVELOPMENT_POOL:
        raise RuntimeError(f"phase-6c development pool does not include {value}")


def cluster_bootstrap_ci(values: np.ndarray, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict:
    """One observation per scenario. Resample scenarios, not timesteps."""
    sample = np.asarray(values, dtype=np.float64)
    if sample.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None}
    generator = np.random.default_rng(int(seed))
    index = generator.integers(0, sample.size, size=(int(draws), sample.size))
    means = sample[index].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean": float(sample.mean()), "ci_low": float(low), "ci_high": float(high)}


def choose_gate(
    *,
    conservation_ok: bool,
    best_coverage_at_3: float,
    dynamic_range_ok: bool,
    relational_confirmed: bool,
    short_horizon_upper_bound_available: bool,
) -> dict:
    """DYN-C is unreachable unless a legal short-horizon upper bound exists."""
    if not conservation_ok:
        gate, name = "DYN-E", "ACCOUNTING_REGRESSION"
    elif float(best_coverage_at_3) < COVERAGE_MIN:
        gate, name = "DYN-F", "CANDIDATE_COVERAGE_FAILURE"
    elif short_horizon_upper_bound_available:
        gate, name = "DYN-C", "MYOPIC_JOINT_CONTROL_SUFFICIENT"
    elif not dynamic_range_ok:
        gate, name = "DYN-D", "BENCHMARK_STILL_WEAK"
    elif relational_confirmed:
        gate, name = "DYN-B", "RELATIONAL_PLANNING_VALUE_CONFIRMED"
    else:
        gate, name = "DYN-A", "BIT_LEVEL_BENCHMARK_DISCRIMINATIVE"
    return {
        "gate": gate,
        "gate_name": name,
        "phase5zp_gate": "CONF-C",
        "phase5zq_gate": "ATTR-C",
        "phase6a_gate": "BV2-E",
        "phase6b_gate": "BITQ-A",
        "planner_ready": False,
        "deployment_verified": False,
        "deployment_pass": False,
        "trained": False,
        "lambda_switch": "NOT_DEFINED",
        "lambda_progress": "NOT_DEFINED",
        "low_level_status": "LOW_LEVEL_RETRAIN_PENDING",
        "fresh_confirmation_untouched": True,
    }
