"""Phase-6D temporal-planning rules. The reserved confirmation pool stays closed."""
from __future__ import annotations

from pathlib import Path

import numpy as np

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6D_TemporalAbstractionAndHoldAwareJointPlanning_CN.md"
DEVELOPMENT_POOL = tuple(range(320000, 320064))
RESERVED_POOL = tuple(range(320100, 320132))
HISTORICAL_CONSUMED = tuple(range(310100, 310126))
HOLDS = (1, 3, 5, 8, 10, 15)
SHORT_HOLDS = (1, 3, 5, 8, 10)
COVERAGE_SCENARIOS = tuple(range(320000, 320016))
MACRO_DISAGREEMENT_MIN = 0.20
POSITIVE_MIN = 0.60
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
REPRODUCTION_ABS_BITS = 1.0
PHASE6C_MEANS = {
    "max_rate_independent": 76118469.85841921,
    "queue_rate": 66085102.04051116,
    "joint_true_bit": 15000000.0,
}
COVERAGE_RELATIVE_GAIN = 0.05
IDLE_DROP_FRACTION = 0.50


def assert_development(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_POOL or value in HISTORICAL_CONSUMED:
        raise RuntimeError(f"phase-6d refuses scenario {value}")
    if value not in DEVELOPMENT_POOL:
        raise RuntimeError(f"phase-6d development pool does not include {value}")


def cluster_bootstrap_ci(values: np.ndarray, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict:
    sample = np.asarray(values, dtype=np.float64)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None}
    generator = np.random.default_rng(int(seed))
    index = generator.integers(0, finite.size, size=(int(draws), finite.size))
    means = finite[index].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean": float(finite.mean()), "ci_low": float(low), "ci_high": float(high)}


def gain_confirmed(delta: np.ndarray) -> dict:
    clean = np.asarray(delta, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    ci = cluster_bootstrap_ci(clean)
    positive = float(np.mean(clean > 1.0)) if clean.size else 0.0
    confirmed = bool(
        clean.size
        and float(np.mean(clean)) > 0.0
        and ci["ci_low"] is not None
        and ci["ci_low"] > 0.0
        and positive >= POSITIVE_MIN
    )
    return {
        "mean": None if not clean.size else float(np.mean(clean)),
        "median": None if not clean.size else float(np.median(clean)),
        "std": None if not clean.size else float(np.std(clean)),
        "positive_fraction": positive,
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "confirmed": confirmed,
    }


def choose_gate(
    *,
    reproduction_ok: bool,
    conservation_ok: bool,
    coverage_failure: bool,
    hold_aware_beats_max_rate: bool,
    temporal_objective_mismatch: bool,
    future_beats_current: bool,
    future_beats_max_rate: bool,
    hold_mismatch: bool,
) -> dict:
    if (not reproduction_ok) or (not conservation_ok):
        gate, name = "TEMP-E", "REPRODUCTION_OR_ACCOUNTING_FAILURE"
    elif coverage_failure:
        gate, name = "TEMP-F", "TEMPORAL_CANDIDATE_COVERAGE_FAILURE"
    elif hold_aware_beats_max_rate and temporal_objective_mismatch:
        gate, name = "TEMP-B", "HOLD_AWARE_JOINT_PLANNING_VALUE_CONFIRMED"
    elif (not hold_aware_beats_max_rate) and future_beats_current and future_beats_max_rate:
        gate, name = "TEMP-C", "FUTURE_ARRIVAL_OBSERVABILITY_LIMIT"
    elif hold_mismatch and not hold_aware_beats_max_rate:
        gate, name = "TEMP-A", "HOLD_MISMATCH_CONFIRMED"
    else:
        gate, name = "TEMP-D", "INDEPENDENT_MAX_RATE_SUFFICIENT"
    return {
        "gate": gate,
        "gate_name": name,
        "phase6a_gate": "BV2-E",
        "phase6b_gate": "BITQ-A",
        "phase6c_gate": "DYN-A",
        "planner_ready": False,
        "deployment_verified": False,
        "deployment_pass": False,
        "trained": False,
        "fresh_confirmation_untouched": True,
        "shadow_is_intermediate_decision_layer": False,
        "non_causal_upper_bound_used_for_training": False,
    }
