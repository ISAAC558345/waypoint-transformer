"""Phase-6E candidate-generator rules. Written before any J15 evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6E_MacroActionCandidateGeneratorRedesign_CN.md"
DESIGN_POOL = tuple(range(320000, 320048))
CONFIRM_POOL = tuple(range(320048, 320064))
DEVELOPMENT_POOL = DESIGN_POOL + CONFIRM_POOL
RESERVED_POOL = tuple(range(320100, 320132))
HISTORICAL_CONSUMED = tuple(range(310100, 310126))
GENERATORS = ("G1_RATE", "G2_DISTANCE", "G3_ETA", "G4_QUEUE_RATE", "G5_REACHABLE_RATE", "G6_REACHABLE_QUEUE_RATE")
SIMPLICITY = {name: index for index, name in enumerate(GENERATORS)}
KS = (3, 4, 5)
REFERENCE_GENERATOR = "G1_RATE"
REFERENCE_K = 5
HOLD = 15
FULL_COVERAGE_MIN = 0.90
MEAN_NREGRET_MAX = 0.05
TAIL_NREGRET_MAX = 0.10
STATE_PASS_FRACTION_MIN = 0.95
NREGRET_EPS_BITS = 1.0
ZERO_REGRET_BITS = 1.0
K4_REGRET_GAP_MAX = 0.02
K4_COVERAGE_DROP_MAX = 0.05
K4_POOL_RATIO_MAX = 0.50
EQUIV_NREGRET = 0.01
EQUIV_COVERAGE = 0.02
EQUIV_RATIO = 1.25
INTERFERENCE_DROP_MIN = 0.10
INTERFERENCE_MIN_SCENARIOS = 8
PHASE6D_FLAGS = {
    "HOLD_MISMATCH_CONFIRMED": False,
    "TEMPORAL_OBJECTIVE_MISMATCH_CONFIRMED": True,
    "FUTURE_ARRIVAL_INFORMATION_VALUE_CONFIRMED": True,
    "TEMPORAL_CANDIDATE_COVERAGE_FAILURE": True,
    "MACRO_RELATION_VALUE_CONFIRMED": True,
}


def assert_allowed(scenario_id: int, *, split: str) -> None:
    value = int(scenario_id)
    if value in RESERVED_POOL or value in HISTORICAL_CONSUMED:
        raise RuntimeError(f"phase-6e refuses scenario {value}")
    pool = DESIGN_POOL if split == "design" else CONFIRM_POOL
    if value not in pool:
        raise RuntimeError(f"phase-6e {split} split does not include {value}")


def rule_passes(row: dict) -> bool:
    return bool(
        int(row["illegal_assignments"]) == 0
        and float(row["full_assignment_coverage"]) >= FULL_COVERAGE_MIN
        and float(row["mean_nregret"]) <= MEAN_NREGRET_MAX
        and float(row["state_pass_fraction"]) >= STATE_PASS_FRACTION_MIN
        and float(row["p95_nregret"]) <= TAIL_NREGRET_MAX
    )


def k4_sufficient(k4: dict | None, k5: dict | None) -> bool:
    if k4 is None or k5 is None:
        return False
    regret_gap = float(k4["mean_nregret"]) - float(k5["mean_nregret"])
    coverage_drop = float(k5["full_assignment_coverage"]) - float(k4["full_assignment_coverage"])
    smaller = float(k4["mean_pool_size"]) <= K4_POOL_RATIO_MAX * float(k5["mean_pool_size"])
    return bool(
        regret_gap <= K4_REGRET_GAP_MAX
        and coverage_drop <= K4_COVERAGE_DROP_MAX
        and smaller
    )


def select_rule(rows: list[dict]) -> dict:
    """Pick one passing rule. Approximate ties resolve to the simpler formula."""
    passing = [row for row in rows if rule_passes(row)]
    if not passing:
        return {
            "selected_generator": None,
            "selected_k": None,
            "candidate_generator_frozen": False,
            "passing_rules": [],
        }
    best_regret = min(float(row["mean_nregret"]) for row in passing)
    band = [row for row in passing if float(row["mean_nregret"]) <= best_regret + EQUIV_NREGRET]
    best_coverage = max(float(row["full_assignment_coverage"]) for row in band)
    band = [row for row in band if float(row["full_assignment_coverage"]) >= best_coverage - EQUIV_COVERAGE]
    smallest_pool = min(float(row["mean_pool_size"]) for row in band)
    band = [row for row in band if float(row["mean_pool_size"]) <= max(smallest_pool, 1.0) * EQUIV_RATIO]
    fastest = min(float(row["mean_runtime_s"]) for row in band)
    band = [row for row in band if float(row["mean_runtime_s"]) <= max(fastest, 1.0e-12) * EQUIV_RATIO]
    band.sort(key=lambda row: (int(row["simplicity"]), int(row["k"]), str(row["generator"])))
    chosen = band[0]
    return {
        "selected_generator": str(chosen["generator"]),
        "selected_k": int(chosen["k"]),
        "candidate_generator_frozen": True,
        "passing_rules": [
            {"generator": str(row["generator"]), "k": int(row["k"])}
            for row in passing
        ],
        "equivalence_band": [
            {"generator": str(row["generator"]), "k": int(row["k"])}
            for row in band
        ],
    }


def choose_gate(
    *,
    conservation_ok: bool,
    illegal_assignments: int,
    passing_rules: list[dict],
    confirmation_pass: bool,
) -> dict:
    if (not conservation_ok) or int(illegal_assignments) > 0:
        gate, name = "CAND-E", "ACCOUNTING_OR_LEGALITY_FAILURE"
    elif not passing_rules:
        gate, name = "CAND-C", "NO_COMPACT_CANDIDATE_RULE_FOUND"
    elif not confirmation_pass:
        gate, name = "CAND-B", "CANDIDATE_GENERALIZATION_FAILED"
    elif all(int(row["k"]) == 5 for row in passing_rules):
        gate, name = "CAND-D", "TOP5_REQUIRED"
    else:
        gate, name = "CAND-A", "MACRO_CANDIDATE_COVERAGE_CONFIRMED"
    return {
        "gate": gate,
        "gate_name": name,
        "phase6b_gate": "BITQ-A",
        "phase6c_gate": "DYN-A",
        "phase6d_gate": "TEMP-F",
        "phase6d_flags": dict(PHASE6D_FLAGS),
        "planner_ready": False,
        "deployment_verified": False,
        "deployment_pass": False,
        "trained": False,
        "fresh_confirmation_untouched": True,
        "used_realized_future_arrival": False,
        "j15_used_as_candidate_score": False,
        "reopen_hold_aware_comparison": gate in {"CAND-A", "CAND-D"},
        "train_hgnn": False,
    }


def scenario_means(values_by_scenario: dict[int, list[float]]) -> np.ndarray:
    if not values_by_scenario:
        return np.zeros(0, dtype=np.float64)
    return np.asarray([float(np.mean(values)) for values in values_by_scenario.values()], dtype=np.float64)


def summarize_distribution(values: np.ndarray) -> dict:
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return {"mean": None, "median": None, "p90": None, "p95": None, "maximum": None}
    return {
        "mean": float(np.mean(sample)),
        "median": float(np.median(sample)),
        "p90": float(np.percentile(sample, 90)),
        "p95": float(np.percentile(sample, 95)),
        "maximum": float(np.max(sample)),
    }
