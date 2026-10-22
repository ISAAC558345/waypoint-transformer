"""Phase-6G low-level adaptation. High-level rules stay frozen."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase1.protocol import (
    ACTION_SCALE,
    ALPHA_RESIDUAL,
    D_SAFE_M,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    LOG_STD_INIT,
    PAYLOAD_SCALE,
    PPO,
    SERVICE_SAFETY_MARGIN_M,
)

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6G_BenchmarkV2_LowLevelMAPPO_CN.md"
LEGACY_ACTOR = ROOT / "results/phase5_hgnn_urgency" / "checkpoints" / "mappo_original.pt"
TRAIN_POOL = tuple(range(321000, 321096))
DEV_POOL = tuple(range(321096, 321128))
RESERVED_POOL = tuple(range(320100, 320132))
BENCHMARK_DEV = tuple(range(320000, 320064))
HISTORICAL = tuple(range(310100, 310126))
HOLD = 15
DECISION_TIMES = (0, 15, 30, 45)
EVAL_ASSIGNMENTS = 8
MODEL_SEEDS = (0, 1, 2, 3, 4)
UPDATES = 300
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
POSITIVE_BIT = 1.0
MIN_POSITIVE_FRACTION = 0.60
MIN_POSITIVE_SEEDS = 4
EXCESSIVE_PROJECTION_RATE = 0.50
SAFETY_RATE_MARGIN = 0.05
CASE_SEED = 0


def assert_allowed(scenario_id: int, *, split: str) -> None:
    value = int(scenario_id)
    if value in RESERVED_POOL or value in BENCHMARK_DEV or value in HISTORICAL:
        raise RuntimeError(f"phase-6g refuses scenario {value}")
    pool = TRAIN_POOL if split == "train" else DEV_POOL
    if value not in pool:
        raise RuntimeError(f"phase-6g {split} split does not include {value}")


def choose_gate(
    *,
    accounting_failures: int,
    interface_failures: int,
    nan_collapse: bool,
    projection_rate: float,
    safety_regression: bool,
    mean_delta: float,
    ci_low: float,
    positive_fraction: float,
    positive_seeds: int,
) -> dict:
    if int(accounting_failures) > 0 or int(interface_failures) > 0:
        gate, name = "LL-E", "INTERFACE_OR_ACCOUNTING_FAILURE"
    elif nan_collapse or float(projection_rate) > EXCESSIVE_PROJECTION_RATE or safety_regression:
        gate, name = "LL-D", "LOW_LEVEL_TRAINING_COLLAPSE"
    elif (
        float(mean_delta) > 0.0
        and float(ci_low) > 0.0
        and float(positive_fraction) >= MIN_POSITIVE_FRACTION
        and int(positive_seeds) >= MIN_POSITIVE_SEEDS
    ):
        gate, name = "LL-A", "V2_LOW_LEVEL_MAPPO_CONFIRMED"
    elif float(mean_delta) > 0.0:
        gate, name = "LL-B", "MEAN_GAIN_BUT_UNSTABLE"
    else:
        gate, name = "LL-C", "NO_LOW_LEVEL_GAIN"
    return {
        "gate": gate,
        "gate_name": name,
        "planner_ready": False,
        "deployment_verified": False,
        "deployment_pass": False,
        "trained_hgnn": False,
        "candidate_generator_modified": False,
        "high_level_modified": False,
        "fresh_confirmation_untouched": True,
        "low_level_frozen": gate == "LL-A",
        "next_phase_regenerates_j15": gate == "LL-A",
        "updates": UPDATES,
        "initialization_rule": "LEGACY_WARM_START if actor state dict loads, else SCRATCH_REQUIRED",
    }
