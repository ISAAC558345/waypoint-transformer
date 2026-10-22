"""Phase-5R constants. Thresholds are fixed before any new scenario is scored."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5r_nonadditive_generalization"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5R_NonAdditiveJointScorerGeneralization_CN.md"
CHECKPOINTS = OUTPUT / "checkpoints"
PHASE5Q_PREDICTIONS = ROOT / "results/phase5q_encoder_scorer_audit" / "phase5q_assignment_predictions.csv"
PHASE5Q_STATES = ROOT / "results/phase5q_encoder_scorer_audit" / "phase5q_seed_state_metrics.csv"
SCENARIOS = tuple(range(310102, 310110))
ROLE = "PHASE5_DEVELOPMENT_GENERALIZATION"
TIMES = (0, 15, 30)
SEEDS = (0, 1, 2, 3, 4)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
PAYLOAD_BLOCK_BITS = 1.0e6
COVERAGE_EPS = 1.0
POOL_HIT_FRACTION = 0.5
COVERAGE_OK_RATIO = 0.5
EXPANDED_WIDTH = 5
MODEL_POOL_WIDTH = 3
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
WORKERS = 4
CACHE_VERSION = 1
ABLATION_REGRET_BITS = 1.0e5
ABLATION_DISAGREEMENT = 0.25


def assert_phase5r_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5r refused deployment holdout")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-5r refused seed {value}")


__all__ = [
    "ABLATION_DISAGREEMENT",
    "ABLATION_REGRET_BITS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CACHE_VERSION",
    "CHECKPOINTS",
    "COVERAGE_EPS",
    "COVERAGE_OK_RATIO",
    "DEPLOYMENT_VERIFIED",
    "EXPANDED_WIDTH",
    "MODEL_POOL_WIDTH",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PHASE5Q_PREDICTIONS",
    "PHASE5Q_STATES",
    "PLANNER_READY",
    "POOL_HIT_FRACTION",
    "REPORT",
    "ROLE",
    "SCENARIOS",
    "SEEDS",
    "TIMES",
    "WORKERS",
    "assert_phase5r_seed",
]
