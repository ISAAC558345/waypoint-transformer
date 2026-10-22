"""Phase-5T thresholds. Fixed before any failure statistic is interpreted."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5t_scorer_failure_audit"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5T_NonAdditiveScorerFailureMechanismAudit_CN.md"
PHASE5S_CHECKPOINTS = ROOT / "results/phase5s_multiscenario_nonadditive" / "checkpoints"
PHASE5O_LABELS = ROOT / "results/phase5o_executor_joint_utility_audit" / "phase5o_executor_conditioned_joint_utility.csv"
PHASE5R_LABELS = ROOT / "results/phase5r_nonadditive_generalization" / "phase5r_ground_truth_jmappo.csv"
PHASE5S_LABELS = ROOT / "results/phase5s_multiscenario_nonadditive" / "phase5s_confirmation_ground_truth.csv"
TRAIN_SCENARIOS = tuple(range(310100, 310110))
CONFIRM_SCENARIOS = tuple(range(310110, 310118))
SCENARIOS = TRAIN_SCENARIOS + CONFIRM_SCENARIOS
TIMES = (0, 15, 30)
SEEDS = (0, 1, 2, 3, 4)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False
PAYLOAD_BLOCK_BITS = 1.0e6
MODEL_POOL_WIDTH = 3
CASE_SCENARIO = 310114
CASE_TIME = 15
OOD_SPEARMAN_STRONG = 0.30
SUPPORT_GAP_STRONG = 0.25
FISHER_SEPARABLE = 1.5
FISHER_WEAK = 1.0
COLLISION_HIGH = 0.10
COLLISION_COSINE = 0.99
ALLPAIR_WEAK = 0.30
TOP1_STRONG = 0.40
TOP5_FRACTION_STRONG = 0.40
SEED_DISAGREE = 0.50
SEED_AGREE = 0.90
SCORE_MARGIN_FAIL = 0.25
TIME_WORSE_RATIO = 1.5
OOD_FEATURES = (
    "velocity_norm_mean",
    "velocity_norm_std",
    "assigned_fraction",
    "assigned_queue_mean",
    "progress_mean",
    "progress_std",
    "total_queue",
    "queue_std",
    "queue_max",
    "urgency_mean",
    "urgency_std",
    "rate_mean",
    "rate_max",
    "distance_mean",
    "distance_min",
    "eta_mean",
    "eta_max",
    "occupied_fraction",
    "interference_mean",
    "interference_max",
    "uav_distance_mean",
    "n_active",
    "n_occupied",
    "assignment_entropy",
)


def assert_consumed_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5t refused deployment holdout")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-5t refused seed {value}")


__all__ = [
    "ALLPAIR_WEAK",
    "CASE_SCENARIO",
    "CASE_TIME",
    "COLLISION_COSINE",
    "COLLISION_HIGH",
    "CONFIRM_SCENARIOS",
    "DEPLOYMENT_VERIFIED",
    "FISHER_SEPARABLE",
    "FISHER_WEAK",
    "MODEL_POOL_WIDTH",
    "OOD_FEATURES",
    "OOD_SPEARMAN_STRONG",
    "OUTPUT",
    "PAYLOAD_BLOCK_BITS",
    "PHASE5O_LABELS",
    "PHASE5R_LABELS",
    "PHASE5S_CHECKPOINTS",
    "PHASE5S_LABELS",
    "PLANNER_READY",
    "REPORT",
    "SCENARIOS",
    "SCORE_MARGIN_FAIL",
    "SEED_AGREE",
    "SEED_DISAGREE",
    "SEEDS",
    "SUPPORT_GAP_STRONG",
    "TIME_WORSE_RATIO",
    "TIMES",
    "TOP1_STRONG",
    "TOP5_FRACTION_STRONG",
    "TRAIN_SCENARIOS",
    "assert_consumed_seed",
]
