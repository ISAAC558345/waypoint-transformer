"""Preregistered experiment settings. Thresholds are fixed before training results exist."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import (
    DEV_POOL,
    PPO,
    TRAIN_POOL,
    UPDATES,
)
from wt.benchmark_v2.phase6i_priority_tag.protocol import (
    AGE_SCALE_STEPS,
    BETA_TAG,
    BLOCK_BITS,
    SCHEMA_VERSION,
    TAG_RATE,
    TAG_SEED,
    TOL_BITS,
)

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_reward"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Experiment_Report_CN.md"
PLAN = ROOT / "docs" / "research" / "HTW_UrgencyAware_Experiment_Branch_Plan_CN.md"
FORMAL_ACTOR = ROOT / "src" / "wt" / "benchmark_v2" / "phase6g_lowlevel_mappo" / "checkpoints" / "phase6g_seed0.pt"
LEGACY_ACTOR = ROOT / "results" / "phase5_hgnn_urgency" / "checkpoints" / "mappo_original.pt"
EXPECTED_FORMAL_ACTOR_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"

OBSERVATION_SCHEMA = "synthetic_priority_observation_v1"
REWARD_SCHEMA = "payload_relative_capped_urgency_v1"
FEATURE_NAMES = ("assigned_head_age_urgency", "assigned_head_synthetic_tag", "assigned_head_valid")
LAMBDA_U = 0.0005
RHO = 0.001
L1_LAMBDA_U = 0.0
TRAIN_SEEDS = (0, 1, 2)
SMOKE_SCENARIOS = (int(TRAIN_POOL[0]), int(TRAIN_POOL[1]))
SMOKE_STEPS = 16
EVAL_ASSIGNMENT_SEED = 7_000_000
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0

# Merge criteria written before any L1/L2 result is known.
URGENCY_METRIC = "urgency_served_bits"
TAGGED_METRIC = "served_tagged_bits"
PAYLOAD_METRIC = "payload_bits"
MIN_PASSING_SEEDS = 2
PAYLOAD_MEAN_MIN = 0.0
PAYLOAD_CI_LOW_MIN = 0.0
URGENCY_CI_LOW_MIN = 0.0
TAGGED_CI_LOW_MIN = 0.0
SAFETY_DIFF_MAX = 0.0
CONSTRAINT_DIFF_MAX = 0.0
DROPPED_DIFF_MAX = 0.0
SINR_DIFF_MIN = 0.0
INTERFERENCE_DIFF_MAX = 0.0

GATE_SMOKE_PASS = "PRIORITY_TAG_EXPERIMENT_SMOKE_PASS"
GATE_SMOKE_BLOCKED = "PRIORITY_TAG_EXPERIMENT_SMOKE_BLOCKED"
GATE_TRAINING_BLOCKED = "PRIORITY_TAG_EXPERIMENT_TRAINING_BLOCKED"
GATE_MERGED = "PRIORITY_TAG_EXPERIMENT_MERGED"
GATE_NOT_MERGED = "PRIORITY_TAG_EXPERIMENT_NOT_MERGED"

WATCHED_PATHS = (
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py",
    "src/wt/lowlevel_v2/observation.py",
    "src/wt/lowlevel_v2/residual_policy.py",
    "src/wt/lowlevel_v2/safety_projector.py",
    "src/wt/benchmark_v2/phase6iaa_formal_planner/planner.py",
    "results/phase5_hgnn_urgency/checkpoints/mappo_original.pt",
)


def assert_train_scenario(scenario_id: int) -> None:
    from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import assert_allowed

    assert_allowed(int(scenario_id), split="train")


def assert_dev_scenario(scenario_id: int) -> None:
    from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import assert_allowed

    assert_allowed(int(scenario_id), split="dev")


__all__ = [
    "AGE_SCALE_STEPS",
    "BETA_TAG",
    "BLOCK_BITS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "DEV_POOL",
    "EXPECTED_FORMAL_ACTOR_SHA256",
    "FEATURE_NAMES",
    "FORMAL_ACTOR",
    "GATE_MERGED",
    "GATE_NOT_MERGED",
    "GATE_SMOKE_BLOCKED",
    "GATE_SMOKE_PASS",
    "GATE_TRAINING_BLOCKED",
    "L1_LAMBDA_U",
    "LAMBDA_U",
    "LEGACY_ACTOR",
    "OBSERVATION_SCHEMA",
    "OUTPUT",
    "PPO",
    "REPORT",
    "REWARD_SCHEMA",
    "RHO",
    "SCHEMA_VERSION",
    "SMOKE_SCENARIOS",
    "SMOKE_STEPS",
    "TAG_RATE",
    "TAG_SEED",
    "TOL_BITS",
    "TRAIN_POOL",
    "TRAIN_SEEDS",
    "UPDATES",
    "WATCHED_PATHS",
]
