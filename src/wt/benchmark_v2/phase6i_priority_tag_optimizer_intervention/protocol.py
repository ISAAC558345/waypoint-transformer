"""Registered optimizer-intervention limits. Thresholds do not move after the pilot."""
from __future__ import annotations

import math
from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import PPO

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_optimizer_intervention"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Optimizer_Intervention_Report_CN.md"
INSTRUCTION_COMMIT = "1a82e56a9decd4727d68ceee51177c91268d50bb"
E13_RESULT_COMMIT = "2bce391e226b0b22657dd93b4201fb630bbaf4bb"
E13_IMPLEMENTATION_COMMIT = "0787d8ddcbb900f659051d04162f4039b3ccfb91"
FORMAL_ACTOR_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"

GATE_COMPLETE = "OPTIMIZER_INTERVENTION_COMPLETE"
GATE_BLOCKED = "OPTIMIZER_INTERVENTION_BLOCKED"
EFFECT_MEASURABLE = "MEASURABLE_OPTIMIZER_EFFECT"
EFFECT_NONE = "NO_PRACTICAL_EFFECT_IN_PILOT"
EFFECT_MIXED = "MIXED_OR_INSUFFICIENT"
SCHEMA_VERSION = "optimizer_clip_group_pilot_v1"

VARIANTS = ("PILOT_J", "PILOT_S")
TRAIN_SEEDS = (0, 1, 2)
UPDATES = 32
SHADOW_UPDATES = (1, 16, 32)
EFFECT_UPDATES = (16, 32)
R_THRESHOLD = 0.01
JOINT_BOUND_ABS = 1.0e-6
REPEAT_ABS = 1.0e-7
J_MATCH_ABS = 1.0e-6
REWARD_ABS = 1.0e-5

SHARED_SOURCES = (
    "src/wt/benchmark_v2/phase6i_priority_tag_reward/mappo.py",
    "src/wt/lowlevel_v2/phase1b/mappo.py",
    "src/wt/round30/ppo.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_queue_lookahead/worker.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_queue_lookahead/rollout.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_queue_lookahead/train.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_direct_service_reward/worker.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_direct_service_reward/reward.py",
    "src/wt/benchmark_v2/phase6i_priority_tag/ledger.py",
    "src/wt/lowlevel_v2/safety_projector.py",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt",
)


def joint_limit() -> float:
    value = float(PPO.max_grad_norm)
    if value != 0.5:
        raise RuntimeError(f"registered max_grad_norm is {value}, not 0.5")
    return value


def separate_limit() -> float:
    return float(joint_limit()) / math.sqrt(2.0)


def assert_registered_ppo() -> None:
    joint_limit()
    observed = (
        int(PPO.rollout_length),
        int(PPO.num_envs),
        int(PPO.update_epochs),
        int(PPO.num_minibatches),
        float(PPO.learning_rate),
        float(PPO.gamma),
        float(PPO.gae_lambda),
        float(PPO.clip_eps),
        float(PPO.entropy_coef),
        float(PPO.value_coef),
    )
    expected = (48, 4, 4, 4, 3.0e-4, 0.99, 0.95, 0.2, 0.01, 0.5)
    if observed != expected:
        raise RuntimeError(f"PPO config drifted from the E12 registration: {observed}")


def schema() -> dict:
    assert_registered_ppo()
    return {
        "schema_version": SCHEMA_VERSION,
        "instruction_commit": INSTRUCTION_COMMIT,
        "e13_result_commit": E13_RESULT_COMMIT,
        "joint_limit": float(joint_limit()),
        "separate_limit": float(separate_limit()),
        "updates": int(UPDATES),
        "shadow_updates": list(SHADOW_UPDATES),
        "effect_updates": list(EFFECT_UPDATES),
        "r_threshold": float(R_THRESHOLD),
        "joint_bound_abs": float(JOINT_BOUND_ABS),
        "repeat_abs": float(REPEAT_ABS),
        "j_match_abs": float(J_MATCH_ABS),
        "variants": list(VARIANTS),
        "train_seeds": list(TRAIN_SEEDS),
    }
