"""Registered full-budget clip-group comparison. Thresholds stay fixed after this file."""
from __future__ import annotations

import math
from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import PPO
from wt.benchmark_v2.phase6i_priority_tag_optimizer_intervention.protocol import joint_limit, separate_limit

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_clip_benchmark"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Clip_Group_Benchmark_Report_CN.md"
INSTRUCTION_COMMIT = "46e965f673791f80a3b4161d3c760d48513abacc"
PREPARE_COMMIT = "4805370832cbcd03f92b295cbfe1381684e7768d"
E14_RESULT_COMMIT = "38ed6b91fc1df002f9eeb347c927f83a0791d00e"
E14_IMPLEMENTATION_COMMIT = "9c971f7cbaf86da7187e0bd90ba93727bc1a990d"
FORMAL_ACTOR_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"
E14_OUTPUT = ROOT / "results" / "phase6i_priority_tag_optimizer_intervention"

DIRECTION_CHOSEN = "CHOSEN_A"
GATE_PASS = "CLIP_GROUP_BENCHMARK_PASS"
GATE_FAIL = "CLIP_GROUP_BENCHMARK_FAIL"
GATE_INCONCLUSIVE = "CLIP_GROUP_BENCHMARK_INCONCLUSIVE"
GATE_BLOCKED = "CLIP_GROUP_BENCHMARK_BLOCKED"
SCHEMA_VERSION = "clip_group_full_budget_v1"

VARIANTS = ("FULL_J", "FULL_S")
CLIP_RULE = {"FULL_J": "PILOT_J", "FULL_S": "PILOT_S"}
TRAIN_SEEDS = (0, 1, 2)
UPDATES = 300
SNAPSHOT_UPDATES = (1, 150, 300)
MIN_PASSING_SEEDS = 2
PAYLOAD_CI_FLOOR_FRACTION = 0.005
DEV_SCENARIO_MIN = 321096
DEV_SCENARIO_MAX = 321127
EVAL_EPISODES = 224

SHARED_SOURCES = (
    "src/wt/benchmark_v2/phase6i_priority_tag_optimizer_intervention/clip.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_optimizer_intervention/agent.py",
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


def assert_registered_budget() -> None:
    if int(UPDATES) != 300 or TRAIN_SEEDS != (0, 1, 2) or VARIANTS != ("FULL_J", "FULL_S"):
        raise RuntimeError("full-budget registration drifted")
    if float(joint_limit()) != 0.5:
        raise RuntimeError("joint clip bound is not 0.5")
    if abs(float(separate_limit()) - 0.5 / math.sqrt(2.0)) > 0.0:
        raise RuntimeError("separate clip bound drifted")
    observed = (
        int(PPO.rollout_length),
        int(PPO.num_envs),
        int(PPO.update_epochs),
        int(PPO.num_minibatches),
        float(PPO.learning_rate),
        float(PPO.max_grad_norm),
    )
    if observed != (48, 4, 4, 4, 3.0e-4, 0.5):
        raise RuntimeError(f"PPO budget drifted: {observed}")


def expected_adam_steps() -> int:
    assert_registered_budget()
    return int(len(TRAIN_SEEDS) * len(VARIANTS) * UPDATES * int(PPO.update_epochs) * int(PPO.num_minibatches))


def schema() -> dict:
    assert_registered_budget()
    return {
        "schema_version": SCHEMA_VERSION,
        "direction_status": DIRECTION_CHOSEN,
        "instruction_commit": INSTRUCTION_COMMIT,
        "prepare_commit": PREPARE_COMMIT,
        "e14_result_commit": E14_RESULT_COMMIT,
        "joint_limit": float(joint_limit()),
        "separate_limit": float(separate_limit()),
        "updates": int(UPDATES),
        "snapshot_updates": list(SNAPSHOT_UPDATES),
        "variants": list(VARIANTS),
        "train_seeds": list(TRAIN_SEEDS),
        "expected_ppo_updates": int(len(TRAIN_SEEDS) * len(VARIANTS) * UPDATES),
        "expected_adam_steps": int(expected_adam_steps()),
        "evaluation_episodes": int(EVAL_EPISODES),
        "payload_ci_floor_fraction": float(PAYLOAD_CI_FLOOR_FRACTION),
        "min_passing_seeds": int(MIN_PASSING_SEEDS),
        "seamless_resume": False,
    }
