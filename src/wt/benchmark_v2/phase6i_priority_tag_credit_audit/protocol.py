"""Tolerances and probe limits locked before any credit-audit probe."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_credit_audit"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Training_Credit_Audit_Report_CN.md"
E12_OUTPUT = ROOT / "results" / "phase6i_priority_tag_queue_lookahead"
E12_ROWS = E12_OUTPUT / "scenario_rows.csv"
INSTRUCTION_COMMIT = "dbc1a90d8e082d498f92e04173030cb4274aa2a2"
E12_RESULT_COMMIT = "fabe0406fe8127da1f2b9fb59a6da347adffe21a"
E12_IMPLEMENTATION_COMMIT = "7067295c7d2d545eed22a3bab66bb48f3becd35b"
FORMAL_ACTOR_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"

GATE_COMPLETE = "TAG_CREDIT_AUDIT_COMPLETE"
GATE_BLOCKED = "TAG_CREDIT_AUDIT_BLOCKED"
SCHEMA_VERSION = "tag_credit_audit_v1"
E12_GATE = "QUEUE_LOOKAHEAD_BENCHMARK_FAIL"

VARIANTS = ("L4_C", "L4_Q")
TRAIN_SEEDS = (0, 1, 2)
SAMPLE_UPDATES = (1, 150, 300)
SAMPLE_ENV_INDEX = 0
SAMPLE_STEPS = 48
SAMPLES_PER_GROUP = 144
UPDATES = 300
WINDOWS = ((1, 100), (101, 200), (201, 300))
MAX_REGISTERED_BATCHES = 12
DIAGNOSTIC_PROCESS_SEED = 9_100_000
DIAGNOSTIC_ASSIGNMENT_SEED = 9_200_000
TERMINAL_SCENARIOS = (321000, 321001)

# Locked before the probe. These do not move after a mismatch is seen.
BIT_ABS = 1.0
REWARD_ABS = 1.0e-5
LOOKAHEAD_ABS = 1.0e-5
FLOAT32_ABS = 1.0e-6
GRAD_SUM_ABS = 1.0e-4
GAE_LINEAR_ABS = 1.0e-5
LOGPROB_ABS = 1.0e-4
RATIO_ABS = 1.0e-4

SHARED_SOURCES = (
    "src/wt/benchmark_v2/phase6i_priority_tag_queue_lookahead/train.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_queue_lookahead/rollout.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_queue_lookahead/worker.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_reward/mappo.py",
    "src/wt/lowlevel_v2/phase1b/mappo.py",
    "src/wt/round30/ppo.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_direct_service_reward/worker.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_direct_service_reward/reward.py",
    "src/wt/benchmark_v2/phase6i_priority_tag/ledger.py",
    "src/wt/lowlevel_v2/safety_projector.py",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt",
)


def schema() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "instruction_commit": INSTRUCTION_COMMIT,
        "e12_result_commit": E12_RESULT_COMMIT,
        "e12_implementation_commit": E12_IMPLEMENTATION_COMMIT,
        "registered_batches": int(MAX_REGISTERED_BATCHES),
        "sample_updates": list(SAMPLE_UPDATES),
        "sample_env_index": int(SAMPLE_ENV_INDEX),
        "bit_abs": float(BIT_ABS),
        "reward_abs": float(REWARD_ABS),
        "lookahead_abs": float(LOOKAHEAD_ABS),
        "float32_abs": float(FLOAT32_ABS),
        "logprob_abs": float(LOGPROB_ABS),
        "ratio_abs": float(RATIO_ABS),
        "verification_reruns": 0,
        "terminal_scenarios": list(TERMINAL_SCENARIOS),
        "diagnostic_process_seed": int(DIAGNOSTIC_PROCESS_SEED),
        "diagnostic_assignment_seed": int(DIAGNOSTIC_ASSIGNMENT_SEED),
        "no_parameter_update": True,
    }
