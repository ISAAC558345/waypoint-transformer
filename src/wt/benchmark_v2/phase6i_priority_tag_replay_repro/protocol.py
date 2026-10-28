"""Single-scenario replay reproducibility check. Thresholds stay at 1 bit."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_replay_repro"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Replay_Reproducibility_Report_CN.md"
SOURCE_ROWS = ROOT / "results" / "phase6i_priority_tag_reward" / "scenario_rows.csv"
SOURCE_EXPERIMENT_COMMIT = "f4ec753d7611d72dc25ace3aeaa8e877d8ef92bb"
SOURCE_DIAGNOSTIC_COMMIT = "51adade38d8f3da5402e67506cf875d80224e82d"
INSTRUCTION_COMMIT = "6e58a6dd4bc2beb4c644841dfde809d96cfa9b3b"
FORMAL_ACTOR_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"
SCENARIO_ID = 321101
MODEL_SEED = 0
ACCOUNTING_TOLERANCE_BITS = 1.0
PATH_EQUAL_METERS = 1.0e-9
INHERITED_TORCH_THREADS = 1
INHERITED_PROCESS_SEED = 2

GATE_LOCALIZED = "REPLAY_DIVERGENCE_LOCALIZED"
GATE_INCONCLUSIVE = "REPLAY_DIVERGENCE_INCONCLUSIVE"
GATE_BLOCKED = "REPLAY_DIVERGENCE_BLOCKED"
SCHEMA_VERSION = "replay_reproducibility_v1"

WATCHED_RELATIVE = (
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/train.py",
    "src/wt/benchmark_v2/bit_queue_v2.py",
    "src/wt/l3b/env.py",
    "src/wt/lowlevel_v2/observation.py",
    "src/wt/lowlevel_v2/residual_policy.py",
    "src/wt/lowlevel_v2/safety_projector.py",
    "src/wt/benchmark_v2/phase6iaa_formal_planner/planner.py",
    "src/wt/round30/ppo.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_reward/evaluate.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_reward/rollout.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_reward/train.py",
    "src/wt/benchmark_v2/phase6i_priority_tag/follow.py",
    "src/wt/benchmark_v2/phase6i_priority_tag/ledger.py",
)
