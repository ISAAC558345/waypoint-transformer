"""Explicit evaluation context for one frozen L0 scenario. Seeds keep their historical roles."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_eval_context"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Eval_Context_Repro_Report_CN.md"
SOURCE_ROWS = ROOT / "results" / "phase6i_priority_tag_reward" / "scenario_rows.csv"
SOURCE_E7_COMMIT = "64ae1907eae2354d64625361c0a0afd866982d42"
INSTRUCTION_COMMIT = "88ceeab4dc2ac6f0e19fc2603d9fdf915ee0590d"
FORMAL_ACTOR_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"

SCENARIO_ID = 321101
REPORTED_MODEL_SEED = 0
ENV_RESET_SEED = 321101
ASSIGNMENT_RNG_SEED = 7_000_000 + 321101
HISTORICAL_INHERITED_GLOBAL_SEED = 2
TORCH_THREADS = 1
ACCOUNTING_TOLERANCE_BITS = 1.0
PATH_TOLERANCE_M = 1.0e-9

GATE_PASS = "EVAL_CONTEXT_REPRO_PASS"
GATE_INCONCLUSIVE = "EVAL_CONTEXT_REPRO_INCONCLUSIVE"
GATE_BLOCKED = "EVAL_CONTEXT_REPRO_BLOCKED"
SCHEMA_VERSION = "explicit_eval_context_v1"
CONTEXT_LABEL = "historical_inherited_process_context"
GLOBAL_SEED_ROLE = "historical_inherited_process_seed"

HASHED_RELATIVE = (
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/protocol.py",
    "src/wt/benchmark_v2/bit_queue_v2.py",
    "src/wt/l3b/env.py",
    "src/wt/lowlevel_v2/observation.py",
    "src/wt/lowlevel_v2/residual_policy.py",
    "src/wt/lowlevel_v2/safety_projector.py",
    "src/wt/round30/ppo.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_reward/protocol.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_reward/evaluate.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_reward/rollout.py",
    "src/wt/benchmark_v2/phase6i_priority_tag/ledger.py",
)
