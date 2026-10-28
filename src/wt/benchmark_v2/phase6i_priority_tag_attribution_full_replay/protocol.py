"""E9 settings. Tolerances and seed roles stay as validated in E8."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_attribution_full_replay"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Reward_Attribution_Report_CN.md"
E8_CONTEXT = ROOT / "results" / "phase6i_priority_tag_eval_context" / "runtime_context.json"
SOURCE_RESULT_COMMIT = "f4ec753d7611d72dc25ace3aeaa8e877d8ef92bb"
E8_COMMIT = "b3ff3842c2d6611f575a43002a6cc9a4482573d1"
INSTRUCTION_COMMIT = "9eca4e10fcc14f2194a61bc64ece3ab1014fdabd"

FORMAL_ACTOR_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"
CHECKPOINT_SHA256 = {
    ("L0", 0): FORMAL_ACTOR_SHA256,
    ("L1", 0): "cedc9bc17b448639793a1d75d1a3c4681b7d060eb2659dfce5a66f2f675ac9c3",
    ("L1", 1): "b1574ddaec7ba4006dbcba0999031a60eea8e91323ac4c68f29ae08b8530aee7",
    ("L1", 2): "ee13fe4ba0b6fd4a9354b798eb1cd27ff5c4fb4f32459d1e3c5745c4c89f3583",
    ("L2", 0): "4a84fdc6db2f14b8dd29c57efcc861194ec3fbbc9e1e52c3047f8e2d77aab41f",
    ("L2", 1): "0a8f113de24a76a43bfd29246c39f3c6cd8dff85d9744b28a9d30f0c5c34eb7a",
    ("L2", 2): "55fe2acece8772f1bdbf4e02b228781c91892fc1228c292f7c86e5e78a1bea8c",
}

PREFLIGHT_VARIANT = "L0"
PREFLIGHT_MODEL_SEED = 0
PREFLIGHT_SCENARIO = 321101
HISTORICAL_INHERITED_GLOBAL_SEED = 2
GLOBAL_SEED_ROLE = "historical_inherited_process_seed"
CONTEXT_LABEL = "historical_inherited_process_context"
TORCH_THREADS = 1
EVAL_ASSIGNMENT_SEED = 7_000_000
EPISODE_STEPS = 48
FULL_EPISODE_COUNT = 224
TIME_LIMIT_SECONDS = 600.0
ACCOUNTING_TOLERANCE_BITS = 1.0
PATH_TOLERANCE_M = 1.0e-9
DECOMPOSITION_TOLERANCE_BITS = 1.0e-6

GATE_COMPLETE = "PRIORITY_TAG_ATTRIBUTION_COMPLETE"
GATE_INCONCLUSIVE = "PRIORITY_TAG_ATTRIBUTION_INCONCLUSIVE"
GATE_BLOCKED = "PRIORITY_TAG_ATTRIBUTION_BLOCKED"
SCHEMA_VERSION = "synthetic_priority_full_replay_attribution_v1"

BIT_FIELDS = (
    "payload_bits",
    "urgency_served_bits",
    "served_tagged_bits",
    "served_ordinary_bits",
    "dropped_tagged_bits",
    "dropped_ordinary_bits",
    "dropped_bits",
    "final_queue_bits",
    "final_tagged_queue_bits",
)
