"""Preregistered separate age and tagged-service bonuses. Coefficients stay fixed."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_direct_service_reward"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Direct_Tag_Reward_Experiment_Report_CN.md"
INSTRUCTION_COMMIT = "ac1545a2070aab767d5d609b12fd75c2e1ee4d2b"
ATTRIBUTION_COMMIT = "88c0e0296d6f64998a856af47f9c55dc6667c04b"
FORMAL_ACTOR_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"

LAMBDA_AGE = 0.0005
RHO_AGE = 0.00025
LAMBDA_TAG = 0.0010
RHO_TAG = 0.00075
TOTAL_BONUS_CAP = 0.001
PAYLOAD_CI_FLOOR_FRACTION = 0.005
REWARD_SCHEMA = "separate_age_and_direct_tag_service_v1"
VARIANT = "L3"
TRAIN_SEEDS = (0, 1, 2)
MIN_PASSING_SEEDS = 2
PREFLIGHT_SCENARIO = 321101

GATE_PASS = "DIRECT_TAG_REWARD_BENCHMARK_PASS"
GATE_FAIL = "DIRECT_TAG_REWARD_BENCHMARK_FAIL"
GATE_INCONCLUSIVE = "DIRECT_TAG_REWARD_BENCHMARK_INCONCLUSIVE"
GATE_BLOCKED = "DIRECT_TAG_REWARD_BLOCKED"
SCHEMA_VERSION = "separate_age_direct_tag_service_v1"

PRIOR_CHECKPOINT_SHA256 = {
    ("L0", 0): FORMAL_ACTOR_SHA256,
    ("L1", 0): "cedc9bc17b448639793a1d75d1a3c4681b7d060eb2659dfce5a66f2f675ac9c3",
    ("L1", 1): "b1574ddaec7ba4006dbcba0999031a60eea8e91323ac4c68f29ae08b8530aee7",
    ("L1", 2): "ee13fe4ba0b6fd4a9354b798eb1cd27ff5c4fb4f32459d1e3c5745c4c89f3583",
    ("L2", 0): "4a84fdc6db2f14b8dd29c57efcc861194ec3fbbc9e1e52c3047f8e2d77aab41f",
    ("L2", 1): "0a8f113de24a76a43bfd29246c39f3c6cd8dff85d9744b28a9d30f0c5c34eb7a",
    ("L2", 2): "55fe2acece8772f1bdbf4e02b228781c91892fc1228c292f7c86e5e78a1bea8c",
}
