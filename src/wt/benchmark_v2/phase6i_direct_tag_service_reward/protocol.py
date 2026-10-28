"""Preregistered L3 settings. Thresholds are fixed before training results exist."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_direct_tag_service_reward"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Direct_Tag_Reward_Report_CN.md"
SOURCE_ROWS = ROOT / "results" / "phase6i_priority_tag_reward" / "scenario_rows.csv"
INSTRUCTION_COMMIT = "dfa8dcb1d51177f46450cac17a17314d05897566"
ATTRIBUTION_COMMIT = "88c0e0296d6f64998a856af47f9c55dc6667c04b"
FORMAL_ACTOR_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"

# Single candidate. Not a searched optimum.
LAMBDA_TAG = 0.002
RHO_TAG = 0.001
REWARD_SCHEMA = "direct_tagged_service_bounded_v1"
VARIANT = "L3"
TRAIN_SEEDS = (0, 1, 2)
MIN_PASSING_SEEDS = 2
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
ACCOUNTING_TOLERANCE_BITS = 1.0
PATH_TOLERANCE_M = 1.0e-9
PREFLIGHT_SCENARIO = 321101

# Section 6, operationalized before any L3 result:
# payload CI low >= 0 is the reading of "no substantial negative degradation".
# final queue paired mean <= 0 is the reading of "no substantial queue worsening".
# SINR, interference, drops, safety, and constraints use the same sign rules as the prior experiment.
TAGGED_MEAN_MIN = 0.0
TAGGED_CI_LOW_MIN = 0.0
PAYLOAD_MEAN_MIN = 0.0
PAYLOAD_CI_LOW_MIN = 0.0
SAFETY_DIFF_MAX = 0.0
CONSTRAINT_DIFF_MAX = 0.0
DROPPED_DIFF_MAX = 0.0
SINR_DIFF_MIN = 0.0
INTERFERENCE_DIFF_MAX = 0.0
QUEUE_DIFF_MAX = 0.0

GATE_PASS = "DIRECT_TAG_REWARD_PASS"
GATE_INCONCLUSIVE = "DIRECT_TAG_REWARD_INCONCLUSIVE"
GATE_BLOCKED = "DIRECT_TAG_REWARD_BLOCKED"
SCHEMA_VERSION = "direct_tagged_service_reward_v1"

PRIOR_CHECKPOINT_SHA256 = {
    ("L0", 0): FORMAL_ACTOR_SHA256,
    ("L1", 0): "cedc9bc17b448639793a1d75d1a3c4681b7d060eb2659dfce5a66f2f675ac9c3",
    ("L1", 1): "b1574ddaec7ba4006dbcba0999031a60eea8e91323ac4c68f29ae08b8530aee7",
    ("L1", 2): "ee13fe4ba0b6fd4a9354b798eb1cd27ff5c4fb4f32459d1e3c5745c4c89f3583",
    ("L2", 0): "4a84fdc6db2f14b8dd29c57efcc861194ec3fbbc9e1e52c3047f8e2d77aab41f",
    ("L2", 1): "0a8f113de24a76a43bfd29246c39f3c6cd8dff85d9744b28a9d30f0c5c34eb7a",
    ("L2", 2): "55fe2acece8772f1bdbf4e02b228781c91892fc1228c292f7c86e5e78a1bea8c",
}
