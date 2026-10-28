"""Rules locked before training. Coefficients and gates are not revised after results."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_queue_lookahead"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Queue_Lookahead_Experiment_Report_CN.md"
E10_ROWS = ROOT / "results" / "phase6i_priority_tag_direct_service_reward" / "scenario_rows.csv"
INSTRUCTION_COMMIT = "ee356388b4b2ee4ba16347a9f5d0cedae9509ace"
E11_RESULT_COMMIT = "bd9ffe66a4ecff5562ccb9d3f1629d58891687a9"
FORMAL_ACTOR_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"

GATE_PASS = "QUEUE_LOOKAHEAD_BENCHMARK_PASS"
GATE_FAIL = "QUEUE_LOOKAHEAD_BENCHMARK_FAIL"
GATE_INCONCLUSIVE = "QUEUE_LOOKAHEAD_BENCHMARK_INCONCLUSIVE"
GATE_BLOCKED = "QUEUE_LOOKAHEAD_BLOCKED"
SCHEMA_VERSION = "queue_lookahead_observation_v1"

VARIANTS = ("L4_C", "L4_Q")
TRAIN_SEEDS = (0, 1, 2)
DIAGNOSTIC_SCENARIOS = (321000, 321001)
SAFETY_CASES = (
    {"variant": "L3", "model_seed": 1, "scenario_id": 321123, "baseline_variant": "L0", "baseline_seed": 0},
    {"variant": "L3", "model_seed": 2, "scenario_id": 321104, "baseline_variant": "L0", "baseline_seed": 0},
)
HEAD_FEATURE_COUNT = 3
LOOKAHEAD_NAMES = (
    "assigned_tagged_backlog_fraction",
    "assigned_fifo_bits_before_next_tag_fraction",
)
ACTOR_FEATURE_COUNT = HEAD_FEATURE_COUNT + len(LOOKAHEAD_NAMES)
UPDATES = 300
MIN_PASSING_SEEDS = 2
PAYLOAD_CI_FLOOR_FRACTION = 0.005
SAMPLE_UPDATES = (1, 150, 300)
SAMPLE_ENV_INDEX = 0
BIT_TOLERANCE = 1.0
PATH_TOLERANCE_M = 1.0e-9
COUNTER_TOLERANCE_M = 1.0e-9
PROJECTOR_TOLERANCE_M = 1.0e-6
DISTANCE_THRESHOLD_M = 1.0
