"""Diagnostic settings for the urgency-reward attribution replay."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_reward_attribution"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Reward_Attribution_Report_CN.md"
SOURCE_OUTPUT = ROOT / "results" / "phase6i_priority_tag_reward"
SOURCE_ROWS = SOURCE_OUTPUT / "scenario_rows.csv"
SOURCE_SUMMARY = SOURCE_OUTPUT / "summary.json"
SOURCE_REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Experiment_Report_CN.md"

SOURCE_RESULT_COMMIT = "f4ec753d7611d72dc25ace3aeaa8e877d8ef92bb"
INSTRUCTION_COMMIT = "2d7572ac9a5a134c763d7780a853e858a297dd25"
FORMAL_ACTOR_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"

TIME_LIMIT_SECONDS = 600.0
ACCOUNTING_TOLERANCE_BITS = 1.0
SUMMARY_MATCH_TOLERANCE = 1.0e-4
FULL_SATURATION_EPSILON = 1.0e-9
BOOST_SHORTFALL_EPSILON = 1.0e-9

GATE_COMPLETE = "URGENCY_REWARD_ATTRIBUTION_COMPLETE"
GATE_BLOCKED = "URGENCY_REWARD_ATTRIBUTION_BLOCKED"
GATE_TIME_LIMIT = "URGENCY_REWARD_ATTRIBUTION_TIME_LIMIT"

SCHEMA_VERSION = "synthetic_priority_reward_attribution_v1"
