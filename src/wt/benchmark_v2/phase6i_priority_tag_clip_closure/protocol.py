"""Closure of the unadopted clip-group line. No new training budget."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_clip_closure"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Clip_Group_Closure_Report_CN.md"
E15_OUTPUT = ROOT / "results" / "phase6i_priority_tag_clip_benchmark"
E14_OUTPUT = ROOT / "results" / "phase6i_priority_tag_optimizer_intervention"

INSTRUCTION_COMMIT = "f36fc9f9e13bd32a9f731d0f73c2a8b76f66918c"
E15_RESULT_COMMIT = "1936e57a72ec7e82692639ed98c96664daae23a7"
E15_IMPLEMENTATION_COMMIT = "e2ca6806977aa05e36786a83e845562f543a448b"
FORMAL_ACTOR_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"

GATE_CLOSED = "CLIP_GROUP_LINE_CLOSED_UNADOPTED"
GATE_BLOCKED = "CLIP_GROUP_CLOSURE_BLOCKED"
E15_GATE = "CLIP_GROUP_BENCHMARK_INCONCLUSIVE"
SCHEMA_VERSION = "clip_group_line_closure_v1"

D_SAFE_M = 1.0
COUNTER_EPS_M = 1.0e-9
PROJECTOR_ACCEPT_EPS_M = 1.0e-6
DEV_SCENARIOS = tuple(range(321096, 321128))
MBIT = 1.0e6

PUBLISHED_TAGGED_MBIT = (
    (0.404471, -0.198182, 1.003285, 0.356634, -0.247417, 0.956496),
    (0.031817, -0.574684, 0.635670, 0.153311, -0.556069, 0.836279),
    (0.531566, -0.285215, 1.435222, 0.266759, -0.609215, 1.224307),
)
PUBLISHED_PAYLOAD_MBIT = (0.375124, -0.461748, -0.161873)
