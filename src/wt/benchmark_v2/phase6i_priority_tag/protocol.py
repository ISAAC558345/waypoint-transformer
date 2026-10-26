"""E1 priority-tag prototype. The formal queue transition stays untouched."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_prototype"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_PriorityTag_Prototype_Report_CN.md"
INSTRUCTION_COMMIT = "445ab1f5999279fba969a968ae115fd44d703f8e"

SCHEMA_VERSION = "synthetic_priority_bucket_v1"
TAG_RATE = 0.25
BETA_TAG = 0.25
TAG_SEED = 606
AGE_SCALE_STEPS = 48
BLOCK_BITS = 1.0e6
TOL_BITS = 1.0
DUST_BITS = 1.0e-8
DRAIN_FAIL_BITS = 1.0e-6
SMOKE_SCENARIO = 321000
SMOKE_STEPS = 16
LAMBDA_CANDIDATES = (0.0, 0.0005, 0.001, 0.002)
LAMBDA_REFERENCE = 0.05
LAMBDA_SELECTED = None

GATE_LEDGER_PASS = "PRIORITY_LEDGER_PROTOTYPE_PASS"
GATE_SCALE_READY = "URGENCY_REWARD_SCALE_READY"
GATE_BLOCKED = "PRIORITY_LEDGER_BLOCKED"


def choose_gate(*, ledger_ok: bool, tracking_ok: bool, scale_reported: bool) -> str:
    """A scale table is reportable only after the ledger follows the formal transition."""
    if not ledger_ok or not tracking_ok:
        return GATE_BLOCKED
    if not scale_reported:
        return GATE_BLOCKED
    return GATE_SCALE_READY
