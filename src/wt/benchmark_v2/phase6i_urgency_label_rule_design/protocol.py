"""Stage D7 offline label-rule design. No rollout and no chosen loss budget."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_urgency_label_rule_design"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_HighLevel_Label_Rule_Design_Report_CN.md"
INSTRUCTION = ROOT / "docs" / "research" / "HTW_UrgencyAware_HighLevel_Label_Rule_Design_Instructions_CN.md"
INSTRUCTION_COMMIT = "d75872b1f53e2d899a6484e927bac2093e9abfe0"

EPS_BITS = 1.0
LAMBDA_H = None
DELTA_PAYLOAD_MAX = None
FRACTIONS = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)
PRIMARY_SOURCE = "D6"
PRIMARY_TIMES = (15, 30)
SEPARATE_TIMES = (0, 45)
CANDIDATES_PER_STATE = 128
D6_POOL_COUNT = 13327

GATE_READY = "HIGHLEVEL_LABEL_RULE_DESIGN_READY"
GATE_INCONCLUSIVE = "HIGHLEVEL_LABEL_RULE_DESIGN_INCONCLUSIVE"
GATE_BLOCKED = "HIGHLEVEL_LABEL_RULE_DESIGN_BLOCKED"


def choose_gate(
    *,
    sources_verified: bool,
    split_ok: bool,
    sensitivity_complete: bool,
    fields_support_rules: bool,
    delta_unset: bool,
) -> str:
    """A finished design stays ready only while the loss budget is still unset."""
    if not sources_verified or not split_ok or not delta_unset:
        return GATE_BLOCKED
    if not sensitivity_complete or not fields_support_rules:
        return GATE_INCONCLUSIVE
    return GATE_READY
