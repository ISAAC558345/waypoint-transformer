"""Stage D2 rules. No loss budget or lambda_H is chosen here."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_urgency_utility_rule"
D0_LABELS = ROOT / "results" / "phase6i_urgency_highlevel_pilot" / "labels.csv"
D0_SELECTED = ROOT / "results" / "phase6i_urgency_highlevel_pilot" / "selected_candidates.csv"
D0_SCHEMA = ROOT / "results" / "phase6i_urgency_highlevel_pilot" / "schema.json"
D1_LABELS = ROOT / "results" / "phase6i_urgency_highlevel_t30_pilot" / "labels.csv"
D1_SELECTED = ROOT / "results" / "phase6i_urgency_highlevel_t30_pilot" / "selected_candidates.csv"
D1_SCHEMA = ROOT / "results" / "phase6i_urgency_highlevel_t30_pilot" / "schema.json"
D1_SUMMARY = ROOT / "results" / "phase6i_urgency_highlevel_t30_pilot" / "summary.json"
INSTRUCTION_COMMIT = "38c9b4586bdb3df782d97fbcfeba1b4c09dfc084"
D0_SCENARIOS = tuple(range(322000, 322004))
D0_TIMES = (0, 15, 30)
D1_SCENARIOS = tuple(range(322004, 322016))
D1_TIME = 30
HOLDOUT_SCENARIO = 322015
EPS_BITS = 1.0
PAYLOAD_SCALE = 1.0e6
EPS_TERM = EPS_BITS / PAYLOAD_SCALE
LAMBDA_H = None
DELTA_PAYLOAD_MAX = None
GATE_READY = "HIGHLEVEL_UTILITY_RULE_READY"
GATE_INCONCLUSIVE = "HIGHLEVEL_UTILITY_RULE_INCONCLUSIVE"
GATE_BLOCKED = "HIGHLEVEL_UTILITY_RULE_BLOCKED"


def choose_gate(*, inputs_ok: bool, frontier_ok: bool, rules_ok: bool) -> str:
    """A finished comparison can be ready while lambda_H stays unset."""
    if not inputs_ok:
        return GATE_BLOCKED
    if not frontier_ok or not rules_ok:
        return GATE_INCONCLUSIVE
    return GATE_READY
