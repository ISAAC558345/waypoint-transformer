"""E3 offline reward-form comparison. No trajectory is regenerated."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_reward_form_e3"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Reward_Form_Comparison_Report_CN.md"
E2_DIR = ROOT / "results" / "phase6i_priority_tag_reward_scale_e2"
E2_COMMIT = "5210dd7611befceed37c897a998cc1c5bd2a76bd"
INSTRUCTION_COMMIT = "a85bd2a96a81b5545bfe5f2821f947d80de4ff68"
SOURCE_NAMES = ("step_rows.csv", "episode_rows.csv", "summary.json", "schema.json")
EXPECTED_SCHEMA = "frozen_mappo_priority_tag_reward_scale_v1"
EXPECTED_GATE = "URGENCY_REWARD_SCALE_INCONCLUSIVE"
EXPECTED_SCENARIOS = tuple(range(321000, 321008))
EXPECTED_STEPS_PER_EPISODE = 48
EXPECTED_STEP_ROWS = 384
EXPECTED_EPISODE_ROWS = 8

LAMBDA_FORM_A = (0.0, 0.0005, 0.001, 0.002)
LAMBDA_FORM_B = (0.0005, 0.001, 0.002)
LAMBDA_REFERENCE = 0.05
RHO_CANDIDATES = (0.001, 0.0025, 0.005)
LAMBDA_SELECTED = None
RHO_SELECTED = None

MEAN_BONUS_OVER_PAYLOAD_MAX = 0.01
P95_BONUS_OVER_PENALTY_MAX = 1.0
PAYLOAD_TIE = 1.0e-6
PENALTY_DUST = 1.0e-12
SIGNAL_RETAINED_MIN = 0.5
CROSSCHECK_ATOL = 1.0e-8

GATE_READY = "URGENCY_REWARD_FORM_READY"
GATE_INCONCLUSIVE = "URGENCY_REWARD_FORM_INCONCLUSIVE"
GATE_BLOCKED = "URGENCY_REWARD_FORM_BLOCKED"


def choose_gate(*, source_ok: bool, analysis_ok: bool, any_candidate_meets_standard: bool) -> str:
    """A ready form must already meet the preregistered conservative standard."""
    if not source_ok or not analysis_ok:
        return GATE_BLOCKED
    if not any_candidate_meets_standard:
        return GATE_INCONCLUSIVE
    return GATE_READY
