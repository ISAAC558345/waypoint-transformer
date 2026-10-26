"""E4 reads committed E2/E3 records and the formal reward source. It does not train."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_reward_protocol_e4"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Reward_Protocol_Review_CN.md"
INSTRUCTION_COMMIT = "fe2967115cbb86e97e1b2d92c2c6bc47399681d0"
E2_COMMIT = "5210dd7611befceed37c897a998cc1c5bd2a76bd"
E3_COMMIT = "6492e9960af094dc28e71748ea4a5b6c8d988f86"
E2_GATE = "URGENCY_REWARD_SCALE_INCONCLUSIVE"
E3_GATE = "URGENCY_REWARD_FORM_INCONCLUSIVE"
E1_SCHEMA = "synthetic_priority_bucket_v1"

E2_FILES = (
    "results/phase6i_priority_tag_reward_scale_e2/step_rows.csv",
    "results/phase6i_priority_tag_reward_scale_e2/episode_rows.csv",
    "results/phase6i_priority_tag_reward_scale_e2/summary.json",
    "results/phase6i_priority_tag_reward_scale_e2/schema.json",
    "docs/research/HTW_UrgencyAware_Reward_Scale_Replication_Report_CN.md",
)
E3_FILES = (
    "results/phase6i_priority_tag_reward_form_e3/form_rows.csv",
    "results/phase6i_priority_tag_reward_form_e3/summary.json",
    "results/phase6i_priority_tag_reward_form_e3/schema.json",
    "docs/research/HTW_UrgencyAware_Reward_Form_Comparison_Report_CN.md",
)
E1_SCHEMA_FILE = "results/phase6i_priority_tag_prototype/schema.json"
FORMAL_FILES = (
    "src/wt/lowlevel_v2/phase1/protocol.py",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/protocol.py",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py",
    "src/wt/benchmark_v2/phase6iaa_formal_planner/planner.py",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt",
)

LAMBDA_CANDIDATES = (0.0005, 0.001, 0.002)
RHO_CANDIDATES = (0.001, 0.0025, 0.005)
CONDITIONAL_LAMBDA = 0.0005
CONDITIONAL_RHO = 0.001
CANDIDATE_STATUS = "CONDITIONAL_CANDIDATE_NOT_APPROVED"
LAMBDA_SELECTED = None
RHO_SELECTED = None
OWNER_APPROVED = False

MEAN_BONUS_OVER_PAYLOAD_MAX = 0.01
P95_BONUS_OVER_PENALTY_MAX = 1.0
PAYLOAD_TIE = 1.0e-6
SIGNAL_RETAINED_MIN = 0.5
LAMBDA_PROJECT = 0.05
LAMBDA_RESIDUAL = 0.01
PAYLOAD_SCALE = 1.0e6
ACTION_SCALE = 1.0

GATE_READY = "URGENCY_REWARD_PROTOCOL_READY"
GATE_WAITING = "URGENCY_REWARD_PROTOCOL_WAITING_FOR_OWNER"
GATE_BLOCKED = "URGENCY_REWARD_PROTOCOL_BLOCKED"


def choose_gate(*, sources_ok: bool, owner_approved: bool) -> str:
    """This stage stays waiting unless the owner has already approved the candidate."""
    if not sources_ok:
        return GATE_BLOCKED
    if not owner_approved:
        return GATE_WAITING
    return GATE_READY
