"""E2 frozen-policy reward scale. No weight is selected and nothing is trained."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_reward_scale_e2"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Reward_Scale_Replication_Report_CN.md"
E1_SUMMARY = ROOT / "results" / "phase6i_priority_tag_prototype" / "summary.json"
E1_REWARD_SCALE = ROOT / "results" / "phase6i_priority_tag_prototype" / "reward_scale.json"
INSTRUCTION_COMMIT = "ab70a42326ecc06de51f678b522ec9bfb560ccb1"
CHECKPOINT = ROOT / "src" / "wt" / "benchmark_v2" / "phase6g_lowlevel_mappo" / "checkpoints" / "phase6g_seed0.pt"

EPISODE_COUNT = 8
EPISODE_STEPS = 48
BUDGET_SECONDS = 600.0
ASSIGNMENT_SEED_BASE = 6_000_000
PENALTY_DUST = 1.0e-12
PAYLOAD_TIE_BITS = 1.0e-6
RATIO_SPAN_MAX = 2.0
E1_RELATIVE_TOLERANCE = 0.50
P95_BONUS_OVER_PENALTY_MAX = 1.0
MEAN_BONUS_OVER_PAYLOAD_MAX = 0.01

LAMBDA_CANDIDATES = (0.0, 0.0005, 0.001, 0.002)
LAMBDA_REFERENCE = 0.05
LAMBDA_SELECTED = None

GATE_REPLICATED = "URGENCY_REWARD_SCALE_REPLICATED"
GATE_INCONCLUSIVE = "URGENCY_REWARD_SCALE_INCONCLUSIVE"
GATE_BLOCKED = "URGENCY_REWARD_SCALE_BLOCKED"

WATCHED = (
    "src/wt/benchmark_v2/bit_queue_v2.py",
    "src/wt/benchmark_v2/phase6i_urgency_stage_a/ledger.py",
    "src/wt/lowlevel_v2/observation.py",
    "src/wt/lowlevel_v2/critic_state.py",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py",
    "src/wt/benchmark_v2/phase6iaa_formal_planner/planner.py",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt",
    "results/phase6i_urgency_label_rule_design/summary.json",
    "results/phase6i_urgency_lowlevel/smoke_summary.json",
)


def choose_gate(
    *,
    integrity_ok: bool,
    episodes: int,
    budget_exceeded: bool,
    positive_candidate_supported: bool,
    ratio_span_ok: bool,
    e1_scale_close: bool,
) -> str:
    """Replication requires a complete frozen-policy sample and a stable conservative bound."""
    if budget_exceeded or not integrity_ok or int(episodes) != int(EPISODE_COUNT):
        return GATE_BLOCKED
    if not positive_candidate_supported or not ratio_span_ok or not e1_scale_close:
        return GATE_INCONCLUSIVE
    return GATE_REPLICATED
