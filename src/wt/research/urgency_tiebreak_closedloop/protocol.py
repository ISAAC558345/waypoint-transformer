"""Stage D3 limits. The tie-break cannot spend payload, and lambda_H stays unset."""
from __future__ import annotations

from pathlib import Path

from wt.control.mappo.protocol import DECISION_TIMES, HOLD
from wt.control.executor.protocol import primary_checkpoint
from wt.planning.formal_planner.protocol import (
    ADAPTER_CONFIG_SHA256,
    EPOCH6_SHA256,
    MAPPO_SHA256,
    SCHEME_C_SHA256,
)
from wt.data.highlevel_graphs.protocol import EPS_BITS, GENERATOR, SELECTED_K
from wt.research.urgency_highlevel_pilot.protocol import (
    FROZEN_SHARD_COMMIT,
    PER_STRATUM,
    SAMPLE_K,
    STRATA,
    TOP_K,
    pilot_scenarios,
)
from wt.models.assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT
from wt.planning.candidate_adapter.protocol import config_sha256
from wt.common.paths import repo_root

ROOT = repo_root()
OUTPUT = ROOT / "results" / "phase6i_urgency_tiebreak_closedloop_pilot"
INSTRUCTION_COMMIT = "cb264320323bbc15b3763f53b0f664bc8421d994"
PILOT_SEED = 603
EPISODE_HORIZON = 48
LABEL_HORIZON_FULL = int(HOLD)
LABEL_HORIZON_FINAL = 3
MAX_CANDIDATES = int(TOP_K + SAMPLE_K)
MAX_STATE_SECONDS = 900.0
MAX_ESTIMATED_SECONDS = 1800.0
# Four shared opening states plus two arms at the three later decision times.
PLANNED_LABEL_STATES = 4 + (2 * 4 * 3)
PAYLOAD_SCALE = 1.0e6
ARMS = ("payload_first", "payload_first_urgency_tiebreak")
GATE_COMPLETE = "URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_COMPLETE"
GATE_INCONCLUSIVE = "URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_INCONCLUSIVE"
GATE_BLOCKED = "URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_BLOCKED"
ORACLE_LABEL = True
LAMBDA_H = None
DELTA_PAYLOAD_MAX = None


def scenarios() -> tuple[int, ...]:
    """The same four train scenarios already checked against the registry."""
    chosen = pilot_scenarios()
    if len(chosen) != 4:
        raise RuntimeError("stage D3 expected the four registered train scenarios")
    return chosen


def label_horizon(time_step: int, episode_horizon: int = EPISODE_HORIZON) -> int:
    remaining = int(episode_horizon) - int(time_step)
    if int(time_step) == int(DECISION_TIMES[-1]):
        if remaining != int(LABEL_HORIZON_FINAL):
            raise RuntimeError("the last decision does not have a 3-step remainder")
        return int(LABEL_HORIZON_FINAL)
    if int(time_step) not in DECISION_TIMES:
        raise RuntimeError(f"stage D3 refuses decision time {time_step}")
    if remaining < int(LABEL_HORIZON_FULL):
        raise RuntimeError("a 15-step label does not fit before the horizon")
    return int(LABEL_HORIZON_FULL)


def choose_gate(*, mechanical_ok: bool, budget_ok: bool, tiebreak_fraction: float | None, urgent_direction: str) -> str:
    """A finished run can still be inconclusive when the tie-break rarely changes the action."""
    if not mechanical_ok or not budget_ok:
        return GATE_BLOCKED
    if tiebreak_fraction is None or float(tiebreak_fraction) < 0.5 or urgent_direction != "consistent_nonnegative":
        return GATE_INCONCLUSIVE
    return GATE_COMPLETE


__all__ = [
    "ADAPTER_CONFIG_SHA256",
    "ARMS",
    "DECISION_TIMES",
    "DELTA_PAYLOAD_MAX",
    "EPOCH6_CKPT",
    "EPOCH6_SHA256",
    "EPISODE_HORIZON",
    "EPS_BITS",
    "FROZEN_SHARD_COMMIT",
    "GATE_BLOCKED",
    "GATE_COMPLETE",
    "GATE_INCONCLUSIVE",
    "GENERATOR",
    "INSTRUCTION_COMMIT",
    "LABEL_HORIZON_FINAL",
    "LABEL_HORIZON_FULL",
    "LAMBDA_H",
    "MAPPO_SHA256",
    "MAX_CANDIDATES",
    "MAX_ESTIMATED_SECONDS",
    "MAX_STATE_SECONDS",
    "ORACLE_LABEL",
    "OUTPUT",
    "PAYLOAD_SCALE",
    "PER_STRATUM",
    "PILOT_SEED",
    "PLANNED_LABEL_STATES",
    "SAMPLE_K",
    "SCHEME_C_CKPT",
    "SCHEME_C_SHA256",
    "SELECTED_K",
    "STRATA",
    "TOP_K",
    "choose_gate",
    "config_sha256",
    "label_horizon",
    "primary_checkpoint",
    "scenarios",
]