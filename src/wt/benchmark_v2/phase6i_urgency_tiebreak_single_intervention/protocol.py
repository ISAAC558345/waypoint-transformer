"""Stage D4 limits. One tie-break is inserted into an otherwise formal Scheme C episode."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DECISION_TIMES, HOLD
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import EPS_BITS, GENERATOR, SELECTED_K
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.protocol import pilot_scenarios
from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.protocol import OUTPUT as D3_OUTPUT
from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.protocol import PILOT_SEED

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_urgency_tiebreak_single_intervention"
INSTRUCTION_COMMIT = "47e120eb0bbc9db35f2d2eafc26b3c88d8ba2771"
MAX_EPISODES = 16
MAX_SECONDS = 20.0 * 60.0
EPISODE_HORIZON = 48
INTERVENTION_TIMES = (0, 30)
CONTROL_ARM = "scheme_c"
GATE_COMPLETE = "URGENCY_TIEBREAK_SINGLE_INTERVENTION_COMPLETE"
GATE_INCONCLUSIVE = "URGENCY_TIEBREAK_SINGLE_INTERVENTION_INCONCLUSIVE"
GATE_BLOCKED = "URGENCY_TIEBREAK_SINGLE_INTERVENTION_BLOCKED"
ORACLE_LABEL = True


def scenarios() -> tuple[int, ...]:
    chosen = pilot_scenarios()
    if tuple(int(item) for item in chosen) != tuple(int(item) for item in range(322000, 322004)):
        raise RuntimeError("stage D4 expected train scenarios 322000-322003")
    return chosen


def arm_name(intervention_time: int | None) -> str:
    if intervention_time is None:
        return CONTROL_ARM
    if int(intervention_time) not in INTERVENTION_TIMES:
        raise RuntimeError(f"stage D4 refuses intervention time {intervention_time}")
    return f"intervention_{int(intervention_time)}"


def payload_sign(deltas: list[float], eps_bits: float = EPS_BITS) -> str:
    """Describe four paired payload gaps. A gap inside 1 bit is not a material loss."""
    values = [float(item) for item in deltas]
    if not values:
        raise RuntimeError("payload sign received no scenarios")
    if all(abs(value) <= float(eps_bits) for value in values):
        return "flat"
    if all(value <= float(eps_bits) for value in values) and any(value < -float(eps_bits) for value in values):
        return "consistent_nonpositive"
    if all(value >= -float(eps_bits) for value in values) and any(value > float(eps_bits) for value in values):
        return "consistent_nonnegative"
    return "mixed"


def mechanism_name(*, sign_0: str, sign_30: str, arrival_paired: bool) -> str:
    """Map the two intervention clocks onto one pre-registered explanation."""
    if not arrival_paired:
        return "unpaired_arrivals"
    if sign_0 == "mixed" or sign_30 == "mixed":
        return "unresolved"
    if sign_0 == "consistent_nonpositive" and sign_30 in {"flat", "consistent_nonnegative"}:
        return "initial_assignment"
    if sign_30 == "consistent_nonpositive" and sign_0 in {"flat", "consistent_nonnegative"}:
        return "mid_episode"
    if sign_0 == "consistent_nonpositive" and sign_30 == "consistent_nonpositive":
        return "both_clocks"
    if sign_0 in {"flat", "consistent_nonnegative"} and sign_30 in {"flat", "consistent_nonnegative"}:
        return "no_payload_loss"
    return "unresolved"


def choose_gate(*, mechanical_ok: bool, budget_ok: bool, mechanism: str) -> str:
    """A finished pair is inconclusive when the two clocks do not identify one mechanism."""
    if not mechanical_ok or not budget_ok:
        return GATE_BLOCKED
    if mechanism in {"unresolved", "unpaired_arrivals"}:
        return GATE_INCONCLUSIVE
    if mechanism in {"initial_assignment", "mid_episode", "both_clocks", "no_payload_loss"}:
        return GATE_COMPLETE
    return GATE_BLOCKED


__all__ = [
    "CONTROL_ARM",
    "D3_OUTPUT",
    "DECISION_TIMES",
    "EPISODE_HORIZON",
    "EPS_BITS",
    "GATE_BLOCKED",
    "GATE_COMPLETE",
    "GATE_INCONCLUSIVE",
    "GENERATOR",
    "HOLD",
    "INSTRUCTION_COMMIT",
    "INTERVENTION_TIMES",
    "MAX_EPISODES",
    "MAX_SECONDS",
    "ORACLE_LABEL",
    "OUTPUT",
    "PILOT_SEED",
    "SELECTED_K",
    "arm_name",
    "choose_gate",
    "mechanism_name",
    "payload_sign",
    "scenarios",
]
