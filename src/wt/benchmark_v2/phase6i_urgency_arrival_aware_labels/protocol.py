"""Stage D5 limits. Arrival-aware labels are a diagnosis, not a training target."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import EPS_BITS, GENERATOR, HOLD, SELECTED_K
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.protocol import (
    FROZEN_SHARD_COMMIT,
    SAMPLE_K,
    TOP_K,
    pilot_scenarios,
)

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_urgency_arrival_aware_labels"
INSTRUCTION_COMMIT = "a3d3a278638c6170851d8c23d9c518bb546e1cbb"
LABEL_SEED = 604
DECISION_TIME = 30
HORIZON = int(HOLD)
MAX_CANDIDATES = int(TOP_K + SAMPLE_K)
MAX_ROLLOUTS = 1024
MAX_STATE_SECONDS = 900.0
MAX_SECONDS = 20.0 * 60.0
MATERIAL_REVERSAL = 0.05
SMALL_REVERSAL = 0.01
GATE_COMPLETE = "ARRIVAL_AWARE_URGENCY_LABEL_COMPLETE"
GATE_INCONCLUSIVE = "ARRIVAL_AWARE_URGENCY_LABEL_INCONCLUSIVE"
GATE_BLOCKED = "ARRIVAL_AWARE_URGENCY_LABEL_BLOCKED"


def scenarios() -> tuple[int, ...]:
    chosen = pilot_scenarios()
    if tuple(int(item) for item in chosen) != tuple(range(322000, 322004)):
        raise RuntimeError("stage D5 expected train scenarios 322000-322003")
    return chosen


def scenario_effect(
    *,
    urgent_reversal: float,
    payload_reversal: float,
    payload_top1_within_one_bit: bool,
    urgent_top1_within_one_bit: bool,
) -> str:
    """Classify one scenario before any cross-scenario story is chosen."""
    material = float(urgent_reversal) >= float(MATERIAL_REVERSAL) or not bool(urgent_top1_within_one_bit)
    small = (
        float(urgent_reversal) < float(SMALL_REVERSAL)
        and float(payload_reversal) < float(SMALL_REVERSAL)
        and bool(payload_top1_within_one_bit)
        and bool(urgent_top1_within_one_bit)
    )
    if material and not small:
        return "material"
    if small and not material:
        return "small"
    return "intermediate"


def study_effect(effects: list[str]) -> str:
    if effects and all(item == "material" for item in effects):
        return "material_consistent"
    if effects and all(item == "small" for item in effects):
        return "small"
    return "unstable"


def choose_gate(*, mechanical_ok: bool, budget_ok: bool, effect: str) -> str:
    """A finished measurement stays inconclusive when the ranking change is not stable."""
    if not mechanical_ok or not budget_ok:
        return GATE_BLOCKED
    if effect == "material_consistent":
        return GATE_COMPLETE
    if effect in {"small", "unstable"}:
        return GATE_INCONCLUSIVE
    return GATE_BLOCKED


__all__ = [
    "DECISION_TIME",
    "EPS_BITS",
    "FROZEN_SHARD_COMMIT",
    "GATE_BLOCKED",
    "GATE_COMPLETE",
    "GATE_INCONCLUSIVE",
    "GENERATOR",
    "HORIZON",
    "INSTRUCTION_COMMIT",
    "LABEL_SEED",
    "MATERIAL_REVERSAL",
    "MAX_CANDIDATES",
    "MAX_ROLLOUTS",
    "MAX_SECONDS",
    "MAX_STATE_SECONDS",
    "OUTPUT",
    "SAMPLE_K",
    "SELECTED_K",
    "SMALL_REVERSAL",
    "TOP_K",
    "choose_gate",
    "scenario_effect",
    "scenarios",
    "study_effect",
]
