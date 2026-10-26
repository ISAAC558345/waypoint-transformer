"""Stage D6 limits. Thresholds are fixed before any new rollout."""
from __future__ import annotations

import json

from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import (
    DEV_TEST_RESERVED,
    EPS_BITS,
    GENERATOR,
    HOLD,
    PRIMARY_TIMES,
    SELECTED_K,
    TRAIN_POOL,
    VALIDATION_POOL,
    assert_allowed,
)
from wt.benchmark_v2.phase6i_urgency_arrival_aware_labels.protocol import FROZEN_SHARD_COMMIT
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.protocol import (
    REGISTRY,
    ROOT,
    SAMPLE_K,
    TOP_K,
    pilot_scenarios,
)
from wt.benchmark_v2.phase6i_urgency_highlevel_t30_pilot.protocol import t30_scenarios

OUTPUT = ROOT / "results" / "phase6i_urgency_arrival_aware_expansion"
INSTRUCTION_COMMIT = "327eb05376b2e1f62581e09526e5a9b0bd782fe2"
LABEL_SEED = 605
DECISION_TIMES = tuple(PRIMARY_TIMES) + (45,)
MAX_CANDIDATES = int(TOP_K + SAMPLE_K)
SCENARIO_COUNT = 8
MAX_STATES = SCENARIO_COUNT * len(DECISION_TIMES)
# Two arms on 32 states. The verification repeat is the same 128 candidates, counted apart.
MAX_STUDY_ROLLOUTS = MAX_STATES * MAX_CANDIDATES * 2
MAX_VERIFICATION_ROLLOUTS = MAX_CANDIDATES * 2
MAX_STATE_SECONDS = 120.0
MAX_SECONDS = 20.0 * 60.0
MATERIAL_REVERSAL = 0.05
SMALL_REVERSAL = 0.01
EXPECTED_SCENARIOS = tuple(range(322016, 322024))
REPEAT_TIME = 30
GATE_COMPLETE = "ARRIVAL_AWARE_LABEL_EXPANSION_COMPLETE"
GATE_INCONCLUSIVE = "ARRIVAL_AWARE_LABEL_EXPANSION_INCONCLUSIVE"
GATE_BLOCKED = "ARRIVAL_AWARE_LABEL_EXPANSION_BLOCKED"
D5_GATE = "ARRIVAL_AWARE_URGENCY_LABEL_COMPLETE"


def horizon_for(time_step: int) -> int:
    """t=45 keeps the real remaining 3 steps. Earlier decisions keep the 15-step label."""
    value = int(time_step)
    if value == 45:
        return 3
    if value in PRIMARY_TIMES:
        return int(HOLD)
    raise RuntimeError(f"expansion refuses time {value}")


def assert_same_horizon(time_steps) -> int:
    """15-step labels and 3-step labels stay in separate summaries."""
    horizons = {horizon_for(item) for item in time_steps}
    if len(horizons) != 1:
        raise RuntimeError("15-step labels and 3-step labels cannot be pooled")
    return int(next(iter(horizons)))


def scenarios() -> tuple[int, ...]:
    """First eight train ids not used by D0, D1, or D5. The registry guard stays in front."""
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    train = tuple(int(item) for item in payload["highlevel_v2_train"])
    validation = tuple(int(item) for item in payload["highlevel_v2_validation"])
    reserved = tuple(int(item) for item in payload["highlevel_v2_dev_test_reserved"])
    if train != tuple(sorted(train)):
        raise RuntimeError("registry train ids are not sorted by scenario id")
    if train != tuple(int(item) for item in TRAIN_POOL):
        raise RuntimeError("registry train pool does not match TRAIN_POOL")
    if validation != tuple(int(item) for item in VALIDATION_POOL):
        raise RuntimeError("registry validation pool does not match VALIDATION_POOL")
    if reserved != tuple(int(item) for item in DEV_TEST_RESERVED):
        raise RuntimeError("registry reserved pool does not match the reserved split")
    excluded = {int(item) for item in pilot_scenarios()} | {int(item) for item in t30_scenarios()}
    chosen = tuple(int(item) for item in train if int(item) not in excluded)[:SCENARIO_COUNT]
    if len(chosen) != SCENARIO_COUNT:
        raise RuntimeError("registry does not contain eight train scenarios after the earlier pilots")
    for item in chosen:
        assert_allowed(int(item), split="train")
    if any(int(item) in excluded or int(item) in validation or int(item) in reserved for item in chosen):
        raise RuntimeError("expansion scenarios overlap an earlier pilot or a held-out split")
    if chosen != EXPECTED_SCENARIOS:
        raise RuntimeError(f"registry continuation is {chosen}, not the expected eight train scenarios")
    return chosen


def repeat_state() -> tuple[int, int]:
    """One pre-registered state is scored twice. The second pass adds no candidates."""
    return int(scenarios()[0]), int(REPEAT_TIME)


def sum_tolerance(tol_bits: float, num_buoys: int) -> float:
    """Per-buoy tolerance accumulated across a summed identity."""
    return float(tol_bits) * int(num_buoys)


def project_seconds(replay_seconds: float, score_seconds: float) -> float:
    """Upper bound: every state costs the measured score, plus one replay per scenario and one repeat."""
    return (
        float(replay_seconds) * int(SCENARIO_COUNT)
        + float(score_seconds) * int(MAX_STATES)
        + float(replay_seconds)
        + float(score_seconds)
    )


def scenario_effect(
    *,
    urgent_reversal: float,
    payload_reversal: float,
    payload_top1_within_one_bit: bool,
    urgent_top1_within_one_bit: bool,
) -> str:
    """Same cutoffs as D5. They are not retuned from the expansion outcome."""
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


def time_effect(effects: list[str]) -> str:
    if effects and all(item == "material" for item in effects):
        return "material_consistent"
    if effects and all(item == "small" for item in effects):
        return "small"
    if effects and all(item == "arms_match" for item in effects):
        return "arms_match"
    return "unstable"


def choose_gate(*, mechanical_ok: bool, budget_ok: bool, reproducible: bool, time_effects: dict) -> str:
    """A finished run is inconclusive when the arrival signal is not stable on arrival windows."""
    if not mechanical_ok or not budget_ok or not reproducible:
        return GATE_BLOCKED
    if len(time_effects) != len(DECISION_TIMES):
        return GATE_BLOCKED
    if any(str(value) == "arms_diverged" for value in time_effects.values()):
        return GATE_BLOCKED
    arrival = [str(value) for value in time_effects.values() if str(value) != "arms_match"]
    quiet = [str(value) for value in time_effects.values() if str(value) == "arms_match"]
    if not arrival:
        return GATE_INCONCLUSIVE
    if all(item == "material_consistent" for item in arrival) and all(item == "arms_match" for item in quiet):
        return GATE_COMPLETE
    return GATE_INCONCLUSIVE
