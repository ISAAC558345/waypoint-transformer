"""Frozen causal Round-18 movement search under Round-19 link semantics."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import time
from typing import Any, Mapping

import numpy as np

from wt.round18.multi_env import EPSILON, expected_arrival_bits, rate_matrix, shadow_step

from .multi_env import SharedSpectrumMultiUAVEnv, SharedSpectrumOptions, shared_shadow_step


ACTION_CANDIDATE_NAMES = (
    "zero", "+x", "-x", "+y", "-y", "queue_rate_direction",
    "nearest_unsatisfied_buoy_direction",
)
# Round-33 has explicit, narrow authorization to expose the existing finite
# beam-search implementation at H=20.  No candidate, score, transition,
# matching, beam-width, or tie-break rule changes here; the loop below already
# executes exactly ``horizon`` shadow transitions.
PLANNING_HORIZONS = (1, 5, 10, 20)
BEAM_WIDTH = 7


@dataclass(frozen=True)
class PlanDecision:
    action: np.ndarray
    horizon: int
    policy: str
    predicted_payload_bits: float
    nodes_expanded: int
    actions_evaluated: int
    matching_assignments_evaluated: int
    runtime_seconds: float
    first_assignment: np.ndarray
    first_candidate_names: tuple[str, ...]
    future_trace_accessed: bool = False


@dataclass(frozen=True)
class _State:
    positions_m: np.ndarray
    remaining_energy: np.ndarray
    queue_bits: np.ndarray


def _direction(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    norm = float(np.linalg.norm(delta))
    return np.zeros(2, dtype=np.float64) if norm <= EPSILON else delta / norm


def per_uav_candidates(positions_m: np.ndarray, queue_bits: np.ndarray, buoy_positions_m: np.ndarray, config: Any) -> list[dict[str, np.ndarray]]:
    """The unchanged seven action candidates based only on current own rates."""
    positions, queues, buoys = np.asarray(positions_m, dtype=np.float64), np.asarray(queue_bits, dtype=np.float64), np.asarray(buoy_positions_m, dtype=np.float64)
    rates, ranges = rate_matrix(positions, buoys, config)
    literals = {"zero": np.array((0.0, 0.0)), "+x": np.array((1.0, 0.0)), "-x": np.array((-1.0, 0.0)), "+y": np.array((0.0, 1.0)), "-y": np.array((0.0, -1.0))}
    eligible = queues > EPSILON
    result = []
    for uav in range(positions.shape[0]):
        choices = dict(literals)
        if np.any(eligible):
            score = np.where(eligible, queues / float(config.queue_capacity_bits) * rates[uav], -np.inf)
            queue_rate_target = int(np.argmax(score)); nearest = int(np.flatnonzero(eligible)[np.argmin(ranges[uav, eligible])])
            choices["queue_rate_direction"] = _direction(positions[uav], buoys[queue_rate_target])
            choices["nearest_unsatisfied_buoy_direction"] = _direction(positions[uav], buoys[nearest])
        else:
            choices["queue_rate_direction"] = np.zeros(2); choices["nearest_unsatisfied_buoy_direction"] = np.zeros(2)
        if tuple(choices) != ACTION_CANDIDATE_NAMES:
            raise RuntimeError("frozen movement candidate order changed")
        result.append(choices)
    return result


def independent_queue_rate_action(positions_m: np.ndarray, queue_bits: np.ndarray, buoy_positions_m: np.ndarray, config: Any) -> np.ndarray:
    return np.vstack([row["queue_rate_direction"] for row in per_uav_candidates(positions_m, queue_bits, buoy_positions_m, config)])


def _joint_candidates(state: _State, buoys: np.ndarray, config: Any):
    choices = per_uav_candidates(state.positions_m, state.queue_bits, buoys, config)
    for names in itertools.product(ACTION_CANDIDATE_NAMES, repeat=len(choices)):
        yield names, np.vstack([choices[m][name] for m, name in enumerate(names)])


def _shadow(state: _State, action: np.ndarray, buoys: np.ndarray, env: SharedSpectrumMultiUAVEnv, matching_mode: str) -> Any:
    if env.options.spectrum_mode == "orthogonal":
        return shadow_step(positions_m=state.positions_m, remaining_energy=state.remaining_energy, queue_bits=state.queue_bits, buoy_positions_m=buoys, action=action, arrival_bits=expected_arrival_bits(env.config), config=env.config)
    options = SharedSpectrumOptions("shared", env.options.interference_overlap_factor, matching_mode, env.options.conflict_isr_threshold)
    return shared_shadow_step(positions_m=state.positions_m, remaining_energy=state.remaining_energy, queue_bits=state.queue_bits, buoy_positions_m=buoys, action=action, arrival_bits=expected_arrival_bits(env.config), config=env.config, options=options)


def _next(step: Any) -> _State:
    return _State(step.positions_m.copy(), step.remaining_energy.copy(), step.queue_bits.copy())


def independent_decision(env: SharedSpectrumMultiUAVEnv, info: Mapping[str, Any]) -> PlanDecision:
    """No other-UAV future intention is an input to the independent rule."""
    started = time.perf_counter()
    positions, queues, buoys = np.asarray(info["uav_positions_m"]), np.asarray(info["raw_queue_bits"]), np.asarray(info["buoy_positions_m"])
    action = independent_queue_rate_action(positions, queues, buoys, env.config)
    step = _shadow(_State(positions.copy(), np.asarray(info["remaining_energy_per_uav"]).copy(), queues.copy()), action, buoys, env, "legacy")
    return PlanDecision(action, 1, "independent_queue_rate", float(np.sum(step.payload_bits)), env.num_uavs, env.num_uavs, int(getattr(step, "matching_assignments_evaluated", 0)), float(time.perf_counter() - started), step.assignment.copy(), tuple("queue_rate_direction" for _ in range(env.num_uavs)))


def joint_plan(env: SharedSpectrumMultiUAVEnv, info: Mapping[str, Any], *, horizon: int) -> PlanDecision:
    """Current-observable finite beam search with joint-aware matching."""
    if env.num_uavs < 2 or int(horizon) not in PLANNING_HORIZONS:
        raise ValueError("Round-19 joint horizon must be one of {1,5,10,20} for M>=2")
    started = time.perf_counter()
    initial = _State(np.asarray(info["uav_positions_m"], dtype=np.float64).copy(), np.asarray(info["remaining_energy_per_uav"], dtype=np.float64).copy(), np.asarray(info["raw_queue_bits"], dtype=np.float64).copy())
    buoys = np.asarray(info["buoy_positions_m"], dtype=np.float64).copy()
    beam: list[tuple[float, tuple[str, ...], _State, np.ndarray, np.ndarray, int]] = [(0.0, tuple(), initial, np.zeros((env.num_uavs, 2)), np.full(env.num_uavs, -1, dtype=np.int64), 0)]
    nodes = actions = matching_evaluated = 0
    for depth in range(int(horizon)):
        expanded = []
        for total, prefix, state, root_action, root_assignment, root_match_count in beam:
            for names, action in _joint_candidates(state, buoys, env.config):
                try:
                    step = _shadow(state, action, buoys, env, "joint_aware")
                except RuntimeError as error:
                    # The frozen Round-18 hard projection is the feasibility
                    # oracle.  A joint candidate for which it cannot converge
                    # is not a legal successor; do not weaken or modify the
                    # projection merely to score it.
                    if "hard UAV separation projection did not converge" not in str(error):
                        raise
                    nodes += 1; actions += 1
                    continue
                count = int(getattr(step, "matching_assignments_evaluated", 0))
                expanded.append((total + float(np.sum(step.payload_bits)), prefix + tuple(names), _next(step), action if depth == 0 else root_action, step.assignment.copy() if depth == 0 else root_assignment, count if depth == 0 else root_match_count))
                nodes += 1; actions += 1; matching_evaluated += count
        expanded.sort(key=lambda row: (-row[0], row[1]))
        beam = expanded[: (49 if int(horizon) == 1 else BEAM_WIDTH)]
        if not beam:
            raise RuntimeError("joint search exhausted every branch")
    selected = beam[0]
    return PlanDecision(selected[3].copy(), int(horizon), f"joint_h{horizon}", float(selected[0]), nodes, actions, matching_evaluated, float(time.perf_counter() - started), selected[4].copy(), selected[1][:env.num_uavs])


__all__ = ["ACTION_CANDIDATE_NAMES", "BEAM_WIDTH", "PLANNING_HORIZONS", "PlanDecision", "independent_decision", "independent_queue_rate_action", "joint_plan", "per_uav_candidates"]
