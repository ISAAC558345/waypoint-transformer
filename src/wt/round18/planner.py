"""Frozen causal finite search policies for the Round-18 audit."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import time
from typing import Any, Mapping

import numpy as np

from .multi_env import (
    EPSILON,
    MultiUAVMaritimeEnv,
    expected_arrival_bits,
    rate_matrix,
    shadow_step,
)


ACTION_CANDIDATE_NAMES = (
    "zero", "+x", "-x", "+y", "-y", "queue_rate_direction",
    "nearest_unsatisfied_buoy_direction",
)
PLANNING_HORIZONS = (1, 5, 10)
BEAM_WIDTH = 7


@dataclass(frozen=True)
class PlanDecision:
    action: np.ndarray
    horizon: int
    policy: str
    predicted_payload_bits: float
    nodes_expanded: int
    actions_evaluated: int
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


def per_uav_candidates(
    positions_m: np.ndarray,
    queue_bits: np.ndarray,
    buoy_positions_m: np.ndarray,
    config: Any,
) -> list[dict[str, np.ndarray]]:
    """Generate each UAV's frozen current-state candidate set."""
    positions = np.asarray(positions_m, dtype=np.float64)
    queues = np.asarray(queue_bits, dtype=np.float64)
    buoys = np.asarray(buoy_positions_m, dtype=np.float64)
    rates, ranges = rate_matrix(positions, buoys, config)
    literals = {
        "zero": np.array((0.0, 0.0), dtype=np.float64),
        "+x": np.array((1.0, 0.0), dtype=np.float64),
        "-x": np.array((-1.0, 0.0), dtype=np.float64),
        "+y": np.array((0.0, 1.0), dtype=np.float64),
        "-y": np.array((0.0, -1.0), dtype=np.float64),
    }
    result: list[dict[str, np.ndarray]] = []
    eligible = queues > EPSILON
    for uav in range(positions.shape[0]):
        choices = dict(literals)
        if np.any(eligible):
            score = np.where(eligible, (queues / float(config.queue_capacity_bits)) * rates[uav], -np.inf)
            queue_rate_target = int(np.argmax(score))
            nearest_target = int(np.flatnonzero(eligible)[np.argmin(ranges[uav, eligible])])
            choices["queue_rate_direction"] = _direction(positions[uav], buoys[queue_rate_target])
            choices["nearest_unsatisfied_buoy_direction"] = _direction(positions[uav], buoys[nearest_target])
        else:
            choices["queue_rate_direction"] = np.zeros(2, dtype=np.float64)
            choices["nearest_unsatisfied_buoy_direction"] = np.zeros(2, dtype=np.float64)
        if tuple(choices) != ACTION_CANDIDATE_NAMES:
            raise RuntimeError("Round-18 candidate action order changed after freeze")
        result.append(choices)
    return result


def independent_queue_rate_action(
    positions_m: np.ndarray,
    queue_bits: np.ndarray,
    buoy_positions_m: np.ndarray,
    config: Any,
) -> np.ndarray:
    """Independent baseline: each UAV sees only its own row of current rates."""
    candidates = per_uav_candidates(positions_m, queue_bits, buoy_positions_m, config)
    return np.vstack([item["queue_rate_direction"] for item in candidates])


def _joint_candidates(state: _State, buoy_positions_m: np.ndarray, config: Any):
    choices = per_uav_candidates(state.positions_m, state.queue_bits, buoy_positions_m, config)
    for names in itertools.product(ACTION_CANDIDATE_NAMES, repeat=len(choices)):
        yield names, np.vstack([choices[index][name] for index, name in enumerate(names)])


def _shadow_from_state(state: _State, action: np.ndarray, buoy_positions_m: np.ndarray, config: Any):
    return shadow_step(
        positions_m=state.positions_m,
        remaining_energy=state.remaining_energy,
        queue_bits=state.queue_bits,
        buoy_positions_m=buoy_positions_m,
        action=action,
        arrival_bits=expected_arrival_bits(config),
        config=config,
    )


def _next_state(step: Any) -> _State:
    return _State(step.positions_m.copy(), step.remaining_energy.copy(), step.queue_bits.copy())


def joint_plan(
    env: MultiUAVMaritimeEnv,
    info: Mapping[str, Any],
    *,
    horizon: int,
) -> PlanDecision:
    """Causal finite beam search with only current state and config expectation."""
    if env.num_uavs < 2:
        raise ValueError("Round-18 joint planner is defined for M >= 2")
    if int(horizon) not in PLANNING_HORIZONS:
        raise ValueError("Round-18 horizon must be one of {1, 5, 10}")
    started = time.perf_counter()
    initial = _State(
        np.asarray(info["uav_positions_m"], dtype=np.float64).copy(),
        np.asarray(info["remaining_energy_per_uav"], dtype=np.float64).copy(),
        np.asarray(info["raw_queue_bits"], dtype=np.float64).copy(),
    )
    buoy_positions = np.asarray(info["buoy_positions_m"], dtype=np.float64).copy()
    # (negative score, lexical action names, state, first action, first assignment)
    beam: list[tuple[float, tuple[str, ...], _State, np.ndarray, np.ndarray]] = [
        (0.0, tuple(), initial, np.zeros((env.num_uavs, 2)), np.full(env.num_uavs, -1, dtype=np.int64))
    ]
    nodes = actions = 0
    for depth in range(int(horizon)):
        expanded: list[tuple[float, tuple[str, ...], _State, np.ndarray, np.ndarray]] = []
        for total, prefix, state, root_action, root_assignment in beam:
            for names, action in _joint_candidates(state, buoy_positions, env.config):
                step = _shadow_from_state(state, action, buoy_positions, env.config)
                next_total = total + float(np.sum(step.payload_bits))
                expanded.append((next_total, prefix + tuple(names), _next_state(step), action if depth == 0 else root_action, step.assignment.copy() if depth == 0 else root_assignment))
                nodes += 1
                actions += 1
        expanded.sort(key=lambda row: (-row[0], row[1]))
        # Every decision evaluates the full frozen 49-action joint root.  For
        # H>1, retaining the pre-registered width-seven beam immediately after
        # that root keeps the continuation bounded without changing candidates.
        beam = expanded[: (49 if int(horizon) == 1 else BEAM_WIDTH)]
        if not beam:
            raise RuntimeError("joint finite search exhausted every branch")
    selected = beam[0]
    return PlanDecision(
        action=selected[3].copy(), horizon=int(horizon), policy=f"joint_h{horizon}",
        predicted_payload_bits=float(selected[0]), nodes_expanded=nodes,
        actions_evaluated=actions, runtime_seconds=float(time.perf_counter() - started),
        first_assignment=selected[4].copy(), first_candidate_names=selected[1][: env.num_uavs],
    )


def independent_decision(
    env: MultiUAVMaritimeEnv, info: Mapping[str, Any]
) -> PlanDecision:
    """Return a current-state-only action without another UAV's intention."""
    started = time.perf_counter()
    positions = np.asarray(info["uav_positions_m"], dtype=np.float64)
    queues = np.asarray(info["raw_queue_bits"], dtype=np.float64)
    buoys = np.asarray(info["buoy_positions_m"], dtype=np.float64)
    action = independent_queue_rate_action(positions, queues, buoys, env.config)
    state = _State(positions.copy(), np.asarray(info["remaining_energy_per_uav"], dtype=np.float64).copy(), queues.copy())
    step = _shadow_from_state(state, action, buoys, env.config)
    return PlanDecision(
        action=action, horizon=1, policy="independent_queue_rate",
        predicted_payload_bits=float(np.sum(step.payload_bits)), nodes_expanded=env.num_uavs,
        actions_evaluated=env.num_uavs, runtime_seconds=float(time.perf_counter() - started),
        first_assignment=step.assignment.copy(),
        first_candidate_names=tuple("queue_rate_direction" for _ in range(env.num_uavs)),
    )


__all__ = [
    "ACTION_CANDIDATE_NAMES", "BEAM_WIDTH", "PLANNING_HORIZONS", "PlanDecision",
    "independent_decision", "independent_queue_rate_action", "joint_plan",
    "per_uav_candidates",
]
