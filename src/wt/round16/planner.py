"""Round-15-equivalent finite lattice with the sole drift transition update."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np

from wt.maritime.env import MaritimeEnv, reflect_buoy_motion
from wt.round15.planner import (
    ACTION_CANDIDATE_NAMES, EPSILON, _channel_kwargs, _expected_arrival,
    _policy_action, _project_position, _qre_action_from_arrays, action_candidates,
    finite_lattice_plan as _round15_plan,
)
from wt.round7.round7_runner import radial_projection


@dataclass(frozen=True)
class PlanDecision:
    action: np.ndarray
    horizon: int
    mode: str
    score_payload_bits: float
    nodes_expanded: int
    candidate_scores: Mapping[str, float]
    runtime_seconds: float
    oracle_exogenous_knowledge: bool
    oracle_future_drift: bool
    deployable_diagnostic: bool


def _causal_buoy_transition(
    positions: np.ndarray, velocities: np.ndarray, config: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Causal constant-velocity prediction; slow-mode innovations are unknown."""
    return reflect_buoy_motion(positions, velocities, config.delta_t, config)


def _causal_shadow_step(
    position: np.ndarray,
    queues: np.ndarray,
    buoy_positions: np.ndarray,
    buoy_velocities: np.ndarray,
    action: np.ndarray,
    config: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    from wt.maritime.channel import compute_channel_metrics

    next_position, _, _ = _project_position(position, action, config)
    rates = np.asarray(
        compute_channel_metrics(next_position, buoy_positions, **_channel_kwargs(config)).rates_bps,
        dtype=float,
    )
    payload = np.minimum(np.asarray(queues, dtype=float), rates * float(config.delta_t))
    next_queues = np.clip(
        queues - payload + _expected_arrival(config), 0.0, float(config.queue_capacity_bits)
    )
    next_buoys, next_velocities = _causal_buoy_transition(buoy_positions, buoy_velocities, config)
    return next_position, next_queues, next_buoys, next_velocities, float(payload.sum())


def _causal_score(env: MaritimeEnv, info: Mapping[str, Any], first_action: np.ndarray, horizon: int) -> tuple[float, int]:
    position = np.asarray(info["uav_position_m"], dtype=float).copy()
    queues = np.asarray(info["raw_queue_bits"], dtype=float).copy()
    buoys = np.asarray(info["buoy_positions_m"], dtype=float).copy()
    velocities = np.asarray(info["buoy_drift_velocity_mps"], dtype=float).copy()
    total, nodes = 0.0, 0
    for step in range(int(horizon)):
        action = first_action if step == 0 else _qre_action_from_arrays(position, queues, buoys, env.config)
        position, queues, buoys, velocities, payload = _causal_shadow_step(
            position, queues, buoys, velocities, action, env.config
        )
        total += payload
        nodes += 1
    return float(total), nodes


def _oracle_score(env: MaritimeEnv, observation: np.ndarray, info: Mapping[str, Any], first_action: np.ndarray, horizon: int) -> tuple[float, int]:
    branch = copy.deepcopy(env)
    obs, current = np.asarray(observation, dtype=float).copy(), dict(info)
    total, nodes = 0.0, 0
    for step in range(int(horizon)):
        action = first_action if step == 0 else _policy_action("queue_rate_energy", obs, current)
        obs, _, terminated, truncated, current = branch.step(radial_projection(action))
        total += float(np.asarray(current["transmitted_payload_bits"], dtype=float).sum())
        nodes += 1
        if terminated or truncated:
            break
    return float(total), nodes


def finite_lattice_plan(
    env: MaritimeEnv, observation: np.ndarray, info: Mapping[str, Any], *, horizon: int, mode: str
) -> PlanDecision:
    """Evaluate the frozen Round-15 lattice, advancing only buoy transition."""
    if int(horizon) not in (1, 5, 10, 20):
        raise ValueError("Round-16 horizon must be one of {1,5,10,20}")
    if mode not in ("causal", "oracle"):
        raise ValueError("Round-16 mode must be 'causal' or 'oracle'")
    if env.config.buoy_drift_mode == "static":
        original = _round15_plan(env, observation, info, horizon=int(horizon), mode=mode)
        return PlanDecision(
            action=original.action, horizon=original.horizon, mode=original.mode,
            score_payload_bits=original.score_payload_bits, nodes_expanded=original.nodes_expanded,
            candidate_scores=original.candidate_scores, runtime_seconds=original.runtime_seconds,
            oracle_exogenous_knowledge=original.oracle_exogenous_knowledge,
            oracle_future_drift=False, deployable_diagnostic=original.deployable_diagnostic,
        )
    started = time.perf_counter()
    candidates = action_candidates(env, observation, info)
    scores: dict[str, float] = {}
    nodes = 0
    for name, action in candidates.items():
        score, expanded = (
            _causal_score(env, info, action, int(horizon))
            if mode == "causal" else _oracle_score(env, observation, info, action, int(horizon))
        )
        scores[name] = score
        nodes += expanded
    order = {name: index for index, name in enumerate(ACTION_CANDIDATE_NAMES)}
    selected = max(scores, key=lambda name: (scores[name], -float(np.linalg.norm(candidates[name])), -order[name]))
    return PlanDecision(
        action=radial_projection(candidates[selected]), horizon=int(horizon), mode=mode,
        score_payload_bits=float(scores[selected]), nodes_expanded=int(nodes),
        candidate_scores=scores, runtime_seconds=float(time.perf_counter() - started),
        oracle_exogenous_knowledge=mode == "oracle", oracle_future_drift=mode == "oracle",
        deployable_diagnostic=mode == "causal",
    )


__all__ = ["ACTION_CANDIDATE_NAMES", "PlanDecision", "finite_lattice_plan"]
