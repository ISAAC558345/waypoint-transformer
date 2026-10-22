"""Frozen-lattice planner with only a causal predictable-channel transition."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import numpy as np

from wt.maritime.channel import compute_channel_metrics
from wt.maritime.env import MaritimeEnv
from wt.round15.planner import (
    ACTION_CANDIDATE_NAMES, EPSILON, _channel_kwargs, _expected_arrival,
    _policy_action, _project_position, _qre_action_from_arrays, action_candidates,
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
    oracle_future_channel: bool
    deployable_diagnostic: bool


def predictable_channel_multiplier(phases: np.ndarray, config: Any) -> np.ndarray:
    if config.channel_variation_mode == "static":
        return np.ones(config.num_buoys, dtype=float)
    return np.exp(float(config.channel_log_amplitude) * np.sin(np.asarray(phases, dtype=float)))


def advance_channel_phase(phases: np.ndarray, config: Any) -> np.ndarray:
    if config.channel_variation_mode == "static":
        return np.asarray(phases, dtype=float).copy()
    return np.mod(
        np.asarray(phases, dtype=float) + 2.0 * math.pi / int(config.channel_period_slots),
        2.0 * math.pi,
    )


def predictable_rates(position: np.ndarray, buoy_positions: np.ndarray, phases: np.ndarray, config: Any) -> np.ndarray:
    geometric = compute_channel_metrics(position, buoy_positions, **_channel_kwargs(config))
    gains = np.asarray(geometric.channel_gains, dtype=float) * predictable_channel_multiplier(phases, config)
    snr = float(config.buoy_tx_power_w) * gains / float(config.noise_power_w)
    return np.asarray(
        float(config.total_bandwidth_hz) / float(config.num_buoys) * np.log1p(snr) / math.log(2.0),
        dtype=float,
    )


def _causal_shadow_step(
    position: np.ndarray, queues: np.ndarray, buoy_positions: np.ndarray,
    phases: np.ndarray, action: np.ndarray, config: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    next_position, _, _ = _project_position(position, action, config)
    rates = predictable_rates(next_position, buoy_positions, phases, config)
    payload = np.minimum(np.asarray(queues, dtype=float), rates * float(config.delta_t))
    next_queues = np.clip(
        queues - payload + _expected_arrival(config), 0.0, float(config.queue_capacity_bits)
    )
    return next_position, next_queues, advance_channel_phase(phases, config), float(payload.sum())


def _causal_score(env: MaritimeEnv, info: Mapping[str, Any], first_action: np.ndarray, horizon: int) -> tuple[float, int]:
    """Forward-simulate only present oscillator state and registered dynamics."""
    position = np.asarray(info["uav_position_m"], dtype=float).copy()
    queues = np.asarray(info["raw_queue_bits"], dtype=float).copy()
    buoys = np.asarray(info["buoy_positions_m"], dtype=float).copy()
    phases = np.asarray(info.get("channel_phase_rad", np.zeros(env.num_buoys)), dtype=float).copy()
    total, nodes = 0.0, 0
    for step in range(int(horizon)):
        action = first_action if step == 0 else _qre_action_from_arrays(position, queues, buoys, env.config)
        position, queues, phases, payload = _causal_shadow_step(
            position, queues, buoys, phases, action, env.config
        )
        total += payload
        nodes += 1
    return float(total), nodes


def _oracle_score(env: MaritimeEnv, observation: np.ndarray, info: Mapping[str, Any], first_action: np.ndarray, horizon: int) -> tuple[float, int]:
    """Diagnostic branch only: copied environment supplies realized future state."""
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


def finite_lattice_plan(env: MaritimeEnv, observation: np.ndarray, info: Mapping[str, Any], *, horizon: int, mode: str) -> PlanDecision:
    if int(horizon) not in (1, 5, 10, 20):
        raise ValueError("Round-17 horizon must be one of {1,5,10,20}")
    if mode not in ("causal", "oracle"):
        raise ValueError("Round-17 mode must be 'causal' or 'oracle'")
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
        oracle_exogenous_knowledge=mode == "oracle", oracle_future_channel=mode == "oracle",
        deployable_diagnostic=mode == "causal",
    )


__all__ = ["ACTION_CANDIDATE_NAMES", "PlanDecision", "advance_channel_phase", "finite_lattice_plan", "predictable_channel_multiplier", "predictable_rates"]
