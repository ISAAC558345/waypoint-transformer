"""Deterministic finite-lattice planning for the Round-15 audit.

There are intentionally no learned models or continuous optimizers here.  A
causal rollout uses only present state and a declared arrival persistence
assumption.  The oracle branch is separately labelled and may traverse the
already materialized branch trace solely as an upper-bound diagnostic.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from wt.maritime.baselines import create_baseline_policy
from wt.maritime.channel import compute_channel_metrics
from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.env import MaritimeEnv
from wt.round7.round7_runner import radial_projection


EPSILON = 1.0e-12
CAUSAL_PERSISTENCE_ASSUMPTION = (
    "per-buoy expected constant arrival from the current environment configuration; "
    "no realized future arrival trace is read"
)
ACTION_CANDIDATE_NAMES = (
    "zero", "+x", "-x", "+y", "-y", "diag_ne", "diag_nw", "diag_se", "diag_sw",
    "nearest_buoy", "max_queue", "queue_rate_energy", "myopic_one_step_greedy",
)


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
    deployable_diagnostic: bool


def _channel_kwargs(config: MaritimeEnvConfig) -> dict[str, float]:
    return {
        "uav_altitude_m": float(config.uav_altitude),
        "total_bandwidth_hz": float(config.total_bandwidth_hz),
        "buoy_tx_power_w": float(config.buoy_tx_power_w),
        "reference_channel_gain": float(config.reference_channel_gain),
        "pathloss_exponent": float(config.pathloss_exponent),
        "noise_power_w": float(config.noise_power_w),
    }


def _literal_actions() -> dict[str, np.ndarray]:
    diagonal = 1.0 / math.sqrt(2.0)
    return {
        "zero": np.array((0.0, 0.0), dtype=float),
        "+x": np.array((1.0, 0.0), dtype=float),
        "-x": np.array((-1.0, 0.0), dtype=float),
        "+y": np.array((0.0, 1.0), dtype=float),
        "-y": np.array((0.0, -1.0), dtype=float),
        "diag_ne": np.array((diagonal, diagonal), dtype=float),
        "diag_nw": np.array((-diagonal, diagonal), dtype=float),
        "diag_se": np.array((diagonal, -diagonal), dtype=float),
        "diag_sw": np.array((-diagonal, -diagonal), dtype=float),
    }


def _policy_action(policy_name: str, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
    policy = create_baseline_policy(policy_name, seed=0)
    policy.reset(seed=0)
    return radial_projection(policy.act(observation, info))


def _project_position(position: np.ndarray, action: np.ndarray, config: MaritimeEnvConfig) -> tuple[np.ndarray, float, float]:
    action = radial_projection(np.asarray(action, dtype=float))
    maximum_step = float(config.max_speed * config.delta_t)
    raw_distance = float(np.linalg.norm(action) * maximum_step)
    candidate = np.asarray(position, dtype=float) + action * maximum_step
    upper = np.array((config.area_width, config.area_height), dtype=float)
    applied = np.clip(candidate, 0.0, upper)
    return applied, raw_distance, float(np.linalg.norm(applied - position))


def _expected_arrival(config: MaritimeEnvConfig) -> np.ndarray:
    if config.arrival_mode == "constant":
        return np.full(config.num_buoys, float(config.constant_arrival_bits_per_step), dtype=float)
    if config.arrival_mode == "poisson":
        expected = float(config.poisson_arrival_packets_per_second * config.delta_t * config.arrival_packet_size_bits)
        return np.full(config.num_buoys, expected, dtype=float)
    raise ValueError("causal Round-15 planner does not support trace-mode arrivals")


def _qre_action_from_arrays(position: np.ndarray, queues: np.ndarray, buoy_positions: np.ndarray, config: MaritimeEnvConfig) -> np.ndarray:
    metrics = compute_channel_metrics(position, buoy_positions, **_channel_kwargs(config))
    rates = np.asarray(metrics.rates_bps, dtype=float)
    eligible = queues > EPSILON
    if not bool(eligible.any()):
        return np.zeros(2, dtype=float)
    horizontal = np.asarray(metrics.horizontal_distances_m, dtype=float)
    maximum_step = float(config.max_speed * config.delta_t)
    maximum_cost = float(config.hover_energy_per_step + config.move_energy_per_meter * maximum_step)
    movement = np.minimum(horizontal, maximum_step)
    energy = (config.hover_energy_per_step + config.move_energy_per_meter * movement) / maximum_cost if maximum_cost > EPSILON else np.zeros_like(horizontal)
    queue_factor = np.where(eligible, queues / max(float(queues[eligible].sum()), EPSILON), 0.0)
    rate_factor = np.where(eligible, rates / max(float(rates[eligible].sum()), EPSILON), 0.0)
    score = np.where(eligible, queue_factor * rate_factor / np.maximum(energy, EPSILON), -np.inf)
    selected = int(np.argmax(score))
    delta = np.asarray(buoy_positions[selected] - position, dtype=float)
    norm = float(np.linalg.norm(delta))
    return np.zeros(2, dtype=float) if norm <= EPSILON else radial_projection(delta / norm)


def _causal_shadow_step(position: np.ndarray, queues: np.ndarray, action: np.ndarray, buoy_positions: np.ndarray, config: MaritimeEnvConfig) -> tuple[np.ndarray, np.ndarray, float, float]:
    next_position, raw_distance, _ = _project_position(position, action, config)
    rates = np.asarray(compute_channel_metrics(next_position, buoy_positions, **_channel_kwargs(config)).rates_bps, dtype=float)
    payload = np.minimum(np.asarray(queues, dtype=float), rates * float(config.delta_t))
    arrival = _expected_arrival(config)
    next_queues = np.clip(queues - payload + arrival, 0.0, float(config.queue_capacity_bits))
    energy = float(config.hover_energy_per_step + config.move_energy_per_meter * raw_distance)
    return next_position, next_queues, float(payload.sum()), energy


def _base_candidates(env: MaritimeEnv, observation: np.ndarray, info: Mapping[str, Any]) -> dict[str, np.ndarray]:
    candidates = _literal_actions()
    for policy_name in ("nearest_buoy", "max_queue", "queue_rate_energy"):
        candidates[policy_name] = _policy_action(policy_name, observation, info)
    return {name: radial_projection(action) for name, action in candidates.items()}


def myopic_one_step_greedy(env: MaritimeEnv, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
    """Causal lexicographic one-step payload rule with no future input."""
    candidates = _base_candidates(env, observation, info)
    position = np.asarray(info["uav_position_m"], dtype=float)
    queues = np.asarray(info["raw_queue_bits"], dtype=float)
    buoy_positions = np.asarray(info["buoy_positions_m"], dtype=float)
    scored: list[tuple[float, float, int, np.ndarray]] = []
    for index, name in enumerate(candidates):
        _, _, payload, _ = _causal_shadow_step(position, queues, candidates[name], buoy_positions, env.config)
        scored.append((payload, -float(np.linalg.norm(candidates[name])), -index, candidates[name]))
    return radial_projection(max(scored, key=lambda row: row[:3])[3])


def action_candidates(env: MaritimeEnv, observation: np.ndarray, info: Mapping[str, Any]) -> dict[str, np.ndarray]:
    candidates = _base_candidates(env, observation, info)
    candidates["myopic_one_step_greedy"] = myopic_one_step_greedy(env, observation, info)
    if tuple(candidates) != ACTION_CANDIDATE_NAMES:
        raise RuntimeError("Round-15 candidate action set changed after protocol freeze")
    if not all(env.action_space.contains(action) for action in candidates.values()):
        raise RuntimeError("Round-15 candidate action is not legal after projection")
    return candidates


def _causal_score(env: MaritimeEnv, info: Mapping[str, Any], first_action: np.ndarray, horizon: int) -> tuple[float, int]:
    position = np.asarray(info["uav_position_m"], dtype=float).copy()
    queues = np.asarray(info["raw_queue_bits"], dtype=float).copy()
    buoy_positions = np.asarray(info["buoy_positions_m"], dtype=float).copy()
    total, nodes = 0.0, 0
    for step in range(int(horizon)):
        action = first_action if step == 0 else _qre_action_from_arrays(position, queues, buoy_positions, env.config)
        position, queues, payload, _ = _causal_shadow_step(position, queues, action, buoy_positions, env.config)
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


def finite_lattice_plan(env: MaritimeEnv, observation: np.ndarray, info: Mapping[str, Any], *, horizon: int, mode: str) -> PlanDecision:
    """Evaluate every frozen first action with the fixed QRE continuation."""
    if int(horizon) not in (1, 5, 10, 20):
        raise ValueError("Round-15 horizon must be one of {1,5,10,20}")
    if mode not in ("causal", "oracle"):
        raise ValueError("Round-15 planning mode must be 'causal' or 'oracle'")
    started = time.perf_counter()
    candidates = action_candidates(env, observation, info)
    scores: dict[str, float] = {}
    node_count = 0
    for name, action in candidates.items():
        score, nodes = (
            _causal_score(env, info, action, int(horizon))
            if mode == "causal"
            else _oracle_score(env, observation, info, action, int(horizon))
        )
        scores[name] = score
        node_count += nodes
    order = {name: index for index, name in enumerate(ACTION_CANDIDATE_NAMES)}
    selected = max(scores, key=lambda name: (scores[name], -float(np.linalg.norm(candidates[name])), -order[name]))
    return PlanDecision(
        action=radial_projection(candidates[selected]),
        horizon=int(horizon),
        mode=mode,
        score_payload_bits=float(scores[selected]),
        nodes_expanded=int(node_count),
        candidate_scores=scores,
        runtime_seconds=float(time.perf_counter() - started),
        oracle_exogenous_knowledge=mode == "oracle",
        deployable_diagnostic=mode == "causal",
    )


__all__ = [
    "ACTION_CANDIDATE_NAMES",
    "CAUSAL_PERSISTENCE_ASSUMPTION",
    "PlanDecision",
    "action_candidates",
    "finite_lattice_plan",
    "myopic_one_step_greedy",
]
