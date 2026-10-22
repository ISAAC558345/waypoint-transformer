"""Actor observations from currently observable L3-B fields only. No future arrivals."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv, active_links, interference_matrix, sinr_rates_for_links

from .contracts import env_target, pairwise_distances, refuse_rematch
from .protocol import ENV_IDLE, HOLD_STEPS

SELF_DIM = 5
TASK_DIM = 8
COMM_DIM = 4
NEIGHBOR_EDGE_DIM = 11
FORBIDDEN_KEYS = frozenset({"remaining_arrivals", "arrivals_future", "future_trajectory"})


def communication_features(env: L3BEnv) -> dict[str, np.ndarray]:
    """Read the existing L3-B SINR model. Does not invent a second formula."""
    links = active_links(env)
    sinr, rate = sinr_rates_for_links(env, links)
    interf = interference_matrix(env)
    noise = float(env.config.noise_power)
    total_interf = noise + np.sum(interf, axis=1)
    inr = (total_interf - noise) / max(noise, 1.0e-18)
    return {
        "sinr": sinr,
        "rate": rate,
        "total_interference": total_interf,
        "interference_to_noise": inr,
        "cross_link_interference": interf,
    }


def _task_vector(env: L3BEnv, option: TaskOption, uav: int) -> np.ndarray:
    k_count = int(env.config.num_buoys)
    tgt = env_target(int(option.association[uav]), k_count)
    valid = tgt >= 0
    vec = np.zeros(TASK_DIM, dtype=np.float64)
    if not valid:
        vec[0] = 0.0
        vec[6] = float(option.remaining_hold) / float(HOLD_STEPS)
        vec[7] = float(env.config.episode_horizon - env.t) / float(env.config.episode_horizon)
        return vec
    rel = env.buoy_positions[tgt] - env.uav_positions[uav]
    dist = float(np.linalg.norm(rel))
    vec[0] = 1.0
    vec[1:3] = rel
    vec[3] = dist
    vec[4] = float(env.queue[tgt])
    vec[5] = float(env.progress[uav])
    vec[6] = float(option.remaining_hold) / float(HOLD_STEPS)
    vec[7] = float(env.config.episode_horizon - env.t) / float(env.config.episode_horizon)
    return vec


def _self_vector(env: L3BEnv, last_movement: np.ndarray, uav: int) -> np.ndarray:
    vec = np.zeros(SELF_DIM, dtype=np.float64)
    vec[0:2] = env.uav_positions[uav]
    vec[2:4] = last_movement[uav]
    vec[4] = 1.0
    return vec


def _comm_vector(env: L3BEnv, option: TaskOption, comm: dict[str, np.ndarray], uav: int) -> np.ndarray:
    vec = np.zeros(COMM_DIM, dtype=np.float64)
    tgt = env_target(int(option.association[uav]), int(env.config.num_buoys))
    if tgt < 0:
        return vec
    vec[0] = float(comm["sinr"][uav])
    vec[1] = float(comm["rate"][uav])
    vec[2] = float(comm["total_interference"][uav])
    vec[3] = float(comm["interference_to_noise"][uav])
    return vec


def _neighbor_edge(env: L3BEnv, option: TaskOption, last_movement: np.ndarray, uav: int, other: int) -> np.ndarray:
    vec = np.zeros(NEIGHBOR_EDGE_DIM, dtype=np.float64)
    rel_pos = env.uav_positions[other] - env.uav_positions[uav]
    rel_move = last_movement[other] - last_movement[uav]
    dist = float(np.linalg.norm(rel_pos))
    other_tgt = env_target(int(option.association[other]), int(env.config.num_buoys))
    idle = other_tgt < 0
    active = (not idle) and bool(env.queue[other_tgt] > 1.0e-9) if other_tgt >= 0 else False
    vec[0:2] = rel_pos
    vec[2:4] = rel_move
    vec[4] = dist
    vec[5] = float(active)
    vec[6] = float(idle)
    if other_tgt >= 0:
        vec[7:9] = env.buoy_positions[other_tgt] - env.uav_positions[uav]
        vec[10] = float(env._in_range(other, other_tgt))
    vec[9] = float(env.progress[other])
    return vec


def neighbor_edges(env: L3BEnv, option: TaskOption, last_movement: np.ndarray, uav: int) -> np.ndarray:
    others = [j for j in range(int(env.config.num_uavs)) if j != int(uav)]
    return np.stack([_neighbor_edge(env, option, last_movement, uav, j) for j in others], axis=0)


def build_actor_observation(
    env: L3BEnv,
    option: TaskOption,
    last_movement: np.ndarray | None = None,
) -> dict[str, Any]:
    refuse_rematch(option)
    m_count = int(env.config.num_uavs)
    last = np.zeros((m_count, 2), dtype=np.float64) if last_movement is None else np.asarray(last_movement, dtype=np.float64)
    comm = communication_features(env)
    self_all = np.stack([_self_vector(env, last, m) for m in range(m_count)])
    task_all = np.stack([_task_vector(env, option, m) for m in range(m_count)])
    comm_all = np.stack([_comm_vector(env, option, comm, m) for m in range(m_count)])
    neigh = np.stack([neighbor_edges(env, option, last, m) for m in range(m_count)])
    nearest = []
    dist = pairwise_distances(env.uav_positions)
    for m in range(m_count):
        row = np.delete(dist[m], m)
        nearest.append(float(np.min(row)) if row.size else float("inf"))
    payload = {
        "self": self_all,
        "task": task_all,
        "communication": comm_all,
        "neighbor_set": neigh,
        "d_nearest": np.asarray(nearest, dtype=np.float64),
        "remaining_hold": int(option.remaining_hold),
        "remaining_episode_time": int(env.config.episode_horizon - env.t),
        "has_velocity_field": False,
        "motion_history_source": "last_movement",
        "leaks_future_arrivals": False,
    }
    for key in FORBIDDEN_KEYS:
        if key in payload:
            raise RuntimeError(f"future leakage key present: {key}")
    return payload


__all__ = [
    "COMM_DIM",
    "NEIGHBOR_EDGE_DIM",
    "SELF_DIM",
    "TASK_DIM",
    "build_actor_observation",
    "communication_features",
    "neighbor_edges",
]
