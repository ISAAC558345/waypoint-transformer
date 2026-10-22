"""Centralized critic schema. Training-time only; no optimizer here."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv

from .contracts import pairwise_distances
from .observation import build_actor_observation, communication_features
from .protocol import HOLD_STEPS


def build_critic_state(
    env: L3BEnv,
    option: TaskOption,
    last_movement: np.ndarray | None = None,
) -> dict[str, Any]:
    actor = build_actor_observation(env, option, last_movement)
    comm = communication_features(env)
    dist = pairwise_distances(env.uav_positions)
    m_count = int(env.config.num_uavs)
    assoc = np.asarray(option.association, dtype=np.float64)
    parts = [
        actor["self"].reshape(-1),
        actor["task"].reshape(-1),
        assoc,
        np.asarray(env.queue, dtype=np.float64),
        np.asarray(env.progress, dtype=np.float64),
        comm["sinr"],
        comm["rate"],
        dist.reshape(-1),
        comm["cross_link_interference"].reshape(-1),
        np.array(
            [
                float(option.remaining_hold) / float(HOLD_STEPS),
                float(env.config.episode_horizon - env.t) / float(env.config.episode_horizon),
            ],
            dtype=np.float64,
        ),
    ]
    vector = np.concatenate(parts, axis=0)
    return {
        "vector": vector,
        "dim": int(vector.size),
        "num_uavs": m_count,
        "fields": [
            "all_uav_self",
            "all_task_options",
            "association",
            "queue",
            "progress",
            "sinr",
            "rate",
            "pairwise_distances",
            "interference_matrix",
            "remaining_hold",
            "remaining_episode_time",
        ],
        "finite": bool(np.all(np.isfinite(vector))),
    }


def expected_critic_dim(num_uavs: int, num_buoys: int) -> int:
    from .observation import SELF_DIM, TASK_DIM

    m_count, k_count = int(num_uavs), int(num_buoys)
    return (
        m_count * SELF_DIM
        + m_count * TASK_DIM
        + m_count
        + k_count
        + m_count
        + m_count
        + m_count
        + m_count * m_count
        + m_count * m_count
        + 2
    )


__all__ = ["build_critic_state", "expected_critic_dim"]
