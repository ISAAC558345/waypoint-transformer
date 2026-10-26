"""Urgency observation built on the unchanged actor and critic builders."""
from __future__ import annotations

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.critic_state import build_critic_state, expected_critic_dim
from wt.lowlevel_v2.observation import TASK_DIM, build_actor_observation

from .protocol import CRITIC_EXTRA, CRITIC_IN_DIM, URGENCY_TASK_EXTRA


def target_urgency(view: dict, association, uav: int) -> tuple[float, int]:
    """Map one UAV to its current target buoy. Idle targets stay invalid."""
    buoy = int(np.asarray(association, dtype=np.int64).reshape(-1)[int(uav)])
    if buoy < 0:
        return 0.0, 0
    urgency = float(view["urgency"][buoy])
    valid = int(view["urgency_valid"][buoy])
    if valid == 0:
        urgency = 0.0
    return urgency, valid


def build_urgency_actor_observation(
    env: L3BEnv,
    option: TaskOption,
    view: dict,
    last_movement: np.ndarray | None = None,
) -> dict:
    """Append pre-service urgency and validity to the original task vector."""
    base = build_actor_observation(env, option, last_movement)
    task = np.asarray(base["task"], dtype=np.float64)
    if int(task.shape[-1]) != int(TASK_DIM):
        raise RuntimeError("formal task width changed")
    extra = np.zeros((int(task.shape[0]), URGENCY_TASK_EXTRA), dtype=np.float64)
    for uav in range(int(task.shape[0])):
        extra[uav, 0], extra[uav, 1] = target_urgency(view, option.association, uav)
    payload = dict(base)
    payload["task"] = np.concatenate([task, extra], axis=-1)
    payload["urgency_source"] = "episode_admission_age_v1_pre_service_view"
    return payload


def build_urgency_critic_state(
    env: L3BEnv,
    option: TaskOption,
    view: dict,
    last_movement: np.ndarray | None = None,
) -> dict:
    """Keep the formal 188-d critic and append one urgency/validity pair per UAV."""
    base = build_critic_state(env, option, last_movement)
    formal = np.asarray(base["vector"], dtype=np.float64)
    expected = int(expected_critic_dim(int(env.config.num_uavs), int(env.config.num_buoys)))
    if int(formal.size) != expected:
        raise RuntimeError("formal critic width changed")
    suffix = np.zeros(int(env.config.num_uavs) * 2, dtype=np.float64)
    for uav in range(int(env.config.num_uavs)):
        urgency, valid = target_urgency(view, option.association, uav)
        suffix[2 * uav] = urgency
        suffix[2 * uav + 1] = float(valid)
    if int(suffix.size) != int(CRITIC_EXTRA):
        raise RuntimeError("critic urgency suffix is not 12")
    vector = np.concatenate([formal, suffix], axis=0)
    if int(vector.size) != int(CRITIC_IN_DIM):
        raise RuntimeError("urgency critic width is not 200")
    payload = dict(base)
    payload["vector"] = vector
    payload["dim"] = int(vector.size)
    payload["fields"] = list(base["fields"]) + ["target_urgency_and_valid"]
    payload["finite"] = bool(np.all(np.isfinite(vector)))
    return payload
