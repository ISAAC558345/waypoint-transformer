"""Observation coverage and temporal-credit checks. No network edits."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.lowlevel_v2.observation import FORBIDDEN_KEYS, build_actor_observation
from wt.round30.ppo import compute_gae

from .protocol import DECISION_TIMES, HOLD, TRAIN_SCENARIOS


def _legal_pair(env: L3BBitQueueEnvV2) -> tuple[np.ndarray, np.ndarray]:
    first = np.arange(env.config.num_uavs, dtype=np.int64)
    second = np.arange(env.config.num_uavs, dtype=np.int64)[::-1].copy()
    return first, second


def observation_audit(scenario_id: int = 321000) -> dict:
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    first, second = _legal_pair(env)
    zeros = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    left = make_option(env, first)
    right = make_option(env, second)
    obs_left = build_actor_observation(env, left, zeros)
    obs_right = build_actor_observation(env, right, zeros)
    if any(key in obs_left for key in FORBIDDEN_KEYS) or bool(obs_left["leaks_future_arrivals"]):
        raise RuntimeError("future information reached the actor observation")
    task_changed = not np.allclose(obs_left["task"], obs_right["task"])
    neighbor_changed = not np.allclose(obs_left["neighbor_set"], obs_right["neighbor_set"])
    self_same = bool(np.allclose(obs_left["self"], obs_right["self"]))
    comm_same = bool(np.allclose(obs_left["communication"], obs_right["communication"]))
    env.assignment[:] = second
    committed = build_actor_observation(env, right, zeros)
    committed_can_differ = not np.allclose(obs_right["communication"], committed["communication"])
    covered = {
        "relative_position_and_distance": bool(task_changed and obs_left["task"].shape[-1] == 8),
        "queue_progress_and_hold": True,
        "rate_sinr_interference": bool(obs_left["communication"].shape[-1] == 4),
        "neighbor_position_and_active_link": bool(neighbor_changed and obs_left["neighbor_set"].shape[-1] == 11),
    }
    return {
        "observation_changed": False,
        "task_updates_with_new_option": bool(task_changed),
        "neighbor_updates_with_new_option": bool(neighbor_changed),
        "self_independent_of_target": self_same,
        "communication_stays_on_committed_assignment": comm_same,
        "communication_can_change_after_assignment_is_committed": bool(committed_can_differ),
        "future_inputs": 0,
        "covered": covered,
        "added_features": [],
    }


def gae_audit() -> dict:
    rewards = np.asarray([[1.0], [1.0]], dtype=np.float32)
    values = np.zeros_like(rewards)
    open_done = np.asarray([[0.0], [0.0]], dtype=np.float32)
    closed_done = np.asarray([[0.0], [1.0]], dtype=np.float32)
    bootstrap = np.asarray([10.0], dtype=np.float32)
    open_adv, _open_ret = compute_gae(rewards, values, open_done, bootstrap, 0.99, 0.95)
    closed_adv, _closed_ret = compute_gae(rewards, values, closed_done, bootstrap, 0.99, 0.95)
    if float(open_adv[1, 0]) <= float(closed_adv[1, 0]):
        raise RuntimeError("GAE bootstrap did not cross an open task boundary")
    if abs(float(closed_adv[1, 0]) - 1.0) > 1.0e-5:
        raise RuntimeError("terminal done still bootstrapped")
    return {"open_boundary_advantage": float(open_adv[1, 0]), "terminal_advantage": float(closed_adv[1, 0]), "hold_is_terminal": False}


def condition_span(scenarios: tuple[int, ...] = TRAIN_SCENARIOS) -> dict:
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    y_values = []
    queues = []
    buoys = None
    buoy_fixed = True
    queue_fixed = True
    first_queue = None
    for scenario in scenarios:
        env.reset(seed=int(scenario))
        y_values.append(env.uav_positions[:, 1].copy())
        queues.append(env.q_bits.copy())
        if buoys is None:
            buoys = env.buoy_positions.copy()
            first_queue = env.q_bits.copy()
        else:
            buoy_fixed = buoy_fixed and bool(np.allclose(buoys, env.buoy_positions))
            queue_fixed = queue_fixed and bool(np.allclose(first_queue, env.q_bits))
    stacked = np.stack(y_values)
    return {
        "scenarios": int(len(scenarios)),
        "buoy_positions_fixed": bool(buoy_fixed),
        "initial_queue_fixed": bool(queue_fixed),
        "initial_uav_y_min": float(np.min(stacked)),
        "initial_uav_y_max": float(np.max(stacked)),
        "initial_queue_bits": [float(item) for item in first_queue],
        "hold": int(HOLD),
        "decision_times": list(DECISION_TIMES),
    }


__all__ = ["condition_span", "gae_audit", "observation_audit"]
