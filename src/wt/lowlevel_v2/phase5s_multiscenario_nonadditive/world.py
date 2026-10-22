"""Snapshots for training and, after checkpoints exist, fresh confirmation."""
from __future__ import annotations

import numpy as np

from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.phase5_hgnn_urgency.features import graph_batch, observable_exclusive_assignment
from wt.lowlevel_v2.phase5_hgnn_urgency.urgency import shared_task_urgency
from wt.lowlevel_v2.phase5f_utility_rank.protocol import FORBIDDEN_INPUTS

from .protocol import assert_confirm_seed, assert_train_seed


def _advance(seed: int, moment: int):
    env = L3BEnv()
    env.reset(seed=int(seed))
    last_positions = None
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    while int(env.t) < int(moment):
        positions = np.asarray(env.uav_positions, dtype=np.float64).copy()
        assignment = observable_exclusive_assignment(env)
        movement = fly_toward(env, assignment)
        info = env.step(movement, assignment)
        last_positions = positions
        last_arrival = np.asarray(info["arrivals_blocks"], dtype=np.float32)
        last_movement = np.asarray(movement, dtype=np.float64)
    if int(env.t) != int(moment):
        raise RuntimeError(f"phase-5s stopped at t={env.t}, expected {moment}")
    graph_last = np.asarray(env.uav_positions, dtype=np.float32) if last_positions is None else np.asarray(last_positions, dtype=np.float32)
    info = env.info()
    batch = graph_batch(info, env, last_arrival, graph_last, urgency=shared_task_urgency(env.queue))
    leaked = set(batch).intersection(FORBIDDEN_INPUTS)
    if leaked or "j_mappo" in batch:
        raise RuntimeError(f"snapshot contains a label field: {sorted(leaked)}")
    cloned = {key: np.asarray(value).copy() for key, value in batch.items()}
    diagnosis = {
        "queue": np.asarray(env.queue, dtype=np.float64).copy(),
        "urgency": np.asarray(shared_task_urgency(env.queue), dtype=np.float64).copy(),
        "distance": np.asarray(info["distance_uav_buoy"], dtype=np.float64).copy(),
        "rate": np.asarray(info["rate_if_assigned"], dtype=np.float64).copy(),
        "eta": np.asarray(info["eta_slots"], dtype=np.float64).copy(),
        "interference": np.asarray(info["interference_uav_uav"], dtype=np.float64).copy(),
    }
    return env, np.asarray(last_movement, dtype=np.float64), cloned, diagnosis


def replay_train(seed: int, moment: int):
    assert_train_seed(int(seed))
    return _advance(int(seed), int(moment))


def replay_confirmation(seed: int, moment: int):
    assert_confirm_seed(int(seed))
    return _advance(int(seed), int(moment))


__all__ = ["replay_confirmation", "replay_train"]
