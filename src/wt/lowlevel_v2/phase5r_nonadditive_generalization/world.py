"""Snapshots and candidate pools for the development-generalization scenarios."""
from __future__ import annotations

import numpy as np

from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.phase5_hgnn_urgency.features import graph_batch, observable_exclusive_assignment
from wt.lowlevel_v2.phase5_hgnn_urgency.urgency import shared_task_urgency
from wt.lowlevel_v2.phase5f_utility_rank.protocol import FORBIDDEN_INPUTS
from wt.lowlevel_v2.phase5m_joint_planner.planner import candidate_lists
from wt.lowlevel_v2.phase5n_joint_planner.matchings import maximal_assignments

from .protocol import assert_phase5r_seed


def replay_snapshot(seed: int, moment: int):
    """Advance with the observable fly assignment. This is the state, not a model decision."""
    assert_phase5r_seed(int(seed))
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
        raise RuntimeError(f"phase-5r stopped at t={env.t}, expected {moment}")
    graph_last = np.asarray(env.uav_positions, dtype=np.float32) if last_positions is None else np.asarray(last_positions, dtype=np.float32)
    batch = graph_batch(env.info(), env, last_arrival, graph_last, urgency=shared_task_urgency(env.queue))
    leaked = set(batch).intersection(FORBIDDEN_INPUTS)
    if leaked or "j_mappo" in batch:
        raise RuntimeError(f"snapshot contains a label field: {sorted(leaked)}")
    cloned = {key: np.asarray(value).copy() for key, value in batch.items()}
    return env, np.asarray(last_movement, dtype=np.float64), cloned


def _mask(options: list[list[int]], num_uavs: int, num_buoys: int) -> np.ndarray:
    mask = np.zeros((int(num_uavs), int(num_buoys)), dtype=bool)
    for uav, buoys in enumerate(options):
        for buoy in buoys:
            mask[int(uav), int(buoy)] = True
    return mask


def candidate_pool(logits: np.ndarray, width: int, num_uavs: int, num_buoys: int) -> dict:
    options = candidate_lists(np.asarray(logits), int(width), int(num_uavs), int(num_buoys))
    assignments = maximal_assignments(options)
    stacked = np.stack([np.asarray(item, dtype=np.int64) for item in assignments], axis=0)
    return {
        "options": options,
        "assignments": stacked,
        "candidate_mask": _mask(options, int(num_uavs), int(num_buoys)),
        "has_idle": float(np.any(stacked < 0)),
        "n_unique": float(np.unique(stacked, axis=0).shape[0]),
    }


__all__ = ["candidate_pool", "replay_snapshot"]
