"""Fifteen-step traces. Records only quantities the environment already exposes."""
from __future__ import annotations

import copy

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import refuse_rematch
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor


def _option(env: L3BEnv, assignment: np.ndarray) -> TaskOption:
    targets = np.asarray(assignment, dtype=np.int64).copy()
    option = TaskOption(
        targets=targets,
        decision_time=int(env.t),
        remaining_hold=15,
        uav_mask=np.ones(env.config.num_uavs, dtype=bool),
        buoy_mask=np.ones(env.config.num_buoys, dtype=bool),
        association=targets.copy(),
        urgency=None,
    )
    refuse_rematch(option)
    return option


def _interference(info: dict) -> float:
    matrix = np.asarray(info["interference_uav_uav"], dtype=np.float64)
    if matrix.size == 0:
        return 0.0
    return float(np.sum(matrix))


def _min_separation(info: dict) -> float:
    distance = np.asarray(info["distance_uav_uav"], dtype=np.float64)
    mask = ~np.eye(distance.shape[0], dtype=bool)
    if not np.any(mask):
        return float("nan")
    return float(np.min(distance[mask]))


def trace_h15(env: L3BEnv, assignment: np.ndarray, actor, last_movement: np.ndarray, mode: str) -> dict:
    if mode not in {"fly", "mappo"}:
        raise ValueError(mode)
    clone = copy.deepcopy(env)
    option = _option(clone, assignment)
    executor = LockedExecutor(actor) if mode == "mappo" else None
    start = float(clone.payload_bits)
    movement = np.asarray(last_movement, dtype=np.float64).copy()
    conflict_rates = []
    conflict_pairs = []
    interference = []
    separation = []
    for _ in range(15):
        if int(clone.t) >= int(clone.config.episode_horizon):
            break
        if mode == "fly":
            executed = fly_toward(clone, option.association)
        else:
            executed, _desired, _delta = executor.act(clone, option, movement)
            movement = np.asarray(executed, dtype=np.float64)
        info = clone.step(executed, option.association)
        conflict_rates.append(float(info["conflict_rate"]))
        conflict_pairs.append(float(info["n_conflict_pairs"]))
        interference.append(_interference(info))
        separation.append(_min_separation(info))
    return {
        "payload": float(clone.payload_bits - start),
        "conflict_rate": float(np.mean(conflict_rates)) if conflict_rates else 0.0,
        "conflict_pairs": float(np.mean(conflict_pairs)) if conflict_pairs else 0.0,
        "interference": float(np.mean(interference)) if interference else 0.0,
        "min_uav_distance_m": float(np.min(separation)) if separation else float("nan"),
    }


__all__ = ["trace_h15"]
