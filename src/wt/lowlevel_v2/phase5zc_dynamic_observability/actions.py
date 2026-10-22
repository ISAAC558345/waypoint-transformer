"""One frozen policy forward. The environment is not stepped."""
from __future__ import annotations

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.contracts import refuse_rematch


def radial_components(
    action: np.ndarray,
    uav_pos: np.ndarray,
    buoy_pos: np.ndarray,
    assignment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radial is the action component toward the assigned buoy. Idle radial is zero."""
    command = np.asarray(action, dtype=np.float64)
    radial = np.zeros(command.shape[0], dtype=np.float64)
    tangential = np.zeros(command.shape[0], dtype=np.float64)
    heading = np.arctan2(command[:, 1], command[:, 0])
    for index, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0:
            continue
        offset = np.asarray(buoy_pos[int(buoy)] - uav_pos[index], dtype=np.float64)
        norm = float(np.linalg.norm(offset))
        if norm < 1.0e-8:
            continue
        unit = offset / norm
        radial[index] = float(np.dot(command[index], unit))
        tangential[index] = float(np.linalg.norm(command[index] - radial[index] * unit))
    return radial, tangential, heading


def assignment_option(env: L3BEnv, assignment: np.ndarray) -> TaskOption:
    chosen = np.asarray(assignment, dtype=np.int64).copy()
    option = TaskOption(
        targets=chosen.copy(),
        decision_time=int(env.t),
        remaining_hold=15,
        uav_mask=np.ones(env.config.num_uavs, dtype=bool),
        buoy_mask=np.ones(env.config.num_buoys, dtype=bool),
        association=chosen.copy(),
        urgency=None,
    )
    refuse_rematch(option)
    return option


def mean_active(values: np.ndarray, assignment: np.ndarray) -> float:
    mask = np.asarray(assignment, dtype=np.int64) >= 0
    chosen = np.asarray(values, dtype=np.float64)[mask]
    if chosen.size == 0:
        return float("nan")
    return float(np.mean(chosen))


__all__ = ["assignment_option", "mean_active", "radial_components"]
