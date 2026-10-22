"""Deterministic Oracle assignment sequences: UAV index order, idle if unassigned."""
from __future__ import annotations

import numpy as np

from wt.htw_phase9.const import IDLE_INDEX, MAX_UAVS
from wt.l3b.env import L3BEnv
from wt.l3b.policies import oracle_option_assignment


def serialize_oracle_assignment(assignment: np.ndarray, num_uavs: int = MAX_UAVS) -> np.ndarray:
    values = np.asarray(assignment, dtype=np.int64)
    sequence = np.full(int(num_uavs), IDLE_INDEX, dtype=np.int64)
    for uav in range(int(num_uavs)):
        if uav >= values.shape[0]:
            break
        buoy = int(values[uav])
        sequence[uav] = IDLE_INDEX if buoy < 0 else buoy
    return sequence


def oracle_sequence_from_env(env: L3BEnv) -> np.ndarray:
    return serialize_oracle_assignment(oracle_option_assignment(env), env.config.num_uavs)


def sequence_to_assignment(sequence: np.ndarray, num_uavs: int) -> np.ndarray:
    assignment = np.full(int(num_uavs), -1, dtype=np.int64)
    for uav, token in enumerate(np.asarray(sequence, dtype=np.int64)[:num_uavs]):
        assignment[uav] = -1 if int(token) >= IDLE_INDEX or int(token) < 0 else int(token)
    return assignment


__all__ = ["oracle_sequence_from_env", "sequence_to_assignment", "serialize_oracle_assignment"]
