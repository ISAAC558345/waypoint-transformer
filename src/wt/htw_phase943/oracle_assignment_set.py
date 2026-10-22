"""Oracle / heuristic assignment as an unordered set of (uav, token) pairs."""
from __future__ import annotations

import numpy as np

from wt.htw_phase9.const import IDLE_INDEX, MAX_UAVS
from wt.l3b.policies import oracle_option_assignment


def assignment_to_tokens(assignment: np.ndarray, num_uavs: int = MAX_UAVS) -> np.ndarray:
    values = np.asarray(assignment, dtype=np.int64)
    tokens = np.full(int(num_uavs), IDLE_INDEX, dtype=np.int64)
    for uav in range(int(num_uavs)):
        if uav >= values.shape[0]:
            break
        tokens[uav] = IDLE_INDEX if int(values[uav]) < 0 else int(values[uav])
    return tokens


def tokens_to_assignment(tokens: np.ndarray, num_uavs: int) -> np.ndarray:
    assignment = np.full(int(num_uavs), -1, dtype=np.int64)
    for uav, token in enumerate(np.asarray(tokens, dtype=np.int64)[:num_uavs]):
        assignment[uav] = -1 if int(token) >= IDLE_INDEX else int(token)
    return assignment


__all__ = ["assignment_to_tokens", "oracle_option_assignment", "tokens_to_assignment"]
