"""Phase-9 constants. L3-B is M=6; Phase-6.1 MAX_UAVS=5 is not reused."""
from __future__ import annotations

MAX_UAVS = 6
MAX_BUOYS = 12
UAV_DIM = 8
BUOY_DIM = 7
GLOBAL_DIM = 3
IDLE_INDEX = MAX_BUOYS
OPTION_DIM = MAX_BUOYS + 1
EDGE_UB_DIM = 3
EDGE_UU_DIM = 2
HOLD = 15
REWARD_MODE = "delta"
SEED = 0
TRAIN_SEEDS = tuple(range(8))
EVAL_SEEDS = tuple(range(8, 16))
MINI_UPDATES = 80


def assignment_to_indices(assignment, max_uavs: int = MAX_UAVS) -> "np.ndarray":
    import numpy as np

    indices = np.full(max_uavs, IDLE_INDEX, dtype=np.int64)
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if uav >= max_uavs:
            break
        indices[uav] = IDLE_INDEX if int(buoy) < 0 else int(buoy)
    return indices


def indices_to_assignment(indices, num_uavs: int | None = None) -> "np.ndarray":
    import numpy as np

    values = np.asarray(indices, dtype=np.int64)
    count = int(values.shape[0] if num_uavs is None else num_uavs)
    assignment = np.full(count, -1, dtype=np.int64)
    for uav in range(count):
        index = int(values[uav]) if uav < values.shape[0] else IDLE_INDEX
        assignment[uav] = -1 if index >= IDLE_INDEX or index < 0 else index
    return assignment


__all__ = [
    "BUOY_DIM",
    "EDGE_UB_DIM",
    "EDGE_UU_DIM",
    "EVAL_SEEDS",
    "GLOBAL_DIM",
    "HOLD",
    "IDLE_INDEX",
    "MAX_BUOYS",
    "MAX_UAVS",
    "MINI_UPDATES",
    "OPTION_DIM",
    "REWARD_MODE",
    "SEED",
    "TRAIN_SEEDS",
    "UAV_DIM",
    "assignment_to_indices",
    "indices_to_assignment",
]
