"""Oracle task option: per-UAV buoy one-hot plus idle. No Task-Waypoint."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from wt.round19.multi_env import SharedSpectrumMultiUAVEnv, gain_matrix, joint_aware_matching
from wt.round30.features import MAX_BUOYS, MAX_UAVS, UAV_DIM, feature_dict

IDLE_INDEX = MAX_BUOYS
OPTION_DIM = MAX_BUOYS + 1
OPTION_UAV_DIM = UAV_DIM + OPTION_DIM
OPTION_HOLD = 5
COMPLETION_COEF = 0.25


def oracle_assignment(info: Mapping[str, Any], env: SharedSpectrumMultiUAVEnv) -> np.ndarray:
    """Joint-aware matching at the current state; -1 is idle. Does not change env matching."""
    queues = np.asarray(info["raw_queue_bits"], dtype=np.float64)
    buoys = np.asarray(info["buoy_positions_m"], dtype=np.float64)
    positions = np.asarray(info["uav_positions_m"], dtype=np.float64)
    gains, _ranges = gain_matrix(positions, buoys, env.config)
    assignment, _matched, _scores, _evaluated = joint_aware_matching(
        queues, gains, env.config, float(env.options.interference_overlap_factor)
    )
    return np.asarray(assignment, dtype=np.int64)


def encode_option(assignment: np.ndarray) -> np.ndarray:
    option = np.zeros((MAX_UAVS, OPTION_DIM), dtype=np.float32)
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if uav >= MAX_UAVS:
            break
        if int(buoy) < 0:
            option[uav, IDLE_INDEX] = 1.0
        else:
            option[uav, int(buoy)] = 1.0
    return option


def decode_option(option: np.ndarray) -> np.ndarray:
    rows = np.asarray(option, dtype=np.float64)
    indices = np.argmax(rows, axis=-1)
    assignment = np.full(MAX_UAVS, -1, dtype=np.int64)
    for uav, index in enumerate(indices):
        if float(rows[uav, index]) <= 1.0e-8:
            assignment[uav] = -1
        elif int(index) == IDLE_INDEX:
            assignment[uav] = -1
        else:
            assignment[uav] = int(index)
    return assignment


def attach_option(features: Mapping[str, np.ndarray], option: np.ndarray) -> dict[str, np.ndarray]:
    uav = np.asarray(features["uav"], dtype=np.float32)
    if uav.ndim != 2 or uav.shape[-1] != UAV_DIM:
        raise ValueError("option adapter expects Round-30 UAV width")
    extra = np.asarray(option, dtype=np.float32)
    if extra.shape != (MAX_UAVS, OPTION_DIM):
        raise ValueError("option one-hot has invalid shape")
    extra = extra * np.asarray(features["uav_mask"], dtype=np.float32)[:, None]
    output = {key: np.asarray(value).copy() for key, value in features.items()}
    output["uav"] = np.concatenate((uav, extra), axis=-1)
    output["_option"] = extra.copy()
    return output


def oracle_option_features(info: Mapping[str, Any], env: SharedSpectrumMultiUAVEnv) -> dict[str, np.ndarray]:
    features = feature_dict(info, env.config)
    return attach_option(features, encode_option(oracle_assignment(info, env)))


def completion_score(assignment_after: np.ndarray, option_assignment: np.ndarray, uav_count: int) -> float:
    """1 if the realized matching equals the frozen option, else 0, averaged over UAVs."""
    realized = np.asarray(assignment_after, dtype=np.int64)
    intended = np.asarray(option_assignment, dtype=np.int64)
    hits = []
    for uav in range(int(uav_count)):
        hits.append(1.0 if int(realized[uav]) == int(intended[uav]) else 0.0)
    return float(np.mean(hits)) if hits else 0.0


__all__ = [
    "COMPLETION_COEF",
    "IDLE_INDEX",
    "OPTION_DIM",
    "OPTION_HOLD",
    "OPTION_UAV_DIM",
    "attach_option",
    "completion_score",
    "decode_option",
    "encode_option",
    "oracle_assignment",
    "oracle_option_features",
]
