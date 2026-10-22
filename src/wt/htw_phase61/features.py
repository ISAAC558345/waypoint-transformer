"""Level-2-A DeepSets features and oracle option one-hot. Does not edit L2A or Round-30 feature_dict."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from wt.l2a.env import L2AEnv

MAX_UAVS = 5
MAX_BUOYS = 12
UAV_DIM = 8
BUOY_DIM = 7
GLOBAL_DIM = 3
IDLE_INDEX = MAX_BUOYS
OPTION_DIM = MAX_BUOYS + 1
OPTION_UAV_DIM = UAV_DIM + OPTION_DIM


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


def attach_option(features: Mapping[str, np.ndarray], option: np.ndarray) -> dict[str, np.ndarray]:
    uav = np.asarray(features["uav"], dtype=np.float32)
    extra = np.asarray(option, dtype=np.float32)
    if uav.shape != (MAX_UAVS, UAV_DIM) or extra.shape != (MAX_UAVS, OPTION_DIM):
        raise ValueError("Phase-6.1 option adapter received an invalid shape")
    extra = extra * np.asarray(features["uav_mask"], dtype=np.float32)[:, None]
    output = {key: np.asarray(value).copy() for key, value in features.items()}
    output["uav"] = np.concatenate((uav, extra), axis=-1)
    output["_option"] = extra.copy()
    return output


def feature_dict_l2a(info: Mapping[str, Any], env: L2AEnv) -> dict[str, np.ndarray]:
    """Present-time L2-A observables only. No remaining arrivals."""
    config = env.config
    positions = np.asarray(info["uav_positions_m"], dtype=np.float32)
    buoys = np.asarray(info["buoy_positions_m"], dtype=np.float32)
    queue = np.asarray(info["queue_blocks"], dtype=np.float32)
    progress = np.asarray(info["service_progress"], dtype=np.float32)
    assignment = np.asarray(info["assignment"], dtype=np.int64)
    cap = float(config.queue_capacity_blocks)
    uav = np.zeros((MAX_UAVS, UAV_DIM), dtype=np.float32)
    buoy = np.zeros((MAX_BUOYS, BUOY_DIM), dtype=np.float32)
    uav_mask = np.zeros(MAX_UAVS, dtype=bool)
    buoy_mask = np.zeros(MAX_BUOYS, dtype=bool)
    assigned_queue = np.zeros(MAX_UAVS, dtype=np.float32)
    for uav_id, buoy_id in enumerate(assignment):
        if int(buoy_id) >= 0:
            assigned_queue[uav_id] = queue[int(buoy_id)] / cap
    uav[: config.num_uavs] = np.column_stack(
        (
            2.0 * positions[:, 0] / float(config.area_width) - 1.0,
            2.0 * positions[:, 1] / float(config.area_height) - 1.0,
            np.zeros(config.num_uavs, dtype=np.float32),
            np.zeros(config.num_uavs, dtype=np.float32),
            progress,
            (assignment >= 0).astype(np.float32),
            assigned_queue,
            np.ones(config.num_uavs, dtype=np.float32),
        )
    )
    occupied = np.zeros(config.num_buoys, dtype=np.float32)
    for buoy_id in assignment:
        if int(buoy_id) >= 0:
            occupied[int(buoy_id)] = 1.0
    buoy[: config.num_buoys] = np.column_stack(
        (
            2.0 * buoys[:, 0] / float(config.area_width) - 1.0,
            2.0 * buoys[:, 1] / float(config.area_height) - 1.0,
            np.clip(queue / cap, 0.0, 1.0),
            np.zeros(config.num_buoys, dtype=np.float32),
            np.zeros(config.num_buoys, dtype=np.float32),
            occupied,
            np.full(config.num_buoys, float(config.num_buoys) / MAX_BUOYS, dtype=np.float32),
        )
    )
    uav_mask[: config.num_uavs] = True
    buoy_mask[: config.num_buoys] = True
    global_ = np.asarray(
        (
            float(info["time_step"]) / float(config.episode_horizon),
            float(np.sum(queue)) / (cap * float(config.num_buoys)),
            float(config.num_uavs) / MAX_UAVS,
        ),
        dtype=np.float32,
    )
    output = {"uav": uav, "buoy": buoy, "uav_mask": uav_mask, "buoy_mask": buoy_mask, "global": global_}
    if not all(np.all(np.isfinite(value)) for value in output.values() if value.dtype != bool):
        raise RuntimeError("Phase-6.1 feature extraction produced NaN/Inf")
    return output


def stack_feature_dicts(items: Iterable[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    rows = list(items)
    if not rows:
        raise ValueError("cannot stack empty Phase-6.1 feature batch")
    return {key: np.stack([np.asarray(row[key]) for row in rows], axis=0) for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global")}


def in_range_completion(env: L2AEnv, assignment: np.ndarray) -> float:
    hits = []
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0:
            hits.append(1.0)
            continue
        hits.append(1.0 if env._in_range(uav, int(buoy)) else 0.0)
    return float(np.mean(hits)) if hits else 0.0


def service_success(info: Mapping[str, Any], assignment: np.ndarray) -> float:
    service = np.asarray(info.get("step_service_blocks", np.zeros(MAX_BUOYS)), dtype=np.float64)
    hits = []
    for buoy in np.asarray(assignment, dtype=np.int64):
        if int(buoy) < 0:
            continue
        hits.append(1.0 if float(service[int(buoy)]) > 0.0 else 0.0)
    return float(np.mean(hits)) if hits else 0.0


__all__ = [
    "BUOY_DIM",
    "GLOBAL_DIM",
    "IDLE_INDEX",
    "MAX_BUOYS",
    "MAX_UAVS",
    "OPTION_DIM",
    "OPTION_UAV_DIM",
    "UAV_DIM",
    "attach_option",
    "encode_option",
    "feature_dict_l2a",
    "in_range_completion",
    "service_success",
    "stack_feature_dicts",
]
