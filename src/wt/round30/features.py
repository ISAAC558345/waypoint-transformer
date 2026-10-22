"""Current-observable, ID-free feature extraction for Round-30 PPO."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


MAX_UAVS, MAX_BUOYS = 2, 5
UAV_DIM, BUOY_DIM, GLOBAL_DIM = 8, 7, 3


def _masked(values: np.ndarray, count: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    result, mask = np.zeros((count, width), dtype=np.float32), np.zeros(count, dtype=bool)
    return result, mask


def feature_dict(info: Mapping[str, Any], config: Any) -> dict[str, np.ndarray]:
    """Build only present-time physical features; no IDs, indices, or future values."""
    m_count, k_count = int(info["num_uavs"]), int(info["num_buoys"])
    if m_count != MAX_UAVS or k_count not in {3, 5}:
        raise ValueError("Round-30 supports only the frozen M=2 / K={3,5} task")
    positions = np.asarray(info["uav_positions_m"], dtype=np.float32)
    velocity = np.asarray(info["uav_velocity_mps"], dtype=np.float32)
    energy = np.asarray(info["remaining_energy_per_uav"], dtype=np.float32)
    path = np.asarray(info["cumulative_distance_m_per_uav"], dtype=np.float32)
    payload = np.asarray(info["transmitted_payload_bits_by_uav"], dtype=np.float32)
    assignment = np.asarray(info["uav_to_buoy_assignment"], dtype=np.int64)
    buoy_positions = np.asarray(info["buoy_positions_m"], dtype=np.float32)
    queue = np.asarray(info["raw_queue_bits"], dtype=np.float32)
    arrival = np.asarray(info["raw_arrival_bits"], dtype=np.float32)
    cumulative = np.asarray(info["cumulative_payload_bits_per_buoy"], dtype=np.float32)
    matched = np.asarray(info["matched_uav_by_buoy"], dtype=np.int64)
    max_step = float(config.max_speed * config.delta_t)
    cap = float(config.queue_capacity_bits)
    uav, uav_mask = _masked(np.empty(0), MAX_UAVS, UAV_DIM)
    buoy, buoy_mask = _masked(np.empty(0), MAX_BUOYS, BUOY_DIM)
    uav[:m_count] = np.column_stack((
        2.0 * positions[:, 0] / float(config.area_width) - 1.0,
        2.0 * positions[:, 1] / float(config.area_height) - 1.0,
        velocity[:, 0] / float(config.max_speed), velocity[:, 1] / float(config.max_speed),
        energy / float(config.initial_energy),
        np.clip(path / max(float(config.episode_horizon) * max_step, 1.0), 0.0, 1.0),
        np.clip(payload / cap, 0.0, 1.0), (assignment >= 0).astype(np.float32),
    ))
    buoy[:k_count] = np.column_stack((
        2.0 * buoy_positions[:, 0] / float(config.area_width) - 1.0,
        2.0 * buoy_positions[:, 1] / float(config.area_height) - 1.0,
        np.clip(queue / cap, 0.0, 1.0), np.clip(arrival / cap, 0.0, 1.0),
        np.clip(cumulative / (cap * float(config.episode_horizon)), 0.0, 1.0),
        (matched >= 0).astype(np.float32), np.full(k_count, float(k_count) / MAX_BUOYS),
    ))
    uav_mask[:m_count], buoy_mask[:k_count] = True, True
    global_ = np.asarray((float(k_count) / MAX_BUOYS, float(info["interference_overlap_factor"]), float(info["time_step"]) / float(config.episode_horizon)), dtype=np.float32)
    output = {"uav": uav, "buoy": buoy, "uav_mask": uav_mask, "buoy_mask": buoy_mask, "global": global_}
    if not all(np.all(np.isfinite(value)) for value in output.values() if value.dtype != bool):
        raise RuntimeError("Round-30 feature extraction produced NaN/Inf")
    return output


def stack_feature_dicts(items: Iterable[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    rows = list(items)
    if not rows:
        raise ValueError("cannot stack empty Round-30 feature batch")
    return {key: np.stack([np.asarray(row[key]) for row in rows], axis=0) for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global")}


__all__ = ["BUOY_DIM", "GLOBAL_DIM", "MAX_BUOYS", "MAX_UAVS", "UAV_DIM", "feature_dict", "stack_feature_dicts"]
