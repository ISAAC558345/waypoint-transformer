"""Decision-time features of one candidate assignment.

The columns are current geometry, the current queue, and the SINR formula
applied at the current coordinates. They are not a MAPPO rollout and they
do not read J15 or a future SINR sequence.
"""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6ii_candidate_features.features import _link_sinr, _unique_mask
from wt.benchmark_v2.phase6is_assignment_head.protocol import RAW_NAMES
from wt.l3b.env import reference_rate


def assignment_raw_features(arrays: dict[str, np.ndarray], assignment: np.ndarray, config) -> np.ndarray:
    """Return ``(N, M, len(RAW_NAMES))`` in UAV order."""
    distance = np.asarray(arrays["uav_buoy_distance_m"], dtype=np.float64)
    gain = np.asarray(arrays["uav_buoy_gain"], dtype=np.float64)
    in_range = np.asarray(arrays["uav_buoy_in_range"], dtype=np.uint8) > 0
    hypothetical = np.asarray(arrays["uav_buoy_hypothetical_rate"], dtype=np.float64)
    queue_bits = np.asarray(arrays["q_bits"], dtype=np.float64)
    association = np.asarray(arrays["association"], dtype=np.int64).reshape(-1)
    progress = np.asarray(arrays["service_progress"], dtype=np.float64).reshape(-1)
    uav_position = np.asarray(arrays["uav_position_m"], dtype=np.float64)
    buoy_position = np.asarray(arrays["buoy_position_m"], dtype=np.float64)
    rows = np.asarray(assignment, dtype=np.int64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or int(rows.shape[1]) != int(distance.shape[0]):
        raise ValueError("assignment width does not match the UAV axis")
    valid = rows >= 0
    safe = np.where(valid, rows, 0)
    uav_index = np.arange(rows.shape[1], dtype=np.int64)[None, :]
    selected_distance = distance[uav_index, safe]
    selected_gain = gain[uav_index, safe]
    selected_range = in_range[uav_index, safe]
    selected_rate = hypothetical[uav_index, safe]
    queue = queue_bits[safe]
    nonempty = queue > 1.0e-9
    unique = _unique_mask(rows, valid, int(queue_bits.shape[0]))
    geometric_active = valid & nonempty & unique
    _rate, interference = _link_sinr(gain, gain, rows, geometric_active, config)
    target = buoy_position[safe]
    relative = target - uav_position[None, :, :]
    scale = np.asarray((float(config.area_width), float(config.area_height)), dtype=np.float64)
    relative = relative / scale
    diagonal = float(np.hypot(config.area_width, config.area_height))
    hold = float(config.option_hold)
    steps = np.maximum(selected_distance - float(config.service_radius_m), 0.0) / float(config.step_length_m)
    reference = max(float(reference_rate(config)), 1.0e-9)
    capacity = float(config.queue_capacity_blocks) * float(config.block_payload_bits)
    cluster = (safe >= 6).astype(np.int64)
    same = cluster[:, :, None] == cluster[:, None, :]
    pair = geometric_active[:, :, None] & geometric_active[:, None, :]
    others = np.sum(same & pair, axis=-1) - geometric_active.astype(np.int64)
    noise = max(float(config.noise_power), 1.0e-18)
    columns = (
        np.where(valid, relative[:, :, 0], 0.0),
        np.where(valid, relative[:, :, 1], 0.0),
        np.where(valid, selected_distance / diagonal, 0.0),
        np.where(valid, steps / hold, 0.0),
        selected_range.astype(np.float64) * valid.astype(np.float64),
        np.where(valid, np.clip(selected_rate / reference, 0.0, 2.0) / 2.0, 0.0),
        (rows != association[None, :]).astype(np.float64) * valid.astype(np.float64),
        np.where((rows != association[None, :]) & valid, progress[None, :], 0.0),
        np.where(valid, np.clip(queue / capacity, 0.0, 1.0), 0.0),
        np.where(geometric_active, np.log1p(interference / noise), 0.0),
        np.where(geometric_active, others / float(rows.shape[1] - 1), 0.0),
        (~valid).astype(np.float64),
    )
    packed = np.stack(columns, axis=-1)
    if int(packed.shape[-1]) != len(RAW_NAMES) or not np.all(np.isfinite(packed)):
        raise RuntimeError("assignment feature columns are not finite")
    return packed.astype(np.float32)


__all__ = ["assignment_raw_features"]
