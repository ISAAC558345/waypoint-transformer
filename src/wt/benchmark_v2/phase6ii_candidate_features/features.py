"""Candidate-conditioned features from the current state only.

No rollout, no J15 label, and no copied current-association interference.
Instantaneous rates follow the environment's SINR formula. The straight-line
window is a max-speed kinematic bound, not a MAPPO trajectory.
"""
from __future__ import annotations

import numpy as np


def _as_assignment(assignment: np.ndarray, n_uavs: int) -> np.ndarray:
    rows = np.asarray(assignment, dtype=np.int64)
    if rows.ndim == 1:
        rows = rows.reshape(1, n_uavs)
    if rows.ndim != 2 or int(rows.shape[1]) != int(n_uavs):
        raise ValueError("assignment must have one column per UAV")
    return rows


def _gather_pairs(matrix: np.ndarray, assignment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe = np.where(assignment >= 0, assignment, 0)
    uav_index = np.arange(matrix.shape[0], dtype=np.int64)[None, :]
    return matrix[uav_index, safe], assignment >= 0


def _unique_mask(assignment: np.ndarray, valid: np.ndarray, n_buoys: int) -> np.ndarray:
    safe = np.where(valid, assignment, 0)
    counts = np.zeros((assignment.shape[0], int(n_buoys)), dtype=np.int16)
    row = np.arange(assignment.shape[0], dtype=np.int64)[:, None]
    row = np.broadcast_to(row, assignment.shape)
    np.add.at(counts, (row, safe), valid.astype(np.int16))
    return counts[row, safe] == 1


def _kappa(config, assignment: np.ndarray) -> np.ndarray:
    safe = np.where(assignment >= 0, assignment, 0)
    cluster = (safe >= 6).astype(np.int64)
    same = cluster[:, :, None] == cluster[:, None, :]
    return np.where(same, float(config.kappa_same_cluster), float(config.kappa_cross_cluster)).astype(np.float64)


def _link_sinr(
    signal_gain: np.ndarray,
    interferer_gain: np.ndarray,
    assignment: np.ndarray,
    active: np.ndarray,
    config,
    pair_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rate and interference power above noise for each assigned UAV."""
    safe = np.where(assignment >= 0, assignment, 0)
    n_uavs = signal_gain.shape[0]
    signal = np.moveaxis(signal_gain[:, safe], 0, -1)[:, np.arange(n_uavs), np.arange(n_uavs)] * float(config.tx_power)
    gathered = np.moveaxis(interferer_gain[:, safe], 0, -1)
    eye = np.eye(n_uavs, dtype=bool)
    pair = active[:, :, None] & active[:, None, :] & ~eye
    weight = pair.astype(np.float64) if pair_weight is None else pair_weight * pair
    interference = np.sum(gathered * float(config.tx_power) * _kappa(config, assignment) * weight, axis=-1)
    sinr = np.where(active, signal / np.maximum(float(config.noise_power) + interference, 1.0e-18), 0.0)
    rate = np.where(active, float(config.bandwidth_hz) * np.log2(1.0 + sinr), 0.0)
    return rate, np.where(active, interference, 0.0)


def _boundary_gain(distance: np.ndarray, config) -> np.ndarray:
    clipped = np.minimum(np.maximum(distance, float(config.min_link_m)), float(config.service_radius_m))
    return clipped ** (-float(config.path_loss_exp))


def _hover_interference(buoy_position: np.ndarray, assignment: np.ndarray, active: np.ndarray, config) -> np.ndarray:
    """Directed interference after every serving UAV sits on its buoy.

    Any two assignments that serve the same buoy set have the same sum.
    """
    delta = buoy_position[:, None, :] - buoy_position[None, :, :]
    distance = np.maximum(np.linalg.norm(delta, axis=-1), float(config.min_link_m))
    buoy_gain = distance ** (-float(config.path_loss_exp))
    safe = np.where(assignment >= 0, assignment, 0)
    n_uavs = assignment.shape[1]
    pair_gain = buoy_gain[safe[:, :, None], safe[:, None, :]]
    eye = np.eye(n_uavs, dtype=bool)
    pair = active[:, :, None] & active[:, None, :] & ~eye
    return np.sum(pair_gain * float(config.tx_power) * _kappa(config, assignment) * pair, axis=(1, 2))


def kinematic_delivered_bits(arrays: dict[str, np.ndarray], assignment: np.ndarray, config) -> np.ndarray:
    """Bits collected while every UAV flies straight to its buoy at max speed.

    This is not a MAPPO rollout and it does not add future arrivals. A UAV
    transmits only inside the service radius, using the path-loss SINR at the
    position implied by the straight line.
    """
    uav_position = np.asarray(arrays["uav_position_m"], dtype=np.float64)
    buoy_position = np.asarray(arrays["buoy_position_m"], dtype=np.float64)
    queue_bits = np.asarray(arrays["q_bits"], dtype=np.float64)
    rows = _as_assignment(assignment, uav_position.shape[0])
    valid = rows >= 0
    safe = np.where(valid, rows, 0)
    unique = _unique_mask(rows, valid, int(queue_bits.shape[0]))
    target = buoy_position[safe]
    current = np.broadcast_to(uav_position[None, :, :], target.shape).copy()
    remaining = np.where(valid, queue_bits[safe], 0.0)
    eligible = valid & unique & (remaining > 1.0e-9)
    delivered = np.zeros(rows.shape[0], dtype=np.float64)
    step = float(config.step_length_m)
    minimum = float(config.min_link_m)
    exponent = float(config.path_loss_exp)
    radius = float(config.service_radius_m)
    n_uavs = rows.shape[1]
    eye = np.eye(n_uavs, dtype=bool)
    kappa = _kappa(config, rows)
    for _step_index in range(int(config.option_hold)):
        delta = target - current
        distance = np.linalg.norm(delta, axis=-1)
        scale = np.minimum(step, distance) / np.maximum(distance, 1.0e-12)
        current = current + delta * scale[:, :, None]
        pair_delta = current[:, None, :, :] - target[:, :, None, :]
        pair_distance = np.linalg.norm(pair_delta, axis=-1)
        gain = np.maximum(pair_distance, minimum) ** (-exponent)
        serving = pair_distance[:, np.arange(n_uavs), np.arange(n_uavs)]
        active = eligible & (serving <= radius) & (remaining > 1.0e-9)
        signal = gain[:, np.arange(n_uavs), np.arange(n_uavs)] * float(config.tx_power)
        pair = active[:, :, None] & active[:, None, :] & ~eye
        interference = np.sum(gain * float(config.tx_power) * kappa * pair, axis=-1)
        sinr = np.where(active, signal / np.maximum(float(config.noise_power) + interference, 1.0e-18), 0.0)
        rate = np.where(active, float(config.bandwidth_hz) * np.log2(1.0 + sinr), 0.0)
        taken = np.minimum(remaining, rate * float(config.delta_t))
        remaining = remaining - taken
        delivered += np.sum(taken, axis=1)
    return delivered


def candidate_features(arrays: dict[str, np.ndarray], assignment: np.ndarray, config) -> dict[str, np.ndarray]:
    """Features for every assignment row. Arrays are ``(N, M)`` unless noted."""
    distance = np.asarray(arrays["uav_buoy_distance_m"], dtype=np.float64)
    gain = np.asarray(arrays["uav_buoy_gain"], dtype=np.float64)
    in_range = np.asarray(arrays["uav_buoy_in_range"], dtype=np.uint8) > 0
    hypothetical = np.asarray(arrays["uav_buoy_hypothetical_rate"], dtype=np.float64)
    queue_bits = np.asarray(arrays["q_bits"], dtype=np.float64)
    association = np.asarray(arrays["association"], dtype=np.int64).reshape(-1)
    progress = np.asarray(arrays["service_progress"], dtype=np.float64).reshape(-1)
    buoy_position = np.asarray(arrays["buoy_position_m"], dtype=np.float64)
    rows = _as_assignment(assignment, distance.shape[0])
    selected_distance, valid = _gather_pairs(distance, rows)
    selected_gain, _ignored = _gather_pairs(gain, rows)
    selected_range, _ignored = _gather_pairs(in_range, rows)
    selected_hypothetical, _ignored = _gather_pairs(hypothetical, rows)
    safe = np.where(valid, rows, 0)
    queue = queue_bits[safe]
    nonempty = queue > 1.0e-9
    unique = _unique_mask(rows, valid, int(queue_bits.shape[0]))
    instant_active = valid & selected_range & nonempty & unique
    geometric_active = valid & nonempty & unique
    instant_rate, instant_interference = _link_sinr(gain, gain, rows, instant_active, config)
    geometric_rate, geometric_interference = _link_sinr(gain, gain, rows, geometric_active, config)
    noise_only = np.where(
        valid & nonempty,
        float(config.bandwidth_hz) * np.log2(1.0 + selected_gain * float(config.tx_power) / max(float(config.noise_power), 1.0e-18)),
        0.0,
    )
    steps = np.where(valid, np.maximum(selected_distance - float(config.service_radius_m), 0.0) / float(config.step_length_m), 0.0)
    hold = float(config.option_hold)
    can_arrive = geometric_active & (steps <= hold + 1.0e-12)
    signal_gain = _boundary_gain(distance, config)
    start = steps
    end = np.where(can_arrive, hold, steps)
    duration = np.maximum(end - start, 0.0)
    overlap = np.maximum(0.0, np.minimum(end[:, :, None], end[:, None, :]) - np.maximum(start[:, :, None], start[:, None, :]))
    weight = np.where(duration[:, :, None] > 0.0, overlap / np.maximum(duration[:, :, None], 1.0e-12), 0.0)
    window_rate, _window_interference = _link_sinr(signal_gain, gain, rows, can_arrive, config, weight)
    straight_line_bits = np.minimum(np.where(valid, queue, 0.0), window_rate * duration * float(config.delta_t))
    switch = (rows != association[None, :]).astype(np.float64)
    return {
        "distance_m": np.where(valid, selected_distance, 0.0),
        "steps_to_radius": steps,
        "in_range": selected_range.astype(np.float64) * valid.astype(np.float64),
        "hypothetical_rate_bps": np.where(valid, selected_hypothetical, 0.0),
        "noise_only_rate_bps": noise_only,
        "switch": switch,
        "abandoned_progress": np.where(switch > 0.0, progress[None, :], 0.0),
        "instant_rate_bps": instant_rate,
        "instant_interference_w": instant_interference,
        "geometric_rate_bps": geometric_rate,
        "geometric_interference_w": geometric_interference,
        "straight_line_bits": straight_line_bits,
        "kinematic_bits": kinematic_delivered_bits(arrays, rows, config),
        "hover_interference_w": _hover_interference(buoy_position, rows, geometric_active, config),
    }


def feature_sums(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sums = {}
    for name, value in features.items():
        array = np.asarray(value, dtype=np.float64)
        sums[name] = np.sum(array, axis=1) if array.ndim == 2 else array.reshape(-1)
    return sums


__all__ = ["candidate_features", "feature_sums", "kinematic_delivered_bits"]
