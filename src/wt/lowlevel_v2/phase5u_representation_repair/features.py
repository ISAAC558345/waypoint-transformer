"""Current-state assignment features. Nothing here reads a future rollout."""
from __future__ import annotations

import numpy as np

from wt.l3b.config import L3BConfig
from wt.l3b.env import kappa_for, path_gain, sinr_rates_for_links

from .protocol import EDGE_FEATURES, PAIR_FEATURES, REL_DIM


def _stats(rows: list[np.ndarray], width: int) -> np.ndarray:
    """Feature-major mean/max/min/std so one feature group is a contiguous slice."""
    if not rows:
        return np.zeros(4 * width, dtype=np.float32)
    stacked = np.stack(rows, axis=0)
    parts = []
    for feature in range(width):
        column = stacked[:, feature]
        parts.extend((float(np.mean(column)), float(np.max(column)), float(np.min(column)), float(np.std(column))))
    return np.asarray(parts, dtype=np.float32)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-8:
        return np.zeros_like(vector)
    return vector / norm


def relational_matrix(snapshot: dict, assignments: np.ndarray, config: L3BConfig) -> np.ndarray:
    """Fixed-length z_rel. UAV order is removed by mean/max/min/std."""
    uav = np.asarray(snapshot["uav_pos"], dtype=np.float64)
    buoy = np.asarray(snapshot["buoy_pos"], dtype=np.float64)
    queue = np.asarray(snapshot["queue"], dtype=np.float64)
    urgency = np.asarray(snapshot["urgency"], dtype=np.float64)
    progress = np.asarray(snapshot["progress"], dtype=np.float64)
    current = np.asarray(snapshot["current_assignment"], dtype=np.int64)
    distance = np.asarray(snapshot["distance"], dtype=np.float64)
    eta = np.asarray(snapshot["eta"], dtype=np.float64)
    rate = np.asarray(snapshot["rate"], dtype=np.float64)
    diag = float(np.hypot(config.area_width, config.area_height))
    capacity = max(float(config.queue_capacity_blocks), 1.0e-9)
    horizon = max(float(config.episode_horizon), 1.0)
    reference = max(float(snapshot["reference_rate"]), 1.0e-9)
    radius = float(config.service_radius_m)
    occupied = np.zeros(queue.shape[0], dtype=np.float64)
    for target in current:
        if int(target) >= 0:
            occupied[int(target)] = 1.0
    chosen = np.asarray(assignments, dtype=np.int64)
    output = np.zeros((chosen.shape[0], REL_DIM), dtype=np.float32)
    for index, assignment in enumerate(chosen):
        edges = []
        links = []
        for uav_id, target in enumerate(assignment):
            if int(target) < 0:
                continue
            buoy_id = int(target)
            offset = buoy[buoy_id] - uav[uav_id]
            gain = path_gain(config, uav[uav_id], buoy[buoy_id])
            edges.append(
                np.asarray(
                    (
                        float(distance[uav_id, buoy_id]) / diag,
                        float(eta[uav_id, buoy_id]) / horizon,
                        float(offset[0]) / float(config.area_width),
                        float(offset[1]) / float(config.area_height),
                        float(rate[uav_id, buoy_id]) / reference,
                        float(np.log1p(gain)),
                        float(queue[buoy_id]) / capacity,
                        float(urgency[buoy_id]),
                        float(progress[uav_id]),
                        float(distance[uav_id, buoy_id] <= radius),
                        float(int(current[uav_id]) == buoy_id),
                        float(occupied[buoy_id]),
                    ),
                    dtype=np.float64,
                )
            )
            links.append((uav_id, buoy_id))
        pairs = []
        interference = []
        for left in range(len(links)):
            uav_i, buoy_i = links[left]
            toward_i = buoy[buoy_i] - uav[uav_i]
            for right in range(left + 1, len(links)):
                uav_j, buoy_j = links[right]
                toward_j = buoy[buoy_j] - uav[uav_j]
                pairs.append(
                    np.asarray(
                        (
                            float(np.linalg.norm(uav[uav_i] - uav[uav_j])) / diag,
                            float(np.linalg.norm(buoy[buoy_i] - buoy[buoy_j])) / diag,
                            float(np.dot(_unit(toward_i), _unit(toward_j))),
                            float(np.linalg.norm(toward_i - toward_j)) / diag,
                            float(min(np.linalg.norm(uav[uav_i] - buoy[buoy_j]), np.linalg.norm(uav[uav_j] - buoy[buoy_i]))) / diag,
                            float(max(np.linalg.norm(uav[uav_i] - buoy[buoy_j]), np.linalg.norm(uav[uav_j] - buoy[buoy_i]))) / diag,
                        ),
                        dtype=np.float64,
                    )
                )
                gain_ji = path_gain(config, uav[uav_j], buoy[buoy_i])
                gain_ij = path_gain(config, uav[uav_i], buoy[buoy_j])
                interference.append(gain_ji * float(config.tx_power) * kappa_for(config, buoy_j, buoy_i))
                interference.append(gain_ij * float(config.tx_power) * kappa_for(config, buoy_i, buoy_j))
        sinr = np.zeros(0, dtype=np.float64)
        if links:
            holder = type("Holder", (), {"config": config, "uav_positions": uav, "buoy_positions": buoy})()
            sinr_values, _rates = sinr_rates_for_links(holder, links)
            sinr = np.asarray([float(sinr_values[uav_id]) for uav_id, _buoy_id in links], dtype=np.float64)
        logged = np.log1p(np.asarray(interference, dtype=np.float64)) if interference else np.zeros(0)
        interference_summary = np.zeros(5, dtype=np.float32)
        if logged.size:
            interference_summary = np.asarray(
                (
                    float(np.mean(logged)),
                    float(np.max(logged)),
                    float(np.min(logged)),
                    float(np.log1p(np.sum(interference))),
                    float(np.std(logged)),
                ),
                dtype=np.float32,
            )
        sinr_summary = np.zeros(3, dtype=np.float32)
        if sinr.size:
            logged_sinr = np.log1p(sinr)
            sinr_summary = np.asarray((float(np.mean(logged_sinr)), float(np.min(logged_sinr)), float(np.max(logged_sinr))), dtype=np.float32)
        output[index] = np.concatenate(
            (
                _stats(edges, len(EDGE_FEATURES)),
                _stats(pairs, len(PAIR_FEATURES)),
                interference_summary,
                sinr_summary,
            )
        )
    return output


__all__ = ["relational_matrix"]
