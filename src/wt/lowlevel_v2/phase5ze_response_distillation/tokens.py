"""Per-UAV inputs already present in the frozen REP and TEMP edges."""
from __future__ import annotations

import numpy as np

from wt.l3b.config import L3BConfig
from wt.l3b.env import path_gain
from wt.lowlevel_v2.phase5l_joint_encoder.protocol import WIDTH
from wt.lowlevel_v2.phase5u_representation_repair.protocol import EDGE_FEATURES as REP_EDGE_FEATURES
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import EDGE_FEATURES as TEMP_EDGE_FEATURES

TOKEN_BLOCKS = (
    ("h_u", WIDTH),
    ("h_b", WIDTH),
    ("h_edge", WIDTH),
    ("rep", len(REP_EDGE_FEATURES)),
    ("temp", len(TEMP_EDGE_FEATURES)),
    ("assigned", 1),
    ("idle", 1),
    ("act0", 2),
)


def token_dim() -> int:
    return int(sum(width for _name, width in TOKEN_BLOCKS))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-6:
        return np.zeros_like(vector)
    return vector / norm


def rep_temp_edges(relational: dict, history: dict, assignment: np.ndarray, config: L3BConfig) -> tuple[np.ndarray, np.ndarray]:
    """One REP edge and one TEMP edge per UAV. Idle rows stay zero. Queue and progress stay inside those edges."""
    chosen = np.asarray(assignment, dtype=np.int64)
    count = int(chosen.shape[0])
    rep = np.zeros((count, len(REP_EDGE_FEATURES)), dtype=np.float64)
    temp = np.zeros((count, len(TEMP_EDGE_FEATURES)), dtype=np.float64)
    uav = np.asarray(relational["uav_pos"], dtype=np.float64)
    buoy = np.asarray(relational["buoy_pos"], dtype=np.float64)
    queue = np.asarray(relational["queue"], dtype=np.float64)
    urgency = np.asarray(relational["urgency"], dtype=np.float64)
    progress = np.asarray(relational["progress"], dtype=np.float64)
    current = np.asarray(relational["current_assignment"], dtype=np.int64)
    distance = np.asarray(relational["distance"], dtype=np.float64)
    eta = np.asarray(relational["eta"], dtype=np.float64)
    rate = np.asarray(relational["rate"], dtype=np.float64)
    diag = float(np.hypot(config.area_width, config.area_height))
    capacity = max(float(config.queue_capacity_blocks), 1.0e-9)
    horizon = max(float(config.episode_horizon), 1.0)
    reference = max(float(relational["reference_rate"]), 1.0e-9)
    radius = float(config.service_radius_m)
    occupied = np.zeros(queue.shape[0], dtype=np.float64)
    for target in current:
        if int(target) >= 0:
            occupied[int(target)] = 1.0
    velocity = np.asarray(history["velocity"], dtype=np.float64)
    last_sinr = np.asarray(history["last_sinr"], dtype=np.float64)
    step = max(float(config.step_length_m), 1.0e-9)
    previous = uav - velocity * step
    buoy_progress = np.zeros(buoy.shape[0], dtype=np.float64)
    for uav_id, target in enumerate(current):
        if int(target) >= 0:
            buoy_progress[int(target)] = max(float(buoy_progress[int(target)]), float(progress[uav_id]))
    for uav_id, target in enumerate(chosen):
        if int(target) < 0:
            continue
        buoy_id = int(target)
        offset = buoy[buoy_id] - uav[uav_id]
        gain = path_gain(config, uav[uav_id], buoy[buoy_id])
        rep[uav_id] = (
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
        )
        dist_now = float(np.linalg.norm(offset))
        dist_prev = float(np.linalg.norm(buoy[buoy_id] - previous[uav_id]))
        direction = _unit(offset)
        speed = _unit(velocity[uav_id])
        valid = float(np.linalg.norm(velocity[uav_id]) >= 1.0e-6)
        same = float(int(current[uav_id]) == buoy_id)
        closing = float(np.dot(velocity[uav_id], direction))
        heading = float(np.dot(speed, direction)) if valid else 0.0
        temp[uav_id] = (
            same,
            1.0 - same,
            float(np.log1p(max(float(last_sinr[uav_id]), 0.0))) * same,
            float(progress[uav_id]) * same,
            float(buoy_progress[buoy_id]),
            closing,
            heading,
            valid,
            (dist_now - dist_prev) / step,
        )
    return rep, temp


def build_tokens(
    h_u: np.ndarray,
    h_b: np.ndarray,
    h_edge: np.ndarray,
    rep: np.ndarray,
    temp: np.ndarray,
    assignments: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Stack the frozen per-UAV inputs. Idle buoy and edge slots are zero."""
    chosen = np.asarray(assignments, dtype=np.int64)
    count, uavs = int(chosen.shape[0]), int(chosen.shape[1])
    buoy = np.clip(chosen, 0, None)
    selected = chosen >= 0
    chosen_b = h_b[buoy]
    chosen_e = h_edge[np.arange(uavs)[None, :], buoy]
    chosen_b = np.where(selected[..., None], chosen_b, 0.0)
    chosen_e = np.where(selected[..., None], chosen_e, 0.0)
    expanded = np.broadcast_to(h_u, (count, uavs, h_u.shape[-1]))
    assigned = selected.astype(np.float64)[..., None]
    idle = (~selected).astype(np.float64)[..., None]
    tokens = np.concatenate((expanded, chosen_b, chosen_e, rep, temp, assigned, idle, np.asarray(actions, dtype=np.float64)), axis=-1)
    if int(tokens.shape[-1]) != token_dim():
        raise RuntimeError("token width drifted from the registered blocks")
    return tokens.astype(np.float32)


__all__ = ["TOKEN_BLOCKS", "build_tokens", "rep_temp_edges", "token_dim"]
