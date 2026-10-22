"""Current-state features. No future trajectory and no remaining-arrival lookup."""
from __future__ import annotations

import numpy as np

from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.phase5_hgnn_urgency.urgency import shared_task_urgency


def energy_cost(env: L3BEnv) -> float:
    """L3-B has no energy state. Do not invent one."""
    if any(hasattr(env, name) for name in ("energy", "uav_energy", "battery")):
        raise RuntimeError("phase-5e found an energy attribute and will not invent a cost")
    return 0.0


def current_observation(env: L3BEnv, last_positions: np.ndarray | None) -> dict[str, np.ndarray]:
    """Decision-time observation. Velocity is the last completed step, or zero at t=0."""
    info = env.info()
    positions = np.asarray(info["uav_positions_m"], dtype=np.float64)
    if last_positions is None:
        velocity = np.zeros_like(positions)
    else:
        previous = np.asarray(last_positions, dtype=np.float64)
        velocity = (positions - previous) / max(float(env.config.step_length_m), 1.0e-9)
    assignment = np.asarray(info["assignment"], dtype=np.int64)
    queue = np.asarray(env.queue, dtype=np.float64)
    capacity = float(env.config.queue_capacity_blocks)
    assigned_queue = np.zeros(env.config.num_uavs, dtype=np.float64)
    for uav, buoy in enumerate(assignment):
        if int(buoy) >= 0:
            assigned_queue[uav] = queue[int(buoy)]
    return {
        "uav_position": positions,
        "uav_velocity": velocity,
        "uav_assigned_flag": (assignment >= 0).astype(np.float64),
        "uav_assigned_queue": assigned_queue,
        "uav_channel": np.asarray(info["rate_bps"], dtype=np.float64),
        "buoy_position": np.asarray(info["buoy_positions_m"], dtype=np.float64),
        "buoy_queue": queue,
        "buoy_capacity": np.full(env.config.num_buoys, capacity, dtype=np.float64),
        "buoy_channel": np.asarray(info["rate_if_assigned"], dtype=np.float64),
        "edge_uav_uav": np.asarray(info["interference_uav_uav"], dtype=np.float64),
        "edge_uav_buoy_distance": np.asarray(info["distance_uav_buoy"], dtype=np.float64),
        "edge_uav_buoy_eta": np.asarray(info["eta_slots"], dtype=np.float64),
        "urgency": shared_task_urgency(queue),
    }


def candidate_features(obs: dict[str, np.ndarray], uav: int, buoy: int) -> dict[str, float]:
    distance = float(obs["edge_uav_buoy_distance"][uav, buoy])
    eta = float(obs["edge_uav_buoy_eta"][uav, buoy])
    interference = float(np.sum(obs["edge_uav_uav"][uav]))
    return {
        "queue": float(obs["buoy_queue"][buoy]),
        "channel": float(obs["buoy_channel"][uav, buoy]),
        "distance": distance,
        "ETA": eta,
        "interference": interference,
        "urgency": float(obs["urgency"][buoy]),
        "travel_cost": distance / max(eta, 1.0e-8),
        "interference_cost": interference,
        "energy_cost": 0.0,
    }


def assignment_similarity(features: dict[str, float]) -> float:
    """Current exclusive-match score. Queue, rate, and distance only."""
    return float(features["queue"]) * float(features["channel"]) / (float(features["distance"]) + 1.0)


__all__ = ["assignment_similarity", "candidate_features", "current_observation", "energy_cost"]
