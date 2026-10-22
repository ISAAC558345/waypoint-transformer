"""Distance potential. Delivered bits are not an input."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.lowlevel_v2.phase1.protocol import PPO


def distance_norm_m(env) -> float:
    """Step length in meters. Distance and the service radius use the same unit."""
    radius = float(env.config.service_radius_m)
    step = float(env.config.step_length_m)
    if not np.isfinite(radius) or not np.isfinite(step) or radius <= 0.0 or step <= 0.0:
        raise RuntimeError("NONFINITE")
    return step


def distance_potential(env, association: np.ndarray) -> float:
    """Negative mean excess distance of UAVs that currently have a target.

    Idle assignments contribute nothing. Inside the service radius the excess is 0,
    so the potential does not pay for delivered bits or for motion toward the buoy center.
    """
    radius = float(env.config.service_radius_m)
    norm = distance_norm_m(env)
    excess = []
    for uav, buoy in enumerate(np.asarray(association, dtype=np.int64)):
        if int(buoy) < 0:
            continue
        dist = float(np.linalg.norm(env.buoy_positions[int(buoy)] - env.uav_positions[uav]))
        if not np.isfinite(dist):
            raise RuntimeError("NONFINITE")
        excess.append(max(dist - radius, 0.0) / norm)
    if not excess:
        return 0.0
    return -float(np.mean(excess))


def shaping_reward(phi: float, phi_next: float, done: bool, gamma: float = PPO.gamma) -> float:
    """Ng shaping. A terminal next state uses potential 0, and gamma is the MAPPO gamma."""
    if not np.isfinite(phi) or not np.isfinite(phi_next) or not np.isfinite(gamma):
        raise RuntimeError("NONFINITE")
    nxt = 0.0 if bool(done) else float(phi_next)
    value = float(gamma) * nxt - float(phi)
    if not np.isfinite(value):
        raise RuntimeError("NONFINITE")
    return float(value)


def _place(env: L3BBitQueueEnvV2) -> np.ndarray:
    count = int(env.config.num_uavs)
    env.uav_positions[:] = np.stack([np.array([15.0 + index * 30.0, 12.0], dtype=np.float64) for index in range(count)])
    return np.full(count, -1, dtype=np.int64)


def direction_report() -> dict:
    """Deterministic signs for approach, retreat, in-radius, idle, and termination."""
    env = L3BBitQueueEnvV2()
    env.reset(seed=321000)
    radius = float(env.config.service_radius_m)
    step = distance_norm_m(env)
    buoy = env.buoy_positions[0].copy()
    association = _place(env)
    association[0] = 0
    env.uav_positions[0] = buoy + np.array([radius + step, 0.0])
    phi_far = distance_potential(env, association)
    env.uav_positions[0] = buoy + np.array([radius + 0.5 * step, 0.0])
    phi_closer = distance_potential(env, association)
    toward = shaping_reward(phi_far, phi_closer, False)
    env.uav_positions[0] = buoy + np.array([radius + 2.0 * step, 0.0])
    phi_away = distance_potential(env, association)
    away = shaping_reward(phi_far, phi_away, False)
    env.uav_positions[0] = buoy + np.array([min(4.0, 0.25 * radius), 0.0])
    phi_inside = distance_potential(env, association)
    env.uav_positions[0] = buoy + np.array([min(2.0, 0.125 * radius), 0.0])
    phi_inside_closer = distance_potential(env, association)
    inside = shaping_reward(phi_inside, phi_inside_closer, False)
    idle = _place(env)
    phi_idle = distance_potential(env, idle)
    env.uav_positions[0] = buoy + np.array([radius + 3.0 * step, 0.0])
    phi_idle_moved = distance_potential(env, idle)
    idle_motion = shaping_reward(phi_idle, phi_idle_moved, False)
    association = _place(env)
    association[0] = 0
    env.uav_positions[0] = buoy + np.array([radius + step, 0.0])
    phi_active = distance_potential(env, association)
    stayed = shaping_reward(phi_active, phi_active, False)
    env.uav_positions[int(env.config.num_uavs) - 1] = buoy + np.array([0.0, radius + 4.0 * step])
    phi_idle_shift = distance_potential(env, association)
    idle_shift = shaping_reward(phi_active, phi_idle_shift, False)
    terminal = shaping_reward(phi_far, -4.0, True)
    terminal_inside = shaping_reward(0.0, -4.0, True)
    payload_before = float(env.payload_bits)
    env.payload_bits = payload_before + 1.0e9
    phi_after_bits = distance_potential(env, association)
    env.payload_bits = payload_before
    return {
        "gamma": float(PPO.gamma),
        "distance_norm_m": float(step),
        "service_radius_m": float(radius),
        "units": "m",
        "phi_one_step": float(phi_far),
        "toward": float(toward),
        "away": float(away),
        "inside": float(inside),
        "idle_motion": float(idle_motion),
        "idle_shift_minus_stay": float(idle_shift - stayed),
        "terminal": float(terminal),
        "terminal_inside": float(terminal_inside),
        "phi_unchanged_by_payload": bool(phi_after_bits == phi_active),
        "toward_positive": bool(toward > 0.0 and abs(phi_far + 1.0) <= 1.0e-6 and phi_closer > phi_far),
        "away_negative": bool(away < 0.0 and phi_away < phi_far),
        "inside_zero": bool(abs(phi_inside) <= 1.0e-8 and abs(phi_inside_closer) <= 1.0e-8 and abs(inside) <= 1.0e-8),
        "idle_motion_adds_nothing": bool(abs(phi_idle) <= 1.0e-8 and abs(phi_idle_moved) <= 1.0e-8 and abs(idle_motion) <= 1.0e-8 and abs(idle_shift - stayed) <= 1.0e-8),
        "terminal_uses_zero_potential": bool(abs(terminal - (-phi_far)) <= 1.0e-6 and abs(terminal_inside) <= 1.0e-8),
        "bits_absent_from_potential": bool(phi_after_bits == phi_active),
    }


def directions_pass(report: dict) -> bool:
    keys = (
        "toward_positive",
        "away_negative",
        "inside_zero",
        "idle_motion_adds_nothing",
        "terminal_uses_zero_potential",
        "bits_absent_from_potential",
    )
    return all(bool(report[key]) for key in keys)


__all__ = [
    "direction_report",
    "directions_pass",
    "distance_norm_m",
    "distance_potential",
    "shaping_reward",
]
