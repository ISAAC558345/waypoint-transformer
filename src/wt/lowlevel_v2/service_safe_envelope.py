"""Service-safe envelope: in-range service UAVs must stay inside a shrunk disk."""
from __future__ import annotations

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv

from .contracts import env_target
from .protocol import SERVICE_SAFETY_MARGIN_PLACEHOLDER_M


def envelope_active(env: L3BEnv, option: TaskOption, safety_margin: float | None = None) -> np.ndarray:
    _ = safety_margin
    m_count = int(env.config.num_uavs)
    k_count = int(env.config.num_buoys)
    flags = np.zeros(m_count, dtype=bool)
    for uav in range(m_count):
        tgt = env_target(int(option.association[uav]), k_count)
        if tgt < 0:
            continue
        if float(env.queue[tgt]) <= 1.0e-9:
            continue
        if not env._in_range(uav, tgt):
            continue
        flags[uav] = True
    return flags


def inner_radius(env: L3BEnv, safety_margin: float | None = None) -> float:
    margin = float(SERVICE_SAFETY_MARGIN_PLACEHOLDER_M if safety_margin is None else safety_margin)
    return float(env.config.service_radius_m) - margin


def next_in_envelope(
    env: L3BEnv,
    option: TaskOption,
    next_positions: np.ndarray,
    safety_margin: float | None = None,
) -> np.ndarray:
    flags = envelope_active(env, option, safety_margin)
    radius = inner_radius(env, safety_margin)
    ok = np.ones(int(env.config.num_uavs), dtype=bool)
    pos = np.asarray(next_positions, dtype=np.float64)
    for uav, active in enumerate(flags):
        if not active:
            continue
        tgt = env_target(int(option.association[uav]), int(env.config.num_buoys))
        dist = float(np.linalg.norm(pos[uav] - env.buoy_positions[tgt]))
        ok[uav] = dist <= radius + 1.0e-9
    return ok


__all__ = ["envelope_active", "inner_radius", "next_in_envelope"]
