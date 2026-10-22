"""Controller-internal auxiliary reward. Does not change L3-B env payload."""
from __future__ import annotations

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv

from .protocol import AUX_DISTANCE, AUX_IN_RANGE


def auxiliary_reward(env: L3BEnv, option: TaskOption) -> dict[str, float]:
    """Compute r_aux from the option that was just executed (after env.step)."""
    m_count = int(env.config.num_uavs)
    radius = float(env.config.service_radius_m)
    step = max(float(env.config.step_length_m), 1.0e-9)
    targets = np.asarray(option.targets, dtype=np.int64)
    in_range = 0.0
    d_out = 0.0
    for uav in range(m_count):
        tgt = int(targets[uav]) if uav < targets.shape[0] else -1
        active = tgt >= 0 and tgt < int(env.config.num_buoys)
        if not active:
            continue
        dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[tgt]))
        d_out += max(dist - radius, 0.0) / step
        if env._in_range(uav, tgt):
            in_range += 1.0
    denom = max(float(m_count), 1.0)
    r_aux = float(AUX_IN_RANGE) * in_range / denom - float(AUX_DISTANCE) * d_out / denom
    return {"r_aux": float(r_aux), "in_range_count": float(in_range), "d_out_sum": float(d_out)}


def env_reward(before_payload: float, after_payload: float, block_payload_bits: float) -> float:
    return (float(after_payload) - float(before_payload)) / max(float(block_payload_bits), 1.0e-9)


__all__ = ["auxiliary_reward", "env_reward"]
