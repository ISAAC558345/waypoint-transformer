"""Single-factor potential shaping. The bit reward is not added a second time."""
from __future__ import annotations

import numpy as np

from wt.lowlevel_v2.phase1.protocol import PPO


def distance_potential(env, association: np.ndarray) -> float:
    """Negative mean excess distance, in environment steps. Inside the radius this is 0.

    The value uses only the current positions and the current assignment.
    """
    radius = float(env.config.service_radius_m)
    step = float(env.config.step_length_m)
    excess = []
    for uav, buoy in enumerate(np.asarray(association, dtype=np.int64)):
        if int(buoy) < 0:
            continue
        dist = float(np.linalg.norm(env.buoy_positions[int(buoy)] - env.uav_positions[uav]))
        excess.append(max(dist - radius, 0.0) / step)
    if not excess:
        return 0.0
    return -float(np.mean(excess))


def shaping_reward(phi: float, phi_next: float, done: bool, gamma: float = PPO.gamma) -> float:
    """Ng shaping. A terminal next state has potential 0."""
    nxt = 0.0 if bool(done) else float(phi_next)
    value = float(gamma) * nxt - float(phi)
    if not np.isfinite(value):
        raise RuntimeError("NONFINITE")
    return float(value)


__all__ = ["distance_potential", "shaping_reward"]
