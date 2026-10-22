"""Controller-internal low-level rewards. Does not change L2-A environment reward."""
from __future__ import annotations

import numpy as np

from wt.l2a.env import L2AEnv


def mean_target_distance_slots(env: L2AEnv, assignment: np.ndarray) -> float:
    lengths = []
    step = float(env.config.step_length_m)
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0:
            continue
        dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
        lengths.append(dist / max(step, 1.0e-9))
    return float(np.mean(lengths)) if lengths else 0.0


def low_level_reward(r_comm: float, completion: float, dist_slots: float, lambda_c: float, lambda_d: float) -> float:
    """r_low = r_comm + λ_c * in-range completion − λ_d * mean distance (in movement slots)."""
    return float(r_comm) + float(lambda_c) * float(completion) - float(lambda_d) * float(dist_slots)


__all__ = ["low_level_reward", "mean_target_distance_slots"]
