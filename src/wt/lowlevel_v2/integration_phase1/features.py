"""Assignment geometry descriptors for executor-shift correlation. Descriptive only."""
from __future__ import annotations

import numpy as np

from wt.l3a.env import competition_pairs
from wt.lowlevel_v2.contracts import minimum_pairwise_distance


def candidate_features(env, assignment: np.ndarray, comm: dict) -> dict:
    chosen = np.asarray(assignment, dtype=np.int64)
    dists = []
    for uav, buoy in enumerate(chosen):
        if int(buoy) >= 0:
            dists.append(float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)])))
    d_uu = []
    for i in range(env.config.num_uavs):
        for j in range(i + 1, env.config.num_uavs):
            d_uu.append(float(np.linalg.norm(env.uav_positions[i] - env.uav_positions[j])))
    pairs = competition_pairs(chosen, env.config.num_buoys)
    cap = max(float(env.config.queue_capacity_blocks), 1.0e-9)
    return {
        "mean_uav_target_dist": float(np.mean(dists)) if dists else float("nan"),
        "max_uav_target_dist": float(np.max(dists)) if dists else float("nan"),
        "queue_pressure": float(np.mean(env.queue / cap)),
        "idle_count": float(np.sum(chosen < 0)),
        "current_mean_sinr": float(np.mean(comm["sinr"])),
        "current_interference": float(np.mean(comm["total_interference"])),
        "uav_pairwise_proximity": float(np.mean(d_uu)) if d_uu else float("nan"),
        "min_pairwise_dist": float(minimum_pairwise_distance(env.uav_positions)),
        "competition_indicator": float(len(pairs)),
    }


__all__ = ["candidate_features"]
