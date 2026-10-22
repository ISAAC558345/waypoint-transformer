"""Legal myopic resolution: independent greedy then exclusive conflict break."""
from __future__ import annotations

import numpy as np

from wt.l3b.env import L3BEnv, conflict_mask
from wt.l3b.policies import exclusive_matching, myopic_assignment


def legal_myopic_assignment(env: L3BEnv) -> np.ndarray:
    """Independent Q/distance greedy, then keep the best UAV per buoy and idle the rest.

    The raw L3-B myopic_assignment allows collisions. This legal version does not
    change scores; it only drops conflicting extra claims. UAV scan order is not
    used as a sequential reservation (that is exclusive_matching).
    """
    requested = np.asarray(myopic_assignment(env), dtype=np.int64)
    scores = np.full(env.config.num_uavs, -1.0e18, dtype=np.float64)
    for uav, buoy in enumerate(requested):
        if int(buoy) < 0:
            continue
        dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
        scores[uav] = float(env.queue[int(buoy)]) / (dist + 1.0)
    legal = requested.copy()
    for buoy in range(int(env.config.num_buoys)):
        owners = np.flatnonzero(legal == buoy)
        if owners.size <= 1:
            continue
        keep = int(owners[int(np.argmax(scores[owners]))])
        for uav in owners:
            if int(uav) != keep:
                legal[int(uav)] = -1
    if bool(np.any(conflict_mask(legal, env.config.num_buoys))):
        raise RuntimeError("legal myopic still contains a collision")
    return legal


def exclusive_assignment(env: L3BEnv) -> np.ndarray:
    chosen = np.asarray(exclusive_matching(env), dtype=np.int64)
    if bool(np.any(conflict_mask(chosen, env.config.num_buoys))):
        raise RuntimeError("exclusive matching produced a collision")
    return chosen


__all__ = ["exclusive_assignment", "legal_myopic_assignment"]
