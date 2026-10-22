"""UAV–Buoy heterogeneous graph features for high-level HGNN. Does not change L2-A physics."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from wt.htw_phase61.features import MAX_BUOYS, MAX_UAVS
from wt.l2a.env import L2AEnv

EDGE_DIM = 4


def build_edge_features(info: Mapping[str, Any], env: L2AEnv) -> np.ndarray:
    positions = np.asarray(info["uav_positions_m"], dtype=np.float64)
    buoys = np.asarray(info["buoy_positions_m"], dtype=np.float64)
    eta = np.asarray(info["eta_slots"], dtype=np.float64)
    config = env.config
    m_count, k_count = int(config.num_uavs), int(config.num_buoys)
    edge = np.zeros((MAX_UAVS, MAX_BUOYS, EDGE_DIM), dtype=np.float32)
    diag = float(np.hypot(config.area_width, config.area_height))
    for uav in range(m_count):
        others = 0.0
        for other in range(m_count):
            if other == uav:
                continue
            if float(np.linalg.norm(positions[uav] - positions[other])) <= float(config.interference_radius_m):
                others += 1.0
        interference = others / max(float(m_count - 1), 1.0)
        for buoy in range(k_count):
            dist = float(np.linalg.norm(positions[uav] - buoys[buoy]))
            travel = float(eta[uav, buoy])
            rate = float(config.service_radius_m) / max(dist, 1.0e-3)
            rate = float(np.clip(rate, 0.0, 1.0))
            edge[uav, buoy] = (
                dist / max(diag, 1.0e-9),
                travel / max(float(config.episode_horizon), 1.0),
                rate,
                interference,
            )
    return edge


__all__ = ["EDGE_DIM", "build_edge_features"]
