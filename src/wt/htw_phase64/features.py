"""High-level DeepSets features for Phase-6.4. No HGNN; arrival uses persistence, not remaining_arrivals()."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from wt.htw_phase61.features import BUOY_DIM, GLOBAL_DIM, MAX_BUOYS, MAX_UAVS, UAV_DIM, feature_dict_l2a, stack_feature_dicts
from wt.l2a.env import L2AEnv

ENHANCED_BUOY_DIM = 12


def feature_dict_high(
    info: Mapping[str, Any],
    env: L2AEnv,
    *,
    last_arrival: np.ndarray | None = None,
    enhanced: bool = False,
) -> dict[str, np.ndarray]:
    features = feature_dict_l2a(info, env)
    if not enhanced:
        return features
    config = env.config
    cap = float(config.queue_capacity_blocks)
    horizon = float(config.episode_horizon)
    eta = np.asarray(info["eta_slots"], dtype=np.float64)
    min_eta = np.min(eta, axis=0)
    progress = np.asarray(info["service_progress"], dtype=np.float32)
    assignment = np.asarray(info["assignment"], dtype=np.int64)
    progress_on_buoy = np.zeros(config.num_buoys, dtype=np.float32)
    for uav, buoy in enumerate(assignment):
        if int(buoy) >= 0:
            progress_on_buoy[int(buoy)] = max(float(progress_on_buoy[int(buoy)]), float(progress[uav]))
    observed = np.zeros(config.num_buoys, dtype=np.float32) if last_arrival is None else np.asarray(last_arrival, dtype=np.float32)
    predicted = observed.copy()
    extra = np.zeros((MAX_BUOYS, ENHANCED_BUOY_DIM - BUOY_DIM), dtype=np.float32)
    extra[: config.num_buoys] = np.column_stack(
        (
            np.clip(min_eta / max(horizon, 1.0), 0.0, 1.0),
            np.clip(min_eta * float(config.step_length_m) / max(float(config.area_width), 1.0), 0.0, 1.0),
            progress_on_buoy,
            np.clip(observed / cap, 0.0, 1.0),
            np.clip(predicted / cap, 0.0, 1.0),
        )
    )
    features["buoy"] = np.concatenate((features["buoy"], extra), axis=-1)
    return features


__all__ = ["ENHANCED_BUOY_DIM", "GLOBAL_DIM", "UAV_DIM", "feature_dict_high", "stack_feature_dicts"]
