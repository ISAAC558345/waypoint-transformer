"""PPO input adapter: concatenate TW_new onto frozen Round-30 feature keys.

PPO GAE, clip, value/entropy coefficients, and optimizer stay in round30/ppo.py.
This module only expands the `uav` feature matrix.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from wt.round30.features import MAX_UAVS, UAV_DIM, feature_dict, stack_feature_dicts

from .contract import GOAL_DIM, TW_UAV_EXTRA, WAYPOINT_DIM, horizon_embedding, validate_tw

CONDITIONED_UAV_DIM = UAV_DIM + TW_UAV_EXTRA


def attach_task_waypoint(
    features: Mapping[str, np.ndarray],
    tw: Mapping[str, np.ndarray] | object,
    *,
    horizon: int | None = None,
    include_goal: bool = True,
    include_waypoint: bool = True,
) -> dict[str, np.ndarray]:
    """Return a copy of Round-30 features with per-UAV TW_new concatenated."""
    sample = validate_tw(tw)
    used_horizon = int(horizon if horizon is not None else sample.horizon)
    extra = np.zeros((MAX_UAVS, TW_UAV_EXTRA), dtype=np.float32)
    if include_goal:
        extra[:, :GOAL_DIM] = sample.goal_distribution
    if include_waypoint:
        extra[:, GOAL_DIM:GOAL_DIM + WAYPOINT_DIM] = sample.waypoint_xy
    extra[:, GOAL_DIM + WAYPOINT_DIM :] = horizon_embedding(used_horizon)[None, :]
    extra *= sample.uav_mask[:, None].astype(np.float32)
    uav = np.asarray(features["uav"], dtype=np.float32)
    if uav.ndim == 2:
        if uav.shape[-1] != UAV_DIM:
            raise ValueError("adapter expects the original Round-30 UAV width")
        conditioned = np.concatenate((uav, extra), axis=-1)
        output = {key: np.asarray(value).copy() for key, value in features.items()}
        output["uav"] = conditioned
        return output
    if uav.ndim != 3 or uav.shape[-1] != UAV_DIM:
        raise ValueError("batched adapter expects [B,M,UAV_DIM]")
    batch = uav.shape[0]
    extra_b = np.broadcast_to(extra[None, ...], (batch, MAX_UAVS, TW_UAV_EXTRA)).copy()
    output = {key: np.asarray(value).copy() for key, value in features.items()}
    output["uav"] = np.concatenate((uav, extra_b), axis=-1)
    return output


def state_only_features(info, config) -> dict[str, np.ndarray]:
    return feature_dict(info, config)


def stacked_state_only(items) -> dict[str, np.ndarray]:
    return stack_feature_dicts(items)


__all__ = [
    "CONDITIONED_UAV_DIM",
    "attach_task_waypoint",
    "stacked_state_only",
    "state_only_features",
]
