"""TW_new contract: goal, subgoal waypoint, and fixed horizon.

This module does not import HGNN, graph edges, or environment physics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from wt.round30.features import MAX_BUOYS, MAX_UAVS

DEFAULT_HORIZON = 5
MAX_HORIZON = 20
GOAL_DIM = MAX_BUOYS
WAYPOINT_DIM = 2
HORIZON_EMBED_DIM = 1
TW_UAV_EXTRA = GOAL_DIM + WAYPOINT_DIM + HORIZON_EMBED_DIM


@dataclass(frozen=True)
class TaskWaypointV2:
    """Per-state TW_new target or prediction."""

    goal_distribution: np.ndarray
    waypoint_xy: np.ndarray
    horizon: int
    goal_valid_uav_mask: np.ndarray
    uav_mask: np.ndarray
    buoy_mask: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal_distribution": np.asarray(self.goal_distribution, dtype=np.float32),
            "waypoint_xy": np.asarray(self.waypoint_xy, dtype=np.float32),
            "horizon": int(self.horizon),
            "goal_valid_uav_mask": np.asarray(self.goal_valid_uav_mask, dtype=bool),
            "uav_mask": np.asarray(self.uav_mask, dtype=bool),
            "buoy_mask": np.asarray(self.buoy_mask, dtype=bool),
        }


def validate_tw(sample: Mapping[str, Any] | TaskWaypointV2) -> TaskWaypointV2:
    row = sample.as_dict() if isinstance(sample, TaskWaypointV2) else dict(sample)
    goal = np.asarray(row["goal_distribution"], dtype=np.float64)
    waypoint = np.asarray(row["waypoint_xy"], dtype=np.float64)
    horizon = int(row["horizon"])
    goal_valid = np.asarray(row["goal_valid_uav_mask"], dtype=bool)
    uav_mask = np.asarray(row["uav_mask"], dtype=bool)
    buoy_mask = np.asarray(row.get("buoy_mask", np.ones(MAX_BUOYS, dtype=bool)), dtype=bool)
    if goal.shape != (MAX_UAVS, GOAL_DIM) or waypoint.shape != (MAX_UAVS, WAYPOINT_DIM):
        raise ValueError("TW_new goal must be [M,K_max] and waypoint [M,2] with Round-30 padding")
    if horizon < 1 or horizon > MAX_HORIZON:
        raise ValueError("horizon must be a positive planning scale")
    if not np.all(np.isfinite(goal)) or not np.all(np.isfinite(waypoint)):
        raise ValueError("TW_new tensors must be finite")
    if np.any(goal < -1.0e-8):
        raise ValueError("goal probabilities must be nonnegative")
    row_sum = goal.sum(axis=-1)
    for index in range(MAX_UAVS):
        if not uav_mask[index]:
            continue
        if goal_valid[index] and abs(float(row_sum[index]) - 1.0) > 1.0e-4:
            raise ValueError("valid goal rows must be a softmax distribution")
        if (not goal_valid[index]) and float(row_sum[index]) > 1.0e-6:
            raise ValueError("idle UAV goal rows must be zero")
    return TaskWaypointV2(goal.astype(np.float32), waypoint.astype(np.float32), horizon, goal_valid, uav_mask, buoy_mask)


def horizon_embedding(horizon: int) -> np.ndarray:
    return np.asarray((float(horizon) / float(MAX_HORIZON),), dtype=np.float32)


__all__ = [
    "DEFAULT_HORIZON",
    "GOAL_DIM",
    "HORIZON_EMBED_DIM",
    "MAX_HORIZON",
    "TaskWaypointV2",
    "TW_UAV_EXTRA",
    "WAYPOINT_DIM",
    "horizon_embedding",
    "validate_tw",
]
