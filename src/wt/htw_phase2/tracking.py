"""Waypoint tracking terms for the Phase-2 low-level controller.

Distances use the same normalized XY frame as TW_new. Environment reward is
never edited here.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htwc_phase1.adapter import CONDITIONED_UAV_DIM
from wt.htwc_phase1.contract import GOAL_DIM, WAYPOINT_DIM
from wt.htwc_phase1.relevance import cosine
from wt.round30.features import MAX_UAVS, UAV_DIM

WAYPOINT_START = UAV_DIM + GOAL_DIM
WAYPOINT_END = WAYPOINT_START + WAYPOINT_DIM
DEFAULT_TRACKING_LAMBDA = 1.0


def normalize_positions_m(positions_m: np.ndarray, config: Any) -> np.ndarray:
    xy = np.zeros((MAX_UAVS, 2), dtype=np.float64)
    raw = np.asarray(positions_m, dtype=np.float64)
    xy[: raw.shape[0], 0] = 2.0 * raw[:, 0] / float(config.area_width) - 1.0
    xy[: raw.shape[0], 1] = 2.0 * raw[:, 1] / float(config.area_height) - 1.0
    return xy


def waypoint_from_uav_features(uav: np.ndarray) -> np.ndarray:
    extra = np.asarray(uav, dtype=np.float64)
    if extra.ndim != 2 or extra.shape[-1] != CONDITIONED_UAV_DIM:
        raise ValueError("low-level TW features must have the Phase-1 conditioned UAV width")
    return extra[:, WAYPOINT_START:WAYPOINT_END].copy()


def tracking_distance(positions_xy: np.ndarray, waypoint_xy: np.ndarray, uav_mask: np.ndarray) -> float:
    mask = np.asarray(uav_mask, dtype=bool)
    if not np.any(mask):
        return 0.0
    delta = np.asarray(positions_xy, dtype=np.float64) - np.asarray(waypoint_xy, dtype=np.float64)
    distances = np.linalg.norm(delta[mask], axis=-1)
    return float(np.mean(distances))


def auxiliary_reward(comm_reward: float, distance: float, tracking_lambda: float) -> float:
    return float(comm_reward) - float(tracking_lambda) * float(distance)


def action_waypoint_cosine(action_xy: np.ndarray, position_xy: np.ndarray, waypoint_xy: np.ndarray) -> float:
    return cosine(action_xy, np.asarray(waypoint_xy, dtype=np.float64) - np.asarray(position_xy, dtype=np.float64))


__all__ = [
    "DEFAULT_TRACKING_LAMBDA",
    "WAYPOINT_END",
    "WAYPOINT_START",
    "action_waypoint_cosine",
    "auxiliary_reward",
    "normalize_positions_m",
    "tracking_distance",
    "waypoint_from_uav_features",
]
