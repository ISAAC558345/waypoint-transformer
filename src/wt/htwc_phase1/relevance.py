"""Action-relevance and waypoint-quality diagnostics. No environment changes."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from wt.round14.guidance import map_task_waypoint_to_motion
from wt.round30.features import MAX_BUOYS, MAX_UAVS

EPSILON = 1.0e-8


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= EPSILON:
        return float("nan")
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def new_tw_direction(waypoint_xy: np.ndarray, current_xy: np.ndarray) -> np.ndarray:
    return np.asarray(waypoint_xy, dtype=np.float64) - np.asarray(current_xy, dtype=np.float64)


def old_tw_direction(phi: np.ndarray, info: Mapping[str, Any], uav_index: int) -> np.ndarray:
    k_count = int(info["num_buoys"])
    mask = np.zeros(MAX_BUOYS, dtype=bool)
    mask[:k_count] = True
    buoys = np.zeros((MAX_BUOYS, 2), dtype=np.float64)
    buoys[:k_count] = np.asarray(info["buoy_positions_m"], dtype=np.float64)
    rates = np.zeros(MAX_BUOYS, dtype=np.float64)
    rates[:k_count] = np.asarray(info["rates_bps"], dtype=np.float64)[uav_index]
    mapped = map_task_waypoint_to_motion(
        "C_service_progress_rate",
        np.asarray(phi[uav_index], dtype=np.float64),
        mask,
        np.asarray(info["uav_positions_m"][uav_index], dtype=np.float64),
        buoys,
        rates,
    )
    return np.asarray(mapped.direction_xy, dtype=np.float64)


def summarize_cosines(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"count": 0.0, "mean_rho": float("nan"), "median_rho": float("nan")}
    array = np.asarray(finite, dtype=np.float64)
    return {"count": float(len(array)), "mean_rho": float(array.mean()), "median_rho": float(np.median(array))}


__all__ = ["MAX_UAVS", "cosine", "new_tw_direction", "old_tw_direction", "summarize_cosines"]
