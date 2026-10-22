"""Phase-9 features and L3-B graph edges. Does not invent energy; energy channel is zero."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from wt.l3a.env import conflict_mask
from wt.l3b.env import L3BEnv

from .const import BUOY_DIM, EDGE_UB_DIM, EDGE_UU_DIM, GLOBAL_DIM, MAX_BUOYS, MAX_UAVS, UAV_DIM


def feature_dict_l3b(
    info: Mapping[str, Any],
    env: L3BEnv,
    *,
    last_arrival: np.ndarray | None = None,
    last_positions: np.ndarray | None = None,
    uav_keep: int | None = None,
    buoy_keep: int | None = None,
) -> dict[str, np.ndarray]:
    config = env.config
    positions = np.asarray(info["uav_positions_m"], dtype=np.float32)
    buoys = np.asarray(info["buoy_positions_m"], dtype=np.float32)
    queue = np.asarray(info["queue_blocks"], dtype=np.float32)
    progress = np.asarray(info["service_progress"], dtype=np.float32)
    assignment = np.asarray(info["assignment"], dtype=np.int64)
    cap = float(config.queue_capacity_blocks)
    if last_positions is None:
        velocity = np.zeros_like(positions)
    else:
        velocity = (positions - np.asarray(last_positions, dtype=np.float32)) / max(float(config.step_length_m), 1.0e-9)
        velocity = np.clip(velocity, -1.0, 1.0)
    observed = np.zeros(config.num_buoys, dtype=np.float32) if last_arrival is None else np.asarray(last_arrival, dtype=np.float32)
    assigned_queue = np.zeros(config.num_uavs, dtype=np.float32)
    for uav, buoy in enumerate(assignment):
        if int(buoy) >= 0:
            assigned_queue[uav] = queue[int(buoy)] / cap
    energy = np.zeros(config.num_uavs, dtype=np.float32)
    uav = np.zeros((MAX_UAVS, UAV_DIM), dtype=np.float32)
    uav[: config.num_uavs] = np.column_stack(
        (
            2.0 * positions[:, 0] / float(config.area_width) - 1.0,
            2.0 * positions[:, 1] / float(config.area_height) - 1.0,
            velocity[:, 0],
            velocity[:, 1],
            progress,
            (assignment >= 0).astype(np.float32),
            assigned_queue,
            energy,
        )
    )
    occupied = np.zeros(config.num_buoys, dtype=np.float32)
    progress_on_buoy = np.zeros(config.num_buoys, dtype=np.float32)
    for uav_id, buoy_id in enumerate(assignment):
        if int(buoy_id) >= 0:
            occupied[int(buoy_id)] = 1.0
            progress_on_buoy[int(buoy_id)] = max(float(progress_on_buoy[int(buoy_id)]), float(progress[uav_id]))
    buoy = np.zeros((MAX_BUOYS, BUOY_DIM), dtype=np.float32)
    buoy[: config.num_buoys] = np.column_stack(
        (
            2.0 * buoys[:, 0] / float(config.area_width) - 1.0,
            2.0 * buoys[:, 1] / float(config.area_height) - 1.0,
            np.clip(queue / cap, 0.0, 1.0),
            np.clip(observed / cap, 0.0, 1.0),
            progress_on_buoy,
            occupied,
            np.full(config.num_buoys, float(config.num_buoys) / MAX_BUOYS, dtype=np.float32),
        )
    )
    uav_mask = np.zeros(MAX_UAVS, dtype=bool)
    buoy_mask = np.zeros(MAX_BUOYS, dtype=bool)
    keep_u = int(config.num_uavs if uav_keep is None else uav_keep)
    keep_k = int(config.num_buoys if buoy_keep is None else buoy_keep)
    uav_mask[:keep_u] = True
    buoy_mask[:keep_k] = True
    global_ = np.asarray(
        (
            float(info["time_step"]) / float(config.episode_horizon),
            float(np.sum(queue)) / (cap * float(config.num_buoys)),
            float(keep_u) / MAX_UAVS,
        ),
        dtype=np.float32,
    )
    return {"uav": uav, "buoy": buoy, "uav_mask": uav_mask, "buoy_mask": buoy_mask, "global": global_}


def build_graph(info: Mapping[str, Any], env: L3BEnv) -> tuple[np.ndarray, np.ndarray]:
    config = env.config
    diag = float(np.hypot(config.area_width, config.area_height))
    distance_ub = np.asarray(info["distance_uav_buoy"], dtype=np.float64)
    eta = np.asarray(info["eta_slots"], dtype=np.float64)
    rate = np.asarray(info["rate_if_assigned"], dtype=np.float64)
    distance_uu = np.asarray(info["distance_uav_uav"], dtype=np.float64)
    interference = np.asarray(info["interference_uav_uav"], dtype=np.float64)
    ref = max(float(env.reference_rate_bps), 1.0e-9)
    edge_ub = np.zeros((MAX_UAVS, MAX_BUOYS, EDGE_UB_DIM), dtype=np.float32)
    edge_uu = np.zeros((MAX_UAVS, MAX_UAVS, EDGE_UU_DIM), dtype=np.float32)
    m_count, k_count = int(config.num_uavs), int(config.num_buoys)
    for uav in range(m_count):
        for buoy in range(k_count):
            edge_ub[uav, buoy] = (
                float(distance_ub[uav, buoy]) / max(diag, 1.0e-9),
                float(np.clip(rate[uav, buoy] / ref, 0.0, 2.0) / 2.0),
                float(eta[uav, buoy]) / max(float(config.episode_horizon), 1.0),
            )
        for other in range(m_count):
            if other == uav:
                continue
            edge_uu[uav, other] = (
                float(distance_uu[uav, other]) / max(diag, 1.0e-9),
                float(np.clip(interference[uav, other] / max(float(config.tx_power), 1.0e-9), 0.0, 1.0)),
            )
    return edge_ub, edge_uu


def stack_feature_dicts(items: Iterable[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    rows = list(items)
    keys = ("uav", "buoy", "uav_mask", "buoy_mask", "global")
    return {key: np.stack([np.asarray(row[key]) for row in rows], axis=0) for key in keys}


def in_range_completion(env: L3BEnv, assignment: np.ndarray) -> float:
    hits = []
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0:
            hits.append(1.0)
            continue
        hits.append(1.0 if env._in_range(uav, int(buoy)) else 0.0)
    return float(np.mean(hits)) if hits else 0.0


def service_success(info: Mapping[str, Any], assignment: np.ndarray) -> float:
    service = np.asarray(info.get("step_service_blocks", np.zeros(MAX_BUOYS)), dtype=np.float64)
    hits = []
    for buoy in np.asarray(assignment, dtype=np.int64):
        if int(buoy) < 0:
            continue
        hits.append(1.0 if float(service[int(buoy)]) > 0.0 else 0.0)
    return float(np.mean(hits)) if hits else 0.0


def conflict_rate(assignment: np.ndarray, num_buoys: int) -> float:
    return float(np.mean(conflict_mask(assignment, num_buoys)))


def interference_utility(info: Mapping[str, Any]) -> float:
    rate = np.asarray(info.get("rate_bps", np.zeros(MAX_UAVS)), dtype=np.float64)
    return float(np.mean(rate))


__all__ = [
    "build_graph",
    "conflict_rate",
    "feature_dict_l3b",
    "in_range_completion",
    "interference_utility",
    "service_success",
    "stack_feature_dicts",
]
