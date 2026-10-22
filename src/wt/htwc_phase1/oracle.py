"""Oracle TW_new from planner shadow windows. Does not mutate the live env."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from wt.round18.multi_env import expected_arrival_bits
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv, shared_shadow_step
from wt.round19.planner import independent_queue_rate_action
from wt.round30.features import MAX_UAVS, feature_dict

from .adapter import attach_task_waypoint
from .contract import DEFAULT_HORIZON, TaskWaypointV2
from .labels import labels_from_planner_window


def _service_from_step(step: Any) -> np.ndarray:
    assignment = np.asarray(step.assignment, dtype=np.int64)
    payload = np.asarray(step.payload_by_uav_bits, dtype=np.float64)
    m_count = int(payload.shape[0])
    k_count = int(np.asarray(step.queue_bits).shape[0])
    service = np.zeros((m_count, k_count), dtype=np.float64)
    for uav, buoy in enumerate(assignment):
        if buoy >= 0:
            service[uav, int(buoy)] = float(payload[uav])
    return service


def oracle_planner_window(
    env: SharedSpectrumMultiUAVEnv,
    info: Mapping[str, Any],
    *,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[TaskWaypointV2, np.ndarray]:
    """Independent queue-rate planner H-step shadow; W is the H-step position."""
    positions = np.asarray(info["uav_positions_m"], dtype=np.float64).copy()
    energy = np.asarray(info["remaining_energy_per_uav"], dtype=np.float64).copy()
    queues = np.asarray(info["raw_queue_bits"], dtype=np.float64).copy()
    buoys = np.asarray(info["buoy_positions_m"], dtype=np.float64)
    path = [positions.copy()]
    services = []
    for _ in range(int(horizon)):
        action = independent_queue_rate_action(positions, queues, buoys, env.config)
        step = shared_shadow_step(
            positions_m=positions,
            remaining_energy=energy,
            queue_bits=queues,
            buoy_positions_m=buoys,
            action=action,
            arrival_bits=expected_arrival_bits(env.config),
            config=env.config,
            options=env.options,
        )
        positions = np.asarray(step.positions_m, dtype=np.float64).copy()
        energy = np.asarray(step.remaining_energy, dtype=np.float64).copy()
        queues = np.asarray(step.queue_bits, dtype=np.float64).copy()
        path.append(positions.copy())
        services.append(_service_from_step(step))
    label = labels_from_planner_window(np.stack(services, axis=0), path[0], path[-1], config=env.config, horizon=int(horizon))
    return label, np.stack(path, axis=0)


def oracle_conditioner(info: Mapping[str, Any], env: SharedSpectrumMultiUAVEnv, *, horizon: int = DEFAULT_HORIZON, include_goal: bool = True, include_waypoint: bool = True):
    features = feature_dict(info, env.config)
    label, _path = oracle_planner_window(env, info, horizon=horizon)
    attached = attach_task_waypoint(features, label, horizon=horizon, include_goal=include_goal, include_waypoint=include_waypoint)
    attached["_oracle_waypoint_xy"] = label.waypoint_xy.copy()
    attached["_oracle_path_m"] = _path
    return attached


def tw_enters_actor(features: Mapping[str, np.ndarray]) -> dict[str, bool]:
    extra = np.asarray(features["uav"])[..., 8:]
    return {
        "uav_width_is_conditioned": int(np.asarray(features["uav"]).shape[-1]) == 16,
        "goal_nonzero": bool(np.any(np.abs(extra[..., :5]) > 1.0e-8)),
        "waypoint_nonzero": bool(np.any(np.abs(extra[..., 5:7]) > 1.0e-8)),
        "horizon_nonzero": bool(np.any(np.abs(extra[..., 7:]) > 1.0e-8)),
    }


__all__ = ["oracle_conditioner", "oracle_planner_window", "tw_enters_actor"]
