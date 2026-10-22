"""Build TW_new labels from existing planner rollouts.

Uses Round-19 planner decisions and the frozen environment step. It does not
add physics, change matching, or decode protected JSONL archives.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round19.planner import independent_decision, independent_queue_rate_action, joint_plan
from wt.round30.features import MAX_BUOYS, MAX_UAVS

from .contract import DEFAULT_HORIZON, TaskWaypointV2, validate_tw

PLANNER_POLICIES = ("independent_queue_rate", "joint_h5")


def _pad_goal(raw: np.ndarray, k_count: int) -> np.ndarray:
    goal = np.zeros((MAX_UAVS, MAX_BUOYS), dtype=np.float64)
    goal[: raw.shape[0], :k_count] = raw
    return goal


def _normalize_goal(service: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    totals = service.sum(axis=1)
    valid = totals > 1.0e-12
    goal = np.zeros_like(service, dtype=np.float64)
    for index, total in enumerate(totals):
        if valid[index]:
            goal[index] = service[index] / total
    return goal, valid


def _normalize_positions(positions_m: np.ndarray, config: Any) -> np.ndarray:
    xy = np.asarray(positions_m, dtype=np.float64)
    scaled = np.zeros((MAX_UAVS, 2), dtype=np.float32)
    scaled[: xy.shape[0], 0] = (2.0 * xy[:, 0] / float(config.area_width) - 1.0).astype(np.float32)
    scaled[: xy.shape[0], 1] = (2.0 * xy[:, 1] / float(config.area_height) - 1.0).astype(np.float32)
    return scaled


def labels_from_planner_window(
    service_by_uav_buoy: np.ndarray,
    start_positions_m: np.ndarray,
    end_positions_m: np.ndarray,
    *,
    config: Any,
    horizon: int = DEFAULT_HORIZON,
    uav_mask: np.ndarray | None = None,
    buoy_mask: np.ndarray | None = None,
) -> TaskWaypointV2:
    """Convert an H-step planner window into (G, W, H). W is the H-step subgoal."""
    service = np.asarray(service_by_uav_buoy, dtype=np.float64)
    if service.ndim != 3:
        raise ValueError("service_by_uav_buoy must have shape [H,M,K]")
    if int(service.shape[0]) != int(horizon):
        raise ValueError("service window length must equal the declared horizon")
    accumulated = service.sum(axis=0)
    m_count, k_count = accumulated.shape
    goal_raw, valid = _normalize_goal(accumulated)
    goal = _pad_goal(goal_raw, k_count)
    goal_valid = np.zeros(MAX_UAVS, dtype=bool)
    goal_valid[:m_count] = valid
    uav = np.zeros(MAX_UAVS, dtype=bool) if uav_mask is None else np.asarray(uav_mask, dtype=bool)
    if uav_mask is None:
        uav[:m_count] = True
    buoy = np.zeros(MAX_BUOYS, dtype=bool) if buoy_mask is None else np.asarray(buoy_mask, dtype=bool)
    if buoy_mask is None:
        buoy[:k_count] = True
    del start_positions_m
    waypoint = _normalize_positions(end_positions_m, config)
    return validate_tw(TaskWaypointV2(goal.astype(np.float32), waypoint, int(horizon), goal_valid, uav, buoy))


def _planner_action(env: SharedSpectrumMultiUAVEnv, info: Mapping[str, Any], policy: str, horizon: int) -> np.ndarray:
    if policy == "independent_queue_rate":
        return independent_queue_rate_action(info["uav_positions_m"], info["raw_queue_bits"], info["buoy_positions_m"], env.config)
    if policy == "joint_h5":
        return joint_plan(env, info, horizon=int(horizon)).action
    raise ValueError(f"unsupported planner policy: {policy}")


def _service_row(info: Mapping[str, Any]) -> np.ndarray:
    assignment = np.asarray(info["uav_to_buoy_assignment"], dtype=np.int64)
    payload = np.asarray(info["transmitted_payload_bits_by_uav"], dtype=np.float64)
    m_count, k_count = int(info["num_uavs"]), int(info["num_buoys"])
    service = np.zeros((m_count, k_count), dtype=np.float64)
    for uav, buoy in enumerate(assignment):
        if buoy >= 0:
            service[uav, int(buoy)] = float(payload[uav])
    return service


def collect_planner_window(
    env: SharedSpectrumMultiUAVEnv,
    info: Mapping[str, Any],
    *,
    horizon: int = DEFAULT_HORIZON,
    policy: str = "independent_queue_rate",
) -> tuple[TaskWaypointV2, dict[str, Any]]:
    """Roll the frozen planner for H steps and emit TW_new labels."""
    if policy not in PLANNER_POLICIES:
        raise ValueError("planner policy must be independent_queue_rate or joint_h5")
    start = np.asarray(info["uav_positions_m"], dtype=np.float64).copy()
    services = []
    current = dict(info)
    last_info = current
    for _ in range(int(horizon)):
        action = _planner_action(env, current, policy, horizon)
        _observation, _reward, terminated, truncated, last_info = env.step(action)
        services.append(_service_row(last_info))
        current = last_info
        if terminated or truncated:
            break
    if len(services) != int(horizon):
        raise RuntimeError("planner window ended before the requested horizon")
    sample = labels_from_planner_window(
        np.stack(services, axis=0),
        start,
        last_info["uav_positions_m"],
        config=env.config,
        horizon=int(horizon),
    )
    return sample, last_info


def independent_decision_available() -> bool:
    return callable(independent_decision)


__all__ = [
    "PLANNER_POLICIES",
    "collect_planner_window",
    "independent_decision_available",
    "labels_from_planner_window",
]
