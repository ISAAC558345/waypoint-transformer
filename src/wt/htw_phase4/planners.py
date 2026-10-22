"""Phase-4 high-level planners. TW contract unchanged; env/PPO/HGNN untouched."""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from wt.htwc_phase1.adapter import attach_task_waypoint
from wt.htwc_phase1.contract import DEFAULT_HORIZON, TaskWaypointV2
from wt.htwc_phase1.labels import labels_from_planner_window
from wt.htwc_phase1.oracle import _service_from_step
from wt.round18.multi_env import expected_arrival_bits
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv, shared_shadow_step
from wt.round19.planner import (
    BEAM_WIDTH,
    PLANNING_HORIZONS,
    _State,
    _joint_candidates,
    _next,
    _shadow,
    independent_queue_rate_action,
    joint_plan,
)
from wt.round30.features import feature_dict

ActionFn = Callable[[SharedSpectrumMultiUAVEnv, Mapping[str, Any]], np.ndarray]


def _info_with_state(info: Mapping[str, Any], positions: np.ndarray, energy: np.ndarray, queues: np.ndarray) -> dict[str, Any]:
    updated = dict(info)
    updated["uav_positions_m"] = np.asarray(positions, dtype=np.float64)
    updated["remaining_energy_per_uav"] = np.asarray(energy, dtype=np.float64)
    updated["raw_queue_bits"] = np.asarray(queues, dtype=np.float64)
    return updated


def planner_a_action(env: SharedSpectrumMultiUAVEnv, info: Mapping[str, Any]) -> np.ndarray:
    return independent_queue_rate_action(info["uav_positions_m"], info["raw_queue_bits"], info["buoy_positions_m"], env.config)


def planner_b_action(env: SharedSpectrumMultiUAVEnv, info: Mapping[str, Any]) -> np.ndarray:
    return joint_plan(env, info, horizon=1).action


def planner_c_action(env: SharedSpectrumMultiUAVEnv, info: Mapping[str, Any], *, plan_horizon: int = 5) -> np.ndarray:
    return joint_plan(env, info, horizon=int(plan_horizon)).action


def planner_window(
    env: SharedSpectrumMultiUAVEnv,
    info: Mapping[str, Any],
    action_fn: ActionFn,
    *,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[TaskWaypointV2, np.ndarray]:
    positions = np.asarray(info["uav_positions_m"], dtype=np.float64).copy()
    energy = np.asarray(info["remaining_energy_per_uav"], dtype=np.float64).copy()
    queues = np.asarray(info["raw_queue_bits"], dtype=np.float64).copy()
    buoys = np.asarray(info["buoy_positions_m"], dtype=np.float64)
    path = [positions.copy()]
    services = []
    current_info = dict(info)
    for _ in range(int(horizon)):
        action = action_fn(env, current_info)
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
        current_info = _info_with_state(current_info, positions, energy, queues)
    label = labels_from_planner_window(np.stack(services, axis=0), path[0], path[-1], config=env.config, horizon=int(horizon))
    return label, np.stack(path, axis=0)


def planner_c_window(
    env: SharedSpectrumMultiUAVEnv,
    info: Mapping[str, Any],
    *,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[TaskWaypointV2, np.ndarray]:
    """One joint beam of length H; TW is the selected trajectory end, not H receding searches."""
    if int(horizon) not in PLANNING_HORIZONS:
        raise ValueError("planner C horizon must be in the frozen joint planning set")
    initial = _State(
        np.asarray(info["uav_positions_m"], dtype=np.float64).copy(),
        np.asarray(info["remaining_energy_per_uav"], dtype=np.float64).copy(),
        np.asarray(info["raw_queue_bits"], dtype=np.float64).copy(),
    )
    buoys = np.asarray(info["buoy_positions_m"], dtype=np.float64).copy()
    start = initial.positions_m.copy()
    beam: list[tuple[float, tuple[str, ...], _State, list[np.ndarray], list[np.ndarray]]] = [
        (0.0, tuple(), initial, [start.copy()], [])
    ]
    for _depth in range(int(horizon)):
        expanded = []
        for total, prefix, state, path, services in beam:
            for names, action in _joint_candidates(state, buoys, env.config):
                try:
                    step = _shadow(state, action, buoys, env, "joint_aware")
                except RuntimeError as error:
                    if "hard UAV separation projection did not converge" not in str(error):
                        raise
                    continue
                expanded.append(
                    (
                        total + float(np.sum(step.payload_bits)),
                        prefix + tuple(names),
                        _next(step),
                        path + [np.asarray(step.positions_m, dtype=np.float64).copy()],
                        services + [_service_from_step(step)],
                    )
                )
        expanded.sort(key=lambda row: (-row[0], row[1]))
        beam = expanded[: (49 if int(horizon) == 1 else BEAM_WIDTH)]
        if not beam:
            raise RuntimeError("planner C beam exhausted")
    _score, _names, _state, path, services = beam[0]
    label = labels_from_planner_window(np.stack(services, axis=0), path[0], path[-1], config=env.config, horizon=int(horizon))
    return label, np.stack(path, axis=0)


def make_conditioner(window_fn, *, horizon: int = DEFAULT_HORIZON):
    def conditioner(info, env):
        features = feature_dict(info, env.config)
        label, path = window_fn(env, info, horizon=horizon)
        attached = attach_task_waypoint(features, label, horizon=horizon)
        attached["_oracle_path_m"] = path
        return attached

    return conditioner


def _window_a(env, info, *, horizon: int = DEFAULT_HORIZON):
    return planner_window(env, info, planner_a_action, horizon=horizon)


def _window_b(env, info, *, horizon: int = DEFAULT_HORIZON):
    return planner_window(env, info, planner_b_action, horizon=horizon)


PLANNER_A = make_conditioner(_window_a)
PLANNER_B = make_conditioner(_window_b)
PLANNER_C = make_conditioner(planner_c_window)


__all__ = [
    "PLANNER_A",
    "PLANNER_B",
    "PLANNER_C",
    "make_conditioner",
    "planner_a_action",
    "planner_b_action",
    "planner_c_action",
    "planner_c_window",
    "planner_window",
]
