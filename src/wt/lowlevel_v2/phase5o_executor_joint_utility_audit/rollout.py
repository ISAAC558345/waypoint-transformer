"""Frozen MAPPO execution of one locked assignment. The actor is not updated."""
from __future__ import annotations

import copy

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.htw_phase9.features import in_range_completion
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.contracts import refuse_rematch
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor

from .protocol import HOLD_STEPS, TRACE_STEPS


def _option(env: L3BEnv, assignment: np.ndarray) -> TaskOption:
    targets = np.asarray(assignment, dtype=np.int64).copy()
    option = TaskOption(
        targets=targets,
        decision_time=int(env.t),
        remaining_hold=int(HOLD_STEPS),
        uav_mask=np.ones(env.config.num_uavs, dtype=bool),
        buoy_mask=np.ones(env.config.num_buoys, dtype=bool),
        association=targets.copy(),
        urgency=None,
    )
    refuse_rematch(option)
    return option


def _target_distances(info: dict, assignment: np.ndarray) -> list[float]:
    distance = np.asarray(info["distance_uav_buoy"], dtype=np.float64)
    values = []
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0:
            continue
        values.append(float(distance[int(uav), int(buoy)]))
    return values


def execute_mappo(env: L3BEnv, assignment: np.ndarray, actor, last_movement: np.ndarray, *, trace: bool) -> dict:
    """Fifteen locked steps. Payload is the increment of cumulative communication bits."""
    clone = copy.deepcopy(env)
    chosen = np.asarray(assignment, dtype=np.int64).copy()
    option = _option(clone, chosen)
    executor = LockedExecutor(actor)
    start = float(clone.payload_bits)
    movement = np.asarray(last_movement, dtype=np.float64).copy()
    distances: list[float] = []
    interference_sum = 0.0
    conflict_rates: list[float] = []
    service_blocks = 0.0
    traces: list[dict] = []
    for step_index in range(1, int(HOLD_STEPS) + 1):
        if int(clone.t) >= int(clone.config.episode_horizon):
            break
        executed, _desired, _delta = executor.act(clone, option, movement)
        movement = np.asarray(executed, dtype=np.float64)
        before = float(clone.payload_bits)
        info = clone.step(executed, option.association)
        distances.extend(_target_distances(info, chosen))
        interference_sum += float(np.sum(np.asarray(info["interference_uav_uav"], dtype=np.float64)))
        conflict_rates.append(float(info["conflict_rate"]))
        service_blocks += float(np.sum(np.asarray(info["step_service_blocks"], dtype=np.float64)))
        if trace and step_index in TRACE_STEPS:
            traces.append(
                {
                    "step": int(step_index),
                    "action": np.asarray(executed, dtype=np.float64).copy(),
                    "sinr": np.asarray(info["sinr"], dtype=np.float64).copy(),
                    "step_payload_bits": float(clone.payload_bits - before),
                    "cumulative_payload_bits": float(clone.payload_bits - start),
                    "step_service_blocks": np.asarray(info["step_service_blocks"], dtype=np.float64).copy(),
                    "service_progress": np.asarray(info["service_progress"], dtype=np.float64).copy(),
                    "in_range": np.asarray(
                        [1.0 if int(buoy) < 0 or clone._in_range(uav, int(buoy)) else 0.0 for uav, buoy in enumerate(chosen)],
                        dtype=np.float64,
                    ),
                    "residual_queue": float(np.sum(np.asarray(info["queue_blocks"], dtype=np.float64))),
                    "interference": float(np.sum(np.asarray(info["interference_uav_uav"], dtype=np.float64))),
                    "distance_to_target": np.asarray(
                        [
                            float(info["distance_uav_buoy"][uav, int(buoy)]) if int(buoy) >= 0 else float("nan")
                            for uav, buoy in enumerate(chosen)
                        ],
                        dtype=np.float64,
                    ),
                }
            )
    assigned = [int(buoy) >= 0 for buoy in chosen]
    assigned_hits = []
    for uav, buoy in enumerate(chosen):
        if int(buoy) < 0:
            continue
        assigned_hits.append(1.0 if clone._in_range(uav, int(buoy)) else 0.0)
    return {
        "j_mappo": float(clone.payload_bits - start),
        "cumulative_interference": float(interference_sum),
        "rollout_conflict_rate": float(np.mean(conflict_rates)) if conflict_rates else 0.0,
        "completion": float(in_range_completion(clone, chosen)),
        "assigned_in_range": float(np.mean(assigned_hits)) if assigned_hits else float("nan"),
        "residual_queue": float(np.sum(clone.queue)),
        "service_blocks": float(service_blocks),
        "mean_uav_target_distance_m": float(np.mean(distances)) if distances else float("nan"),
        "min_uav_target_distance_m": float(np.min(distances)) if distances else float("nan"),
        "n_assigned": float(sum(assigned)),
        "traces": traces,
    }


__all__ = ["execute_mappo"]
