"""One executed environment step. The caller environment is not stepped."""
from __future__ import annotations

import copy
import time

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv, active_links
from wt.lowlevel_v2.contracts import refuse_rematch
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor
from wt.lowlevel_v2.phase5x_executor_lookahead.shadow import fingerprint, max_abs_diff

from .protocol import RESPONSE_FEATURES


def assemble_response(
    *,
    sinr_0: np.ndarray,
    sinr_1: np.ndarray,
    interference_0: np.ndarray,
    interference_1: np.ndarray,
    progress_0: np.ndarray,
    progress_1: np.ndarray,
    rate_1: np.ndarray,
    pos_0: np.ndarray,
    pos_1: np.ndarray,
    buoy_pos: np.ndarray,
    assignment_0: np.ndarray,
    assignment_1: np.ndarray,
    in_range: np.ndarray,
    service_active: np.ndarray,
) -> np.ndarray:
    """Per-UAV H=1 response. Idle targets leave distance and motion at zero."""
    count = int(np.asarray(assignment_1).shape[0])
    out = np.zeros((count, len(RESPONSE_FEATURES)), dtype=np.float64)
    for index in range(count):
        out[index, 0] = float(sinr_1[index])
        out[index, 1] = float(sinr_1[index]) - float(sinr_0[index])
        out[index, 2] = float(interference_1[index])
        out[index, 3] = float(interference_1[index]) - float(interference_0[index])
        out[index, 4] = float(progress_1[index])
        out[index, 5] = float(progress_1[index]) - float(progress_0[index])
        out[index, 6] = float(rate_1[index])
        out[index, 11] = float(int(assignment_1[index]) == int(assignment_0[index]))
        target = int(assignment_1[index])
        if target < 0:
            continue
        direction = np.asarray(buoy_pos[target] - pos_0[index], dtype=np.float64)
        displacement = np.asarray(pos_1[index] - pos_0[index], dtype=np.float64)
        distance_0 = float(np.linalg.norm(direction))
        distance_1 = float(np.linalg.norm(pos_1[index] - buoy_pos[target]))
        out[index, 7] = distance_1
        out[index, 8] = distance_1 - distance_0
        out[index, 9] = float(in_range[index])
        out[index, 10] = float(service_active[index])
        direction_norm = float(np.linalg.norm(direction))
        motion_norm = float(np.linalg.norm(displacement))
        if direction_norm >= 1.0e-8:
            unit = direction / direction_norm
            radial = float(np.dot(displacement, unit))
            out[index, 12] = radial
            if motion_norm >= 1.0e-8:
                out[index, 13] = radial / motion_norm
    return out


def one_step(env: L3BEnv, assignment: np.ndarray, actor, last_movement: np.ndarray) -> dict:
    """Clone, one frozen action, one environment step. Payload is audited and then dropped."""
    before = fingerprint(env)
    movement_before = np.array(last_movement, copy=True)
    clone = copy.deepcopy(env)
    chosen = np.asarray(assignment, dtype=np.int64).copy()
    option = TaskOption(
        targets=chosen.copy(),
        decision_time=int(clone.t),
        remaining_hold=15,
        uav_mask=np.ones(clone.config.num_uavs, dtype=bool),
        buoy_mask=np.ones(clone.config.num_buoys, dtype=bool),
        association=chosen.copy(),
        urgency=None,
    )
    refuse_rematch(option)
    start_payload = float(clone.payload_bits)
    sinr_0 = np.array(clone.last_sinr, copy=True)
    progress_0 = np.array(clone.progress, copy=True)
    pos_0 = np.array(clone.uav_positions, copy=True)
    assignment_0 = np.array(clone.assignment, copy=True)
    buoy_pos = np.array(clone.buoy_positions, copy=True)
    interference_0 = np.asarray(clone.info()["interference_uav_uav"], dtype=np.float64).sum(axis=1)
    executor = LockedExecutor(actor)
    started = time.perf_counter()
    executed, _desired, _delta = executor.act(clone, option, np.array(last_movement, copy=True))
    action_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    info = clone.step(np.asarray(executed, dtype=np.float64), option.association)
    env_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    active = {int(uav) for uav, _buoy in active_links(clone)}
    in_range = np.asarray([float(clone._in_range(uav, int(buoy))) if int(buoy) >= 0 else 0.0 for uav, buoy in enumerate(chosen)], dtype=np.float64)
    service_active = np.asarray([float(uav in active) for uav in range(clone.config.num_uavs)], dtype=np.float64)
    response = assemble_response(
        sinr_0=sinr_0,
        sinr_1=np.asarray(info["sinr"], dtype=np.float64),
        interference_0=interference_0,
        interference_1=np.asarray(info["interference_uav_uav"], dtype=np.float64).sum(axis=1),
        progress_0=progress_0,
        progress_1=np.asarray(info["service_progress"], dtype=np.float64),
        rate_1=np.asarray(info["rate_bps"], dtype=np.float64),
        pos_0=pos_0,
        pos_1=np.asarray(info["uav_positions_m"], dtype=np.float64),
        buoy_pos=buoy_pos,
        assignment_0=assignment_0,
        assignment_1=np.asarray(info["assignment"], dtype=np.int64),
        in_range=in_range,
        service_active=service_active,
    )
    extract_ms = (time.perf_counter() - started) * 1000.0
    assigned = [uav for uav, buoy in enumerate(chosen) if int(buoy) >= 0]
    sinr = np.asarray(info["sinr"], dtype=np.float64)
    after = fingerprint(env)
    difference = max_abs_diff(before, after)
    if not np.array_equal(movement_before, np.asarray(last_movement)):
        difference = max(difference, 1.0)
    return {
        "response": response,
        "mean_sinr": float(np.mean(sinr[assigned])) if assigned else float("nan"),
        "progress_sum": float(np.sum(info["service_progress"])),
        "payload": float(info["cumulative_payload_bits"]) - start_payload,
        "integrity": float(difference),
        "action_ms": float(action_ms),
        "env_ms": float(env_ms),
        "extract_ms": float(extract_ms),
    }


__all__ = ["assemble_response", "one_step"]
