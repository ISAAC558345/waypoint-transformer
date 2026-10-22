"""Frozen H=1 transition plus the ACT0 that produced it. The caller is not stepped."""
from __future__ import annotations

import copy
import time

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv, active_links
from wt.lowlevel_v2.contracts import refuse_rematch
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor
from wt.lowlevel_v2.phase5x_executor_lookahead.shadow import fingerprint, max_abs_diff
from wt.lowlevel_v2.phase5zd_executor_response_value.response import assemble_response


def frozen_h1(env: L3BEnv, assignment: np.ndarray, actor, last_movement: np.ndarray) -> dict:
    """Same response definition as Phase-5ZD, also returning the executed action."""
    before = fingerprint(env)
    movement_before = np.array(last_movement, copy=True)
    started = time.perf_counter()
    clone = copy.deepcopy(env)
    clone_ms = (time.perf_counter() - started) * 1000.0
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
    started = time.perf_counter()
    interference_0 = np.asarray(clone.info()["interference_uav_uav"], dtype=np.float64).sum(axis=1)
    extract_ms = (time.perf_counter() - started) * 1000.0
    executor = LockedExecutor(actor)
    started = time.perf_counter()
    executed, _desired, _delta = executor.act(clone, option, np.array(last_movement, copy=True))
    action_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    info = clone.step(np.asarray(executed, dtype=np.float64), option.association)
    step_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    active = {int(uav) for uav, _buoy in active_links(clone)}
    in_range = np.asarray(
        [float(clone._in_range(uav, int(buoy))) if int(buoy) >= 0 else 0.0 for uav, buoy in enumerate(chosen)],
        dtype=np.float64,
    )
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
    extract_ms += (time.perf_counter() - started) * 1000.0
    assigned = [uav for uav, buoy in enumerate(chosen) if int(buoy) >= 0]
    sinr = np.asarray(info["sinr"], dtype=np.float64)
    after = fingerprint(env)
    difference = max_abs_diff(before, after)
    if not np.array_equal(movement_before, np.asarray(last_movement)):
        difference = max(difference, 1.0)
    return {
        "response": response,
        "action": np.asarray(executed, dtype=np.float64),
        "mean_sinr": float(np.mean(sinr[assigned])) if assigned else float("nan"),
        "progress_sum": float(np.sum(info["service_progress"])),
        "payload": float(info["cumulative_payload_bits"]) - start_payload,
        "integrity": float(difference),
        "clone_ms": float(clone_ms),
        "action_ms": float(action_ms),
        "step_ms": float(step_ms),
        "extract_ms": float(extract_ms),
    }


__all__ = ["frozen_h1"]
