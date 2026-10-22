"""Frozen MAPPO traces used only to explain an existing assignment."""
from __future__ import annotations

import copy

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.contracts import refuse_rematch
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor
from wt.lowlevel_v2.phase5o_executor_joint_utility_audit.protocol import HOLD_STEPS


def run_assignment(env: L3BEnv, assignment: np.ndarray, actor, last_movement: np.ndarray, *, trace: bool) -> dict:
    """Fifteen locked steps. The actor weights stay frozen."""
    clone = copy.deepcopy(env)
    chosen = np.asarray(assignment, dtype=np.int64).copy()
    option = TaskOption(
        targets=chosen.copy(),
        decision_time=int(clone.t),
        remaining_hold=int(HOLD_STEPS),
        uav_mask=np.ones(clone.config.num_uavs, dtype=bool),
        buoy_mask=np.ones(clone.config.num_buoys, dtype=bool),
        association=chosen.copy(),
        urgency=None,
    )
    refuse_rematch(option)
    executor = LockedExecutor(actor)
    start = float(clone.payload_bits)
    movement = np.asarray(last_movement, dtype=np.float64).copy()
    steps = []
    completed = 0.0
    sinr_values: list[float] = []
    for step_index in range(1, int(HOLD_STEPS) + 1):
        if int(clone.t) >= int(clone.config.episode_horizon):
            break
        executed, _desired, _delta = executor.act(clone, option, movement)
        movement = np.asarray(executed, dtype=np.float64)
        info = clone.step(executed, option.association)
        delivered = np.asarray(info["step_service_blocks"], dtype=np.float64)
        completed += float(np.sum(delivered))
        assigned = [uav for uav, buoy in enumerate(chosen) if int(buoy) >= 0]
        sinr = np.asarray(info["sinr"], dtype=np.float64)
        rate = np.asarray(info["rate_bps"], dtype=np.float64)
        interference = np.asarray(info["interference_uav_uav"], dtype=np.float64)
        received = interference.sum(axis=1)
        if assigned:
            sinr_values.extend(float(sinr[uav]) for uav in assigned)
        payload = float(clone.payload_bits - start)
        if trace:
            for uav, buoy in enumerate(chosen):
                distance = float("nan")
                if int(buoy) >= 0:
                    distance = float(info["distance_uav_buoy"][uav, int(buoy)])
                steps.append(
                    {
                        "step": int(step_index),
                        "uav": int(uav),
                        "x": float(info["uav_positions_m"][uav, 0]),
                        "y": float(info["uav_positions_m"][uav, 1]),
                        "action_x": float(executed[uav, 0]),
                        "action_y": float(executed[uav, 1]),
                        "target": int(buoy),
                        "distance": distance,
                        "sinr": float(sinr[uav]),
                        "interference": float(received[uav]),
                        "rate": float(rate[uav]),
                        "service_progress": float(info["service_progress"][uav]),
                        "completed_blocks": float(completed),
                        "cumulative_payload": payload,
                    }
                )
    return {
        "j_mappo": float(clone.payload_bits - start),
        "completed_blocks": float(completed),
        "mean_sinr": float(np.mean(sinr_values)) if sinr_values else float("nan"),
        "min_sinr": float(np.min(sinr_values)) if sinr_values else float("nan"),
        "steps": steps,
    }


__all__ = ["run_assignment"]
