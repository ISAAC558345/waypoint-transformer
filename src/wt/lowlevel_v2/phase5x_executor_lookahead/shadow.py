"""Shadow execution of a frozen assignment. The caller env is not stepped."""
from __future__ import annotations

import copy
import time

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv, conflict_mask
from wt.lowlevel_v2.contracts import refuse_rematch
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import PAYLOAD_BLOCK_BITS

from .protocol import HORIZONS


def fingerprint(env: L3BEnv) -> dict:
    """Everything L3BEnv actually stores. Velocity is not an environment field."""
    return {
        "uav_positions": np.array(env.uav_positions, copy=True),
        "buoy_positions": np.array(env.buoy_positions, copy=True),
        "queue": np.array(env.queue, copy=True),
        "progress": np.array(env.progress, copy=True),
        "served_k": np.array(env.served_k, copy=True),
        "assignment": np.array(env.assignment, copy=True),
        "last_sinr": np.array(env.last_sinr, copy=True),
        "last_rate": np.array(env.last_rate, copy=True),
        "payload_bits": float(env.payload_bits),
        "dropped": float(env.dropped),
        "t": int(env.t),
        "rng": copy.deepcopy(env.rng.bit_generator.state),
    }


def max_abs_diff(before: dict, after: dict) -> float:
    largest = 0.0
    for key in ("uav_positions", "buoy_positions", "queue", "progress", "served_k", "assignment", "last_sinr", "last_rate"):
        largest = max(largest, float(np.max(np.abs(np.asarray(before[key]) - np.asarray(after[key])))))
    largest = max(largest, abs(float(before["payload_bits"]) - float(after["payload_bits"])))
    largest = max(largest, abs(float(before["dropped"]) - float(after["dropped"])))
    largest = max(largest, abs(int(before["t"]) - int(after["t"])))
    if before["rng"] != after["rng"]:
        largest = max(largest, 1.0)
    return largest


def rollout_horizons(env: L3BEnv, assignment: np.ndarray, actor, last_movement: np.ndarray) -> dict:
    """One 15-step clone. Checkpoints keep only executed information."""
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
    executor = LockedExecutor(actor)
    start_payload = float(clone.payload_bits)
    start_progress = np.array(clone.progress, copy=True)
    initial_assignment = np.array(clone.assignment, copy=True)
    initial_distance = _assigned_distances(clone, chosen)
    movement = np.array(last_movement, copy=True)
    sinr_values: list[float] = []
    rate_values: list[float] = []
    interference_values: list[float] = []
    distance_values: list[float] = []
    action_values: list[float] = []
    action_norms: list[float] = []
    completed = 0.0
    per_uav_blocks = np.zeros(clone.config.num_uavs, dtype=np.float64)
    checkpoints = {}
    started = time.perf_counter()
    for step_index in range(1, 16):
        if int(clone.t) >= int(clone.config.episode_horizon):
            break
        executed, _desired, _delta = executor.act(clone, option, movement)
        movement = np.asarray(executed, dtype=np.float64)
        executed_array = np.asarray(executed, dtype=np.float64)
        action_values.extend(float(value) for value in executed_array.ravel())
        action_norms.append(float(np.mean(np.linalg.norm(executed_array, axis=1))))
        info = clone.step(executed, option.association)
        delivered = np.asarray(info["step_service_blocks"], dtype=np.float64)
        completed += float(np.sum(delivered))
        for uav, buoy in enumerate(chosen):
            if int(buoy) >= 0:
                per_uav_blocks[uav] += float(delivered[int(buoy)])
        assigned = [uav for uav, buoy in enumerate(chosen) if int(buoy) >= 0]
        sinr = np.asarray(info["sinr"], dtype=np.float64)
        rate = np.asarray(info["rate_bps"], dtype=np.float64)
        received = np.asarray(info["interference_uav_uav"], dtype=np.float64).sum(axis=1)
        distances = _assigned_distances(clone, chosen)
        if assigned:
            sinr_values.extend(float(sinr[uav]) for uav in assigned)
            rate_values.extend(float(rate[uav]) for uav in assigned)
            interference_values.extend(float(received[uav]) for uav in assigned)
            distance_values.extend(distances)
        if step_index in HORIZONS:
            blocked = conflict_mask(clone.assignment, clone.config.num_buoys)
            reached = []
            for uav, buoy in enumerate(chosen):
                if int(buoy) < 0:
                    continue
                in_radius = bool(clone._in_range(uav, int(buoy)))
                progressed = float(clone.progress[uav]) > float(start_progress[uav]) + 1.0e-8 or per_uav_blocks[uav] > 0.0
                reached.append(float(in_radius and not bool(blocked[uav]) and progressed))
            continued = 0
            switches = 0
            for uav, buoy in enumerate(chosen):
                if int(buoy) != int(initial_assignment[uav]):
                    switches += 1
                if int(buoy) >= 0 and int(buoy) == int(initial_assignment[uav]) and float(clone.progress[uav]) + per_uav_blocks[uav] + 1.0e-8 >= float(start_progress[uav]):
                    continued += 1
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            checkpoints[int(step_index)] = {
                "payload_h": float(clone.payload_bits - start_payload),
                "completed_blocks_h": float(completed),
                "progress_gain_h": float(completed + np.sum(clone.progress) - np.sum(start_progress)),
                "progress_end_sum": float(np.sum(clone.progress)),
                "q_exec_h": float(clone.payload_bits - start_payload) + float(np.sum(clone.progress)) * float(PAYLOAD_BLOCK_BITS),
                "mean_sinr_h": _mean(sinr_values),
                "min_sinr_h": float(np.min(sinr_values)) if sinr_values else float("nan"),
                "mean_rate_h": _mean(rate_values),
                "cumulative_rate_h": float(np.sum(rate_values)) if rate_values else 0.0,
                "mean_interference_h": _mean(interference_values),
                "max_interference_h": float(np.max(interference_values)) if interference_values else float("nan"),
                "mean_distance_h": _mean(distance_values),
                "distance_reduction_h": float(np.mean(initial_distance) - np.mean(distances)) if initial_distance and distances else float("nan"),
                "number_in_service_radius_h": int(sum(clone._in_range(uav, int(buoy)) for uav, buoy in enumerate(chosen) if int(buoy) >= 0)),
                "service_continuity_preserved_h": int(continued),
                "number_of_target_switches": int(switches),
                "low_level_action_norm": _mean(action_norms),
                "low_level_action_variance": float(np.var(action_values)) if action_values else float("nan"),
                "reachability_h": _mean(reached),
                "elapsed_ms": float(elapsed_ms),
            }
    after = fingerprint(env)
    if not np.array_equal(movement_before, np.asarray(last_movement)):
        raise RuntimeError("shadow rollout mutated the caller movement")
    difference = max_abs_diff(before, after)
    if difference > 0.0:
        raise RuntimeError(f"shadow rollout changed the caller environment by {difference}")
    return {"checkpoints": checkpoints, "integrity": difference}


def _assigned_distances(env: L3BEnv, assignment: np.ndarray) -> list[float]:
    values = []
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0:
            continue
        values.append(float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)])))
    return values


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


__all__ = ["fingerprint", "max_abs_diff", "rollout_horizons"]
