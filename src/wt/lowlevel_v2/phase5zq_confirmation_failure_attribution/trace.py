"""Replay the frozen confirmation controller and keep the environment's own fields."""
from __future__ import annotations

import numpy as np

from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.phase5_hgnn_urgency.urgency import shared_task_urgency
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor
from wt.lowlevel_v2.phase5zp_one_shot_fresh_confirmation.episode import (
    DecisionFailure,
    _publish_failure,
    select_primary,
    select_static,
)
from wt.lowlevel_v2.phase5zp_one_shot_fresh_confirmation.protocol import HOLD_STEPS

from .protocol import FRESH_POOL, VALUE_SEEDS, conservation_error


def _blank_decision() -> dict:
    return {
        "vector": "",
        "selected_value": float("nan"),
        "h1_sinr_mean": float("nan"),
        "h1_rate_mean": float("nan"),
        "assignment": np.full(6, -1, dtype=np.int64),
    }


def trace_episode(bundle: dict, scenario: int, system: str, *, static_seed: int | None = None) -> dict:
    """Same control loop as Phase-5ZP. The recorder does not choose actions."""
    env = L3BEnv()
    env.reset(seed=int(scenario))
    executor = LockedExecutor(bundle["actor"])
    if bundle["actor"] is None:
        raise RuntimeError("frozen actor is missing")
    capacity = float(env.config.queue_capacity_blocks)
    initial = float(np.sum(env.queue))
    pending = 0
    option = None
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    last_movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    previous_assignment = np.full(env.config.num_uavs, -2, dtype=np.int64)
    sinr_sum = 0.0
    progress_sum = 0.0
    steps = 0
    arrivals = 0.0
    decision = _blank_decision()
    option_start = 0
    option_index = -1
    uav_rows = []
    decisions = []
    drops = []
    while int(env.t) < int(env.config.episode_horizon):
        published = False
        if pending <= 0:
            try:
                if system == "primary":
                    decision = select_primary(bundle, env, last_arrival, last_positions, last_movement)
                else:
                    decision = select_static(bundle, env, last_arrival, last_positions, int(static_seed))
                option = decision["option"]
                pending = int(HOLD_STEPS)
                option.remaining_hold = int(HOLD_STEPS)
                published = True
                option_start = int(env.t)
                option_index += 1
                urgency = shared_task_urgency(env.queue)
                distance = np.linalg.norm(env.uav_positions[:, None, :] - env.buoy_positions[None, :, :], axis=-1)
                for uav, buoy in enumerate(decision["assignment"]):
                    same = int(buoy) >= 0 and int(env.served_k[uav]) == int(buoy)
                    others = [other for other, target in enumerate(decision["assignment"]) if other != uav and int(target) == int(buoy) and int(buoy) >= 0]
                    decisions.append(
                        {
                            "scenario_id": int(scenario),
                            "system": system,
                            "static_seed": "" if static_seed is None else int(static_seed),
                            "time_step": int(env.t),
                            "option_index": int(option_index),
                            "uav": int(uav),
                            "selected_buoy": int(buoy),
                            "queue": float(env.queue[int(buoy)]) if int(buoy) >= 0 else 0.0,
                            "urgency": float(urgency[int(buoy)]) if int(buoy) >= 0 else 0.0,
                            "current_progress": float(env.progress[uav]) if same else 0.0,
                            "distance": float(distance[uav, int(buoy)]) if int(buoy) >= 0 else float("nan"),
                            "instantaneous_rate": float(env.last_rate[uav]),
                            "h1_sinr": float(decision["h1_sinr_mean"]),
                            "expected_value": float(decision["selected_value"]),
                            "already_partially_served": bool(same and float(env.progress[uav]) > 0.0),
                            "other_uav_same_target": int(len(others) > 0),
                        }
                    )
            except DecisionFailure:
                _publish_failure("diagnostic", option)
        progress_before = np.asarray(env.progress, dtype=np.float64).copy()
        queue_before = np.asarray(env.queue, dtype=np.float64).copy()
        dropped_before = float(env.dropped)
        payload_before = float(env.payload_bits)
        positions_before = np.asarray(env.uav_positions, dtype=np.float64).copy()
        if option is None:
            executed = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
            association = np.asarray(env.assignment, dtype=np.int64).copy()
        else:
            association = np.asarray(option.association, dtype=np.int64).copy()
            executed, desired, _delta = executor.act(env, option, last_movement)
            if executor.projector.project(env, desired, option).constraint_status != "ok":
                executed = np.zeros_like(desired)
        info = env.step(executed, association)
        if option is not None:
            pending -= 1
            option.remaining_hold = int(pending)
        service = np.asarray(info["step_service_blocks"], dtype=np.float64)
        incoming = np.asarray(info["arrivals_blocks"], dtype=np.float64)
        overflow = np.maximum(0.0, queue_before - service + incoming - capacity)
        arrivals += float(np.sum(incoming))
        for buoy, amount in enumerate(overflow):
            if float(amount) > 0.0:
                drops.append(
                    {
                        "scenario_id": int(scenario),
                        "system": system,
                        "static_seed": "" if static_seed is None else int(static_seed),
                        "time_step": int(env.t - 1),
                        "buoy": int(buoy),
                        "dropped_blocks": float(amount),
                        "queue_before": float(queue_before[buoy]),
                        "arrival": float(incoming[buoy]),
                        "service": float(service[buoy]),
                    }
                )
        hold_now = int(option.remaining_hold) if option is not None else -1
        for uav in range(env.config.num_uavs):
            target = int(association[uav])
            switched = int(previous_assignment[uav] != -2 and previous_assignment[uav] != target)
            completed = bool(target >= 0 and float(service[target]) > 0.0)
            uav_rows.append(
                {
                    "scenario_id": int(scenario),
                    "system": system,
                    "static_seed": "" if static_seed is None else int(static_seed),
                    "time_step": int(env.t - 1),
                    "uav": int(uav),
                    "option_index": int(option_index),
                    "option_start": int(option_start),
                    "remaining_hold": int(hold_now),
                    "published": bool(published),
                    "assignment": target,
                    "previous_assignment": int(previous_assignment[uav]),
                    "target_switch": int(switched),
                    "progress_before": float(progress_before[uav]),
                    "progress_after": float(env.progress[uav]),
                    "completed_block": int(completed),
                    "completed_bits": float(env.config.block_payload_bits) if completed else 0.0,
                    "queue_before": float(queue_before[target]) if target >= 0 else 0.0,
                    "queue_after_sum": float(np.sum(env.queue)),
                    "dropped_step": float(overflow[target]) if target >= 0 else 0.0,
                    "arrival_step": float(incoming[target]) if target >= 0 else 0.0,
                    "sinr": float(info["sinr"][uav]),
                    "rate": float(info["rate_bps"][uav]),
                    "in_service": int(float(info["rate_bps"][uav]) > 0.0 or completed),
                    "distance": float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[target])) if target >= 0 else float("nan"),
                    "position_before_x": float(positions_before[uav, 0]),
                }
            )
        previous_assignment = np.asarray(association, dtype=np.int64).copy()
        last_arrival = np.asarray(info["arrivals_blocks"], dtype=np.float32)
        last_positions = positions_before.astype(np.float32)
        last_movement = np.asarray(executed, dtype=np.float64)
        sinr_sum += float(np.mean(info["sinr"]))
        progress_sum += float(np.mean(info["service_progress"]))
        steps += 1
        _ = dropped_before
    completed = float(env.payload_bits / float(env.config.block_payload_bits))
    residual = float(np.sum(env.queue))
    dropped = float(env.dropped)
    error = conservation_error(initial, arrivals, completed, dropped, residual)
    return {
        "scenario_id": int(scenario),
        "system": system,
        "static_seed": None if static_seed is None else int(static_seed),
        "initial": initial,
        "arrivals": arrivals,
        "completed": completed,
        "dropped": dropped,
        "residual": residual,
        "conservation_error": error,
        "payload": float(env.payload_bits),
        "mean_sinr": float(sinr_sum / max(steps, 1)),
        "mean_progress_gauge": float(progress_sum / max(steps, 1)),
        "rows": uav_rows,
        "decisions": decisions,
        "drops": drops,
    }


def trace_pool(bundle: dict) -> list[dict]:
    episodes = []
    for scenario in FRESH_POOL:
        print(f"reproduce {int(scenario)}", flush=True)
        episodes.append(trace_episode(bundle, int(scenario), "primary"))
        for seed in VALUE_SEEDS:
            episodes.append(trace_episode(bundle, int(scenario), "model_s", static_seed=int(seed)))
    return episodes
