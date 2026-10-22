"""Closed-loop and fixed-assignment evaluation for several high-level policies."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6g_lowlevel_mappo.evaluate import _actor_move
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6ik_lowlevel_opt.evaluate import paired_summary
from wt.lowlevel_v2.contracts import minimum_pairwise_distance
from wt.lowlevel_v2.observation import communication_features
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M, LOG_STD_INIT
from wt.lowlevel_v2.residual_policy import ResidualActor

from .protocol import HOLD, assert_split_scenario


def load_robust_actor(path) -> ResidualActor:
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    return actor


def _entry(records: list[tuple[int, int]]) -> float:
    if not records:
        return 0.0
    return float(np.mean([item[0] for item in records]))


def run_episode(scenario_id: int, actor: ResidualActor, policy: str, selector, fixed_assignments: list | None = None) -> dict:
    assert_split_scenario(int(scenario_id), "eval")
    if fixed_assignments is None and policy == "uniform_g1_k6":
        raise RuntimeError("uniform tasks are training-only")
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    actor.eval()
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    limit = int(env.config.episode_horizon)
    radius = float(env.config.service_radius_m)
    position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
    predecessor = None
    option = None
    pending = 0
    served_sum = 0.0
    window_bits = []
    window_buffer = 0.0
    path = 0.0
    sinr, interference, projected, safety = [], [], [], []
    service_steps = 0
    entries: list[tuple[int, int]] = []
    watch: dict[int, int] = {}
    assignments = []
    anomalies = []
    while int(env.t) < limit:
        if pending <= 0:
            if option is not None:
                window_bits.append(float(window_buffer))
                window_buffer = 0.0
            if fixed_assignments is None:
                chosen = selector.choose(env, policy, None, position_prev, predecessor)
                assignment = np.asarray(chosen["assignment"], dtype=np.int64)
                option = chosen["option"]
            else:
                if len(assignments) >= len(fixed_assignments):
                    anomalies.append("fixed assignment sequence ended early")
                    break
                assignment = np.asarray(fixed_assignments[len(assignments)], dtype=np.int64)
                option = make_option(env, assignment)
            if not is_legal(assignment, int(env.config.num_uavs), int(env.config.num_buoys)):
                anomalies.append("illegal assignment")
                break
            if not np.array_equal(option.association, option.targets):
                anomalies.append("task option diverged")
                break
            assignments.append([int(item) for item in assignment])
            pending = int(HOLD)
            watch = {}
            for uav, buoy in enumerate(assignment):
                if int(buoy) < 0:
                    continue
                dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
                if dist > radius:
                    watch[uav] = 0
        if option is None:
            anomalies.append("no TaskOption")
            break
        association = np.asarray(option.association, dtype=np.int64).copy()
        position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
        predecessor = int(env.t)
        movement, intervened, _norm = _actor_move(env, actor, option, last)
        payload_before = float(env.payload_bits)
        info = env.step(movement, association)
        if float(np.max(np.abs(env.conservation_error_bits()))) > float(env.tol_bits):
            anomalies.append("accounting")
            break
        served = np.asarray(info["served_bits"], dtype=np.float64)
        step_served = float(np.sum(served))
        if abs(float(env.payload_bits) - payload_before - step_served) > float(env.tol_bits):
            anomalies.append("payload double count")
            break
        last = np.asarray(movement, dtype=np.float64)
        option.remaining_hold -= 1
        pending -= 1
        served_sum += step_served
        window_buffer += step_served
        path += float(np.sum(np.linalg.norm(movement, axis=-1) * env.config.step_length_m))
        sinr.append(float(np.mean(info["sinr"])))
        interference.append(float(np.mean(communication_features(env)["total_interference"])))
        projected.append(int(intervened))
        safety.append(int(minimum_pairwise_distance(env.uav_positions) + 1.0e-9 < float(D_SAFE_M)))
        for uav, buoy in enumerate(association):
            if int(buoy) < 0:
                continue
            dist_after = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
            if float(served[int(buoy)]) > 1.0:
                service_steps += 1
            if uav in watch:
                watch[uav] += 1
                if dist_after <= radius:
                    entries.append((int(watch.pop(uav)), 1))
        if pending <= 0:
            entries.extend((int(elapsed), 0) for elapsed in watch.values())
            watch = {}
    if option is not None and not anomalies:
        window_bits.append(float(window_buffer))
    if watch:
        entries.extend((int(elapsed), 0) for elapsed in watch.values())
    payload = float(env.payload_bits)
    full_windows = window_bits[:3]
    return {
        "scenario_id": int(scenario_id),
        "policy": str(policy),
        "complete": int(env.t) == limit and not anomalies and abs(payload - served_sum) <= float(env.tol_bits),
        "payload_bits": payload,
        "served_bits": float(served_sum),
        "window_bits15": float(np.mean(full_windows)) if full_windows else 0.0,
        "service_steps": int(service_steps),
        "path_length_m": float(path),
        "sinr_mean": float(np.mean(sinr)) if sinr else 0.0,
        "interference_mean": float(np.mean(interference)) if interference else 0.0,
        "projection_rate": float(np.mean(projected)) if projected else 0.0,
        "safety_violation_rate": float(np.mean(safety)) if safety else 0.0,
        "mean_first_entry_steps": _entry(entries),
        "assignments": assignments,
        "anomaly_count": int(len(anomalies)),
        "anomalies": anomalies,
    }


def fixed_summary(baseline: list[dict], candidate: list[dict]) -> dict:
    left = {int(row["scenario_id"]): row for row in baseline}
    delta = [float(row["payload_bits"]) - float(left[int(row["scenario_id"])]["payload_bits"]) for row in candidate]
    values = np.asarray(delta, dtype=np.float64)
    return {
        "scenarios": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "positive": int(np.sum(values > 1.0)),
        "negative": int(np.sum(values < -1.0)),
    }


__all__ = ["fixed_summary", "load_robust_actor", "paired_summary", "run_episode"]
