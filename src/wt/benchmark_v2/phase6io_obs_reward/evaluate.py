"""Scheme-C evaluation of the original residual map. Shaping is not used at evaluation."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.benchmark_v2.phase6in_residual_scale.action import scaled_move
from wt.benchmark_v2.phase6ik_lowlevel_opt.evaluate import paired_summary
from wt.lowlevel_v2.contracts import minimum_pairwise_distance
from wt.lowlevel_v2.observation import communication_features
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M, LOG_STD_INIT
from wt.lowlevel_v2.residual_policy import ResidualActor

from .protocol import HOLD, OFFICIAL_ALPHA, assert_report_scenario


def load_shaped_actor(path) -> ResidualActor:
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    return actor


def _entry_times(records: list[tuple[int, int]]) -> dict[str, float]:
    if not records:
        return {"mean_first_entry_steps": 0.0, "entry_fraction": 0.0, "censored_fraction": 0.0}
    elapsed = np.asarray([item[0] for item in records], dtype=np.float64)
    entered = np.asarray([item[1] for item in records], dtype=np.float64)
    return {
        "mean_first_entry_steps": float(np.mean(elapsed)),
        "entry_fraction": float(np.mean(entered)),
        "censored_fraction": float(1.0 - np.mean(entered)),
    }


def run_residual_episode(scenario_id: int, planner: FormalPlanner) -> dict:
    assert_report_scenario(int(scenario_id))
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    limit = int(env.config.episode_horizon)
    radius = float(env.config.service_radius_m)
    position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
    predecessor = None
    option = None
    pending = 0
    served_sum = 0.0
    path = 0.0
    sinr, interference, projected, safety, correction = [], [], [], [], []
    service_steps = 0
    outside_steps = 0
    invalid_steps = 0
    outside_entries: list[tuple[int, int]] = []
    watch: dict[int, int] = {}
    anomalies = []
    decisions = 0
    while int(env.t) < limit:
        if pending <= 0:
            selection = planner.select_assignment(env, position_prev, predecessor)
            if selection.status != "selected" or selection.option is None:
                anomalies.append(selection.reason or selection.status)
                break
            option = selection.option
            pending = int(HOLD)
            decisions += 1
            watch = {}
            association_now = np.asarray(option.association, dtype=np.int64)
            for uav, buoy in enumerate(association_now):
                if int(buoy) < 0:
                    continue
                dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
                if dist > radius:
                    watch[uav] = 0
        if option is None:
            anomalies.append("no TaskOption")
            break
        association = np.asarray(option.association, dtype=np.int64).copy()
        before = np.asarray(env.uav_positions, dtype=np.float64).copy()
        position_prev = before
        predecessor = int(env.t)
        movement, report = scaled_move(env, planner.actor, option, last, float(OFFICIAL_ALPHA))
        info = env.step(movement, association)
        if not np.array_equal(np.asarray(option.association, dtype=np.int64), association):
            anomalies.append("association changed")
            break
        if float(np.max(np.abs(env.conservation_error_bits()))) > float(env.tol_bits):
            anomalies.append("accounting")
            break
        last = np.asarray(movement, dtype=np.float64)
        option.remaining_hold -= 1
        pending -= 1
        served = np.asarray(info["served_bits"], dtype=np.float64)
        served_sum += float(np.sum(served))
        path += float(np.sum(np.linalg.norm(movement, axis=-1) * env.config.step_length_m))
        sinr.append(float(np.mean(info["sinr"])))
        interference.append(float(np.mean(communication_features(env)["total_interference"])))
        projected.append(int(report["intervened"]))
        correction.append(float(report["correction_norm"]))
        safety.append(int(minimum_pairwise_distance(env.uav_positions) + 1.0e-9 < float(D_SAFE_M)))
        for uav, buoy in enumerate(association):
            if int(buoy) < 0:
                continue
            dist_before = float(np.linalg.norm(before[uav] - env.buoy_positions[int(buoy)]))
            dist_after = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
            if dist_before > radius:
                outside_steps += 1
                if dist_after >= dist_before - 1.0e-6:
                    invalid_steps += 1
            if float(served[int(buoy)]) > 1.0:
                service_steps += 1
            if uav in watch:
                watch[uav] += 1
                if dist_after <= radius:
                    outside_entries.append((int(watch.pop(uav)), 1))
        if pending <= 0:
            outside_entries.extend((int(elapsed), 0) for elapsed in watch.values())
            watch = {}
    if watch:
        outside_entries.extend((int(elapsed), 0) for elapsed in watch.values())
    payload = float(env.payload_bits)
    return {
        "scenario_id": int(scenario_id),
        "complete": int(env.t) == limit and not anomalies and abs(payload - served_sum) <= float(env.tol_bits),
        "payload_bits": payload,
        "served_bits": float(served_sum),
        "service_steps": int(service_steps),
        "invalid_motion_steps": int(invalid_steps),
        "outside_steps": int(outside_steps),
        "path_length_m": float(path),
        "sinr_mean": float(np.mean(sinr)) if sinr else 0.0,
        "interference_mean": float(np.mean(interference)) if interference else 0.0,
        "projection_rate": float(np.mean(projected)) if projected else 0.0,
        "correction_norm": float(np.mean(correction)) if correction else 0.0,
        "safety_violation_rate": float(np.mean(safety)) if safety else 0.0,
        "decision_count": int(decisions),
        "anomaly_count": int(len(anomalies)),
        "anomalies": anomalies,
        **_entry_times(outside_entries),
    }


def evaluate_residual(actor: ResidualActor, scenarios: tuple[int, ...]) -> list[dict]:
    planner = FormalPlanner("scheme_c")
    actor.eval()
    planner.actor = actor
    rows = []
    for scenario in scenarios:
        row = run_residual_episode(int(scenario), planner)
        rows.append(row)
        print(
            f"scenario {scenario} payload={row['payload_bits']:.3f} complete={row['complete']} invalid={row['invalid_motion_steps']}",
            flush=True,
        )
    return rows


__all__ = ["evaluate_residual", "load_shaped_actor", "paired_summary", "run_residual_episode"]
