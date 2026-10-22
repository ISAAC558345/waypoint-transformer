"""Scheme-C dev episodes for the adaptive residual actor."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import HOLD
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.benchmark_v2.phase6ik_lowlevel_opt.evaluate import paired_summary
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import minimum_pairwise_distance
from wt.lowlevel_v2.observation import build_actor_observation, communication_features
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M, LOG_STD_INIT
from wt.lowlevel_v2.residual_policy import ResidualActor

from .action import adaptive_command
from .protocol import EVAL_POOL, assert_eval_scenario
from .worker import _map_args, apply_command


def load_adaptive_actor(path) -> tuple[ResidualActor, dict]:
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    return actor, dict(payload["calibration"])


def adaptive_move(env, actor: ResidualActor, option, last_movement: np.ndarray, calibration: dict) -> tuple[np.ndarray, dict]:
    if actor.training:
        raise RuntimeError("adaptive actor left eval mode")
    obs = build_actor_observation(env, option, last_movement)
    with torch.no_grad():
        mu, _log_std = actor(
            torch.as_tensor(obs["self"], dtype=torch.float32),
            torch.as_tensor(obs["task"], dtype=torch.float32),
            torch.as_tensor(obs["communication"], dtype=torch.float32),
            torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
            torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
        )
    base = fly_toward(env, np.asarray(option.association, dtype=np.int64))
    commanded = adaptive_command(base, mu.detach().cpu().numpy(), **_map_args(calibration))
    return apply_command(env, option, commanded)


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


def run_adaptive_episode(scenario_id: int, planner: FormalPlanner, calibration: dict) -> dict:
    assert_eval_scenario(int(scenario_id))
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
    command_turn, executed_turn = [], []
    excessive_heading, excessive_speed, collision, envelope = [], [], [], []
    service_steps = 0
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
        position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
        predecessor = int(env.t)
        movement, report = adaptive_move(env, planner.actor, option, last, calibration)
        info = env.step(movement, association)
        if not np.array_equal(np.asarray(option.association, dtype=np.int64), association):
            anomalies.append("association changed")
            break
        error = float(np.max(np.abs(env.conservation_error_bits())))
        if error > float(env.tol_bits):
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
        command_turn.append(float(report["command_turn_deg"]))
        executed_turn.append(float(report["executed_turn_deg"]))
        excessive_heading.append(int(report["excessive_heading"]))
        excessive_speed.append(int(report["excessive_speed"]))
        collision.append(int(report["collision_risk"]))
        envelope.append(int(report["envelope_risk"]))
        safety.append(int(minimum_pairwise_distance(env.uav_positions) + 1.0e-9 < float(D_SAFE_M)))
        for uav, buoy in enumerate(association):
            if int(buoy) < 0:
                continue
            if float(served[int(buoy)]) > 1.0:
                service_steps += 1
            if uav in watch:
                dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
                watch[uav] += 1
                if dist <= radius:
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
        "path_length_m": float(path),
        "sinr_mean": float(np.mean(sinr)) if sinr else 0.0,
        "interference_mean": float(np.mean(interference)) if interference else 0.0,
        "projection_rate": float(np.mean(projected)) if projected else 0.0,
        "correction_norm": float(np.mean(correction)) if correction else 0.0,
        "command_turn_deg": float(np.mean(command_turn)) if command_turn else 0.0,
        "executed_turn_deg": float(np.mean(executed_turn)) if executed_turn else 0.0,
        "excessive_heading_rate": float(np.mean(excessive_heading)) if excessive_heading else 0.0,
        "excessive_speed_rate": float(np.mean(excessive_speed)) if excessive_speed else 0.0,
        "collision_risk_rate": float(np.mean(collision)) if collision else 0.0,
        "envelope_risk_rate": float(np.mean(envelope)) if envelope else 0.0,
        "safety_violation_rate": float(np.mean(safety)) if safety else 0.0,
        "decision_count": int(decisions),
        "anomaly_count": int(len(anomalies)),
        "anomalies": anomalies,
        **_entry_times(outside_entries),
    }


def evaluate_adaptive(actor: ResidualActor, calibration: dict, scenarios: tuple[int, ...]) -> list[dict]:
    for scenario in scenarios:
        assert_eval_scenario(int(scenario))
    planner = FormalPlanner("scheme_c")
    actor.eval()
    planner.actor = actor
    rows = []
    for scenario in scenarios:
        row = run_adaptive_episode(int(scenario), planner, calibration)
        rows.append(row)
        print(
            f"adaptive scenario {scenario} payload={row['payload_bits']:.3f} complete={row['complete']} proj={row['projection_rate']:.3f} turn={row['command_turn_deg']:.2f}",
            flush=True,
        )
    return rows


def smoke_episode(actor: ResidualActor, calibration: dict, scenario_id: int = 321000) -> dict:
    """One training-pool episode. Confirms kinematics, accounting, and the safety path."""
    from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import assert_allowed

    assert_allowed(int(scenario_id), split="train")
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    actor.eval()
    option = None
    pending = 0
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    served_sum = 0.0
    max_step = 0.0
    max_turn = 0.0
    intervened = 0
    steps = 0
    from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option, sample_assignment

    rng = np.random.default_rng(11)
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            option = make_option(env, sample_assignment(env, rng))
            pending = int(HOLD)
        association = np.asarray(option.association, dtype=np.int64).copy()
        movement, report = adaptive_move(env, actor, option, last, calibration)
        speed = np.linalg.norm(movement, axis=-1)
        if float(np.max(speed)) > 1.0 + 1.0e-6:
            raise RuntimeError("adaptive command exceeded one step")
        info = env.step(movement, association)
        if not np.array_equal(np.asarray(option.association, dtype=np.int64), association):
            raise RuntimeError("association changed")
        if float(np.max(np.abs(env.conservation_error_bits()))) > float(env.tol_bits):
            raise RuntimeError("accounting")
        served_sum += float(np.sum(info["served_bits"]))
        max_step = max(max_step, float(np.max(speed) * env.config.step_length_m))
        max_turn = max(max_turn, float(report["command_turn_deg"]))
        intervened += int(report["intervened"])
        steps += 1
        pending -= 1
        option.remaining_hold = int(pending)
        last = movement.copy()
    payload_ok = abs(float(env.payload_bits) - served_sum) <= float(env.tol_bits)
    if not payload_ok or steps != int(env.config.episode_horizon):
        raise RuntimeError("smoke episode did not finish cleanly")
    return {
        "scenario_id": int(scenario_id),
        "payload_bits": float(env.payload_bits),
        "max_step_m": float(max_step),
        "max_command_turn_deg": float(max_turn),
        "projection_rate": float(intervened / max(steps, 1)),
        "safety_ok": True,
        "accounting_ok": True,
    }


__all__ = ["evaluate_adaptive", "load_adaptive_actor", "paired_summary", "run_adaptive_episode", "smoke_episode"]
