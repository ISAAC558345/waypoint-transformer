"""Scheme-C dev episodes at a candidate residual scale. Actor weights stay frozen."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.lowlevel_v2.contracts import minimum_pairwise_distance
from wt.lowlevel_v2.observation import communication_features
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M

from .action import scaled_move
from .protocol import EVAL_POOL, HOLD, assert_eval_scenario


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


def run_scaled_episode(scenario_id: int, planner: FormalPlanner, alpha: float) -> dict:
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
    sinr, interference = [], []
    projected, safety, correction = [], [], []
    command_turn, executed_turn = [], []
    heading_fix, speed_fix = [], []
    command_speed, executed_speed, residual = [], [], []
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
        movement, report = scaled_move(env, planner.actor, option, last, float(alpha))
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
        heading_fix.append(float(report["heading_correction_deg"]))
        speed_fix.append(float(report["speed_correction"]))
        command_speed.append(float(report["command_speed"]))
        executed_speed.append(float(report["executed_speed"]))
        residual.append(float(report["residual_norm"]))
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
        "alpha": float(alpha),
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
        "heading_correction_deg": float(np.mean(heading_fix)) if heading_fix else 0.0,
        "speed_correction": float(np.mean(speed_fix)) if speed_fix else 0.0,
        "command_speed": float(np.mean(command_speed)) if command_speed else 0.0,
        "executed_speed": float(np.mean(executed_speed)) if executed_speed else 0.0,
        "residual_norm": float(np.mean(residual)) if residual else 0.0,
        "safety_violation_rate": float(np.mean(safety)) if safety else 0.0,
        "decision_count": int(decisions),
        "anomaly_count": int(len(anomalies)),
        "anomalies": anomalies,
        **_entry_times(outside_entries),
    }


def evaluate_scale(planner: FormalPlanner, alpha: float, scenarios: tuple[int, ...]) -> list[dict]:
    rows = []
    for scenario in scenarios:
        if int(scenario) not in EVAL_POOL:
            raise RuntimeError(f"phase-6in refused scenario {scenario}")
        row = run_scaled_episode(int(scenario), planner, float(alpha))
        rows.append(row)
        print(
            f"alpha {alpha:.2f} scenario {scenario} payload={row['payload_bits']:.3f} complete={row['complete']} proj={row['projection_rate']:.3f} turn={row['command_turn_deg']:.2f}",
            flush=True,
        )
    return rows


def paired_summary(baseline: list[dict], candidate: list[dict], *, draws: int, seed: int) -> dict:
    left = {int(row["scenario_id"]): row for row in baseline}
    right = {int(row["scenario_id"]): row for row in candidate}
    keys = sorted(set(left) & set(right))
    if not keys:
        raise RuntimeError("paired evaluation has no shared scenarios")
    delta = np.asarray([right[key]["payload_bits"] - left[key]["payload_bits"] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    choice = rng.integers(0, len(delta), size=(int(draws), len(delta)))
    means = delta[choice].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    positive = delta[delta > 1.0]
    top_share = float(np.max(positive) / np.sum(positive)) if positive.size else 0.0
    without_best = float(np.mean(np.delete(delta, int(np.argmax(delta))))) if len(delta) > 1 else float(delta[0])
    return {
        "scenarios": [int(key) for key in keys],
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "positive": int(np.sum(delta > 1.0)),
        "tie": int(np.sum(np.abs(delta) <= 1.0)),
        "negative": int(np.sum(delta < -1.0)),
        "top_positive_share": top_share,
        "mean_without_best": without_best,
        "baseline_mean_payload": float(np.mean([left[key]["payload_bits"] for key in keys])),
        "candidate_mean_payload": float(np.mean([right[key]["payload_bits"] for key in keys])),
        "baseline_service_steps": float(np.mean([left[key]["service_steps"] for key in keys])),
        "candidate_service_steps": float(np.mean([right[key]["service_steps"] for key in keys])),
        "baseline_entry_steps": float(np.mean([left[key]["mean_first_entry_steps"] for key in keys])),
        "candidate_entry_steps": float(np.mean([right[key]["mean_first_entry_steps"] for key in keys])),
        "baseline_path_m": float(np.mean([left[key]["path_length_m"] for key in keys])),
        "candidate_path_m": float(np.mean([right[key]["path_length_m"] for key in keys])),
        "baseline_sinr": float(np.mean([left[key]["sinr_mean"] for key in keys])),
        "candidate_sinr": float(np.mean([right[key]["sinr_mean"] for key in keys])),
        "baseline_interference": float(np.mean([left[key]["interference_mean"] for key in keys])),
        "candidate_interference": float(np.mean([right[key]["interference_mean"] for key in keys])),
        "baseline_projection": float(np.mean([left[key]["projection_rate"] for key in keys])),
        "candidate_projection": float(np.mean([right[key]["projection_rate"] for key in keys])),
        "baseline_correction": float(np.mean([left[key]["correction_norm"] for key in keys])),
        "candidate_correction": float(np.mean([right[key]["correction_norm"] for key in keys])),
        "baseline_safety": float(np.mean([left[key]["safety_violation_rate"] for key in keys])),
        "candidate_safety": float(np.mean([right[key]["safety_violation_rate"] for key in keys])),
        "baseline_turn_deg": float(np.mean([left[key]["command_turn_deg"] for key in keys])),
        "candidate_turn_deg": float(np.mean([right[key]["command_turn_deg"] for key in keys])),
        "baseline_executed_turn_deg": float(np.mean([left[key]["executed_turn_deg"] for key in keys])),
        "candidate_executed_turn_deg": float(np.mean([right[key]["executed_turn_deg"] for key in keys])),
        "incomplete": int(sum(not row["complete"] for row in list(left.values()) + list(right.values()))),
    }


__all__ = ["evaluate_scale", "paired_summary", "run_scaled_episode"]
