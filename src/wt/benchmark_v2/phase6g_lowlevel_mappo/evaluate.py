"""Frozen execution comparisons. Training does not call this module."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import geometry
from wt.benchmark_v2.phase6c.rollout import choose_assignment
from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
from wt.benchmark_v2.phase6e.generators import build_pools
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import (
    ALPHA_RESIDUAL,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    D_SAFE_M,
    DECISION_TIMES,
    DEV_POOL,
    EVAL_ASSIGNMENTS,
    HOLD,
    POSITIVE_BIT,
    SAFETY_RATE_MARGIN,
    SERVICE_SAFETY_MARGIN_M,
    assert_allowed,
)
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.observation import build_actor_observation, communication_features
from wt.lowlevel_v2.residual_policy import ResidualActor
from wt.lowlevel_v2.safety_projector import SafetyProjector


def _snapshot(env: L3BBitQueueEnvV2) -> dict:
    snap = capture_state(env)
    snap["initial_q_bits"] = np.asarray(env.initial_q_bits, dtype=np.float64).copy()
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in snap.items()}


def _restore(env: L3BBitQueueEnvV2, payload: dict) -> None:
    snap = {}
    for key, value in payload.items():
        if key in {"assignment", "served_k"}:
            snap[key] = np.asarray(value, dtype=np.int64)
        elif isinstance(value, list):
            snap[key] = np.asarray(value, dtype=np.float64)
        else:
            snap[key] = value
    restore_state(env, snap)
    env.initial_q_bits = np.asarray(payload["initial_q_bits"], dtype=np.float64).copy()


def build_eval_set() -> dict:
    rng = np.random.default_rng(321_096)
    states = []
    for scenario in DEV_POOL:
        assert_allowed(int(scenario), split="dev")
        env = L3BBitQueueEnvV2()
        install_light_info(env)
        env.reset(seed=int(scenario))
        for decision_time in DECISION_TIMES:
            if int(env.t) != int(decision_time):
                raise RuntimeError("eval state is not on a hold boundary")
            pools, _packed, illegal = build_pools(env, (("G1_RATE", 5),))
            if int(illegal) != 0:
                raise RuntimeError("eval pool is illegal")
            pool = pools[("G1_RATE", 5)]
            picked = [np.asarray(pool[int(index)], dtype=np.int64).tolist() for index in rng.choice(len(pool), size=EVAL_ASSIGNMENTS, replace=False)]
            states.append({"scenario_id": int(scenario), "decision_time": int(decision_time), "state": _snapshot(env), "assignments": picked})
            advance = np.asarray(picked[0], dtype=np.int64)
            steps = 0
            while steps < HOLD and int(env.t) < int(env.config.episode_horizon):
                env.step(fly_toward(env, advance), advance)
                steps += 1
    return {"source": "321096-321127", "assignments_per_state": EVAL_ASSIGNMENTS, "states": states, "used_j15": False}


def _actor_move(env, actor: ResidualActor, option, last_movement: np.ndarray):
    obs = build_actor_observation(env, option, last_movement)
    with torch.no_grad():
        mu, _log_std = actor(
            torch.as_tensor(obs["self"], dtype=torch.float32),
            torch.as_tensor(obs["task"], dtype=torch.float32),
            torch.as_tensor(obs["communication"], dtype=torch.float32),
            torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
            torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
        )
    pre = mu.detach().cpu().numpy().astype(np.float64)
    delta = float(ALPHA_RESIDUAL) * np.tanh(pre)
    desired = clip_unit_step(fly_toward(env, option.association) + delta)
    projected = SafetyProjector(CollisionConfig(0.4, 0.2, placeholder=False), safety_margin=float(SERVICE_SAFETY_MARGIN_M)).project(env, desired, option)
    failure = projected.constraint_status != "ok" or not projected.finite
    executed = np.zeros_like(desired) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
    return clip_unit_step(executed), bool(failure or float(projected.intervention_norm) > 1.0e-3), float(np.mean(np.linalg.norm(delta, axis=-1)))


def _mean_distance(env, association: np.ndarray) -> float:
    values = []
    for uav, buoy in enumerate(np.asarray(association, dtype=np.int64)):
        if int(buoy) < 0:
            continue
        values.append(float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)])))
    return float(np.mean(values)) if values else 0.0


def execute_option(payload: dict, assignment: list[int], actor: ResidualActor | None) -> dict:
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(payload["scenario_id"]))
    _restore(env, payload["state"])
    option = make_option(env, np.asarray(assignment, dtype=np.int64))
    horizon = min(HOLD, int(env.config.episode_horizon) - int(env.t))
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    start_distance = _mean_distance(env, option.association)
    served = 0.0
    sinr, rate, interference, distance, residual_norms = [], [], [], [], []
    path = 0.0
    projected = 0
    safety = 0
    illegal = 0
    accounting = 0
    series = []
    first_service = horizon
    for step in range(horizon):
        association = np.asarray(option.association, dtype=np.int64).copy()
        if actor is None:
            movement = fly_toward(env, association)
            projected_flag = False
            residual_norm = 0.0
        else:
            movement, projected_flag, residual_norm = _actor_move(env, actor, option, last)
        info = env.step(movement, association)
        if not np.array_equal(option.association, association):
            illegal += 1
        if float(np.max(np.abs(env.conservation_error_bits()))) > float(env.tol_bits):
            accounting += 1
        step_served = float(np.sum(info["served_bits"]))
        served += step_served
        if step_served > float(env.tol_bits) and first_service == horizon:
            first_service = step
        comm = communication_features(env)
        sinr.append(float(np.mean(info["sinr"])))
        rate.append(float(np.mean(info["rate_bps"])))
        interference.append(float(np.mean(comm["total_interference"])))
        distance.append(_mean_distance(env, association))
        path += float(np.sum(np.linalg.norm(movement, axis=-1)) * env.config.step_length_m)
        projected += int(projected_flag)
        residual_norms.append(float(residual_norm))
        safety += int(minimum_pairwise_distance(env.uav_positions) + 1.0e-9 < float(D_SAFE_M))
        series.append({"sinr": sinr[-1], "rate": rate[-1], "served_bits": step_served, "interference": interference[-1], "distance": distance[-1], "position": np.asarray(env.uav_positions, dtype=np.float64).copy()})
        option.remaining_hold -= 1
        last = movement.copy()
    final_distance = distance[-1] if distance else start_distance
    return {
        "served_bits": served,
        "mean_sinr": float(np.mean(sinr) if sinr else 0.0),
        "mean_rate": float(np.mean(rate) if rate else 0.0),
        "mean_interference": float(np.mean(interference) if interference else 0.0),
        "distance_reduction": float(start_distance - final_distance),
        "path_length": path,
        "service_utilization": float(np.mean([row["served_bits"] > 1.0 for row in series])) if series else 0.0,
        "projection_rate": float(projected / max(horizon, 1)),
        "residual_action_norm": float(np.mean(residual_norms) if residual_norms else 0.0),
        "safety_violation_rate": float(safety / max(horizon, 1)),
        "final_distance": float(final_distance),
        "first_service_step": int(first_service),
        "illegal": int(illegal),
        "accounting_failures": int(accounting),
        "series": series,
        "horizon": int(horizon),
    }


def bootstrap_mean(values: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    data = np.asarray(values, dtype=np.float64)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    count = int(data.size)
    for index in range(BOOTSTRAP_DRAWS):
        draws[index] = float(np.mean(data[rng.integers(0, count, count)]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(np.mean(data)), float(low), float(high)


def scenario_means(rows: list[dict], executor: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row["executor"] != executor:
            continue
        grouped.setdefault(int(row["scenario_id"]), []).append(float(row["delta_bits_vs_fly"]))
    return {scenario: float(np.mean(values)) for scenario, values in grouped.items()}


def run_full_episode(scenario: int, actor: ResidualActor | None) -> dict:
    assert_allowed(int(scenario), split="dev")
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario))
    option = None
    pending = 0
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    sinr, rate = [], []
    path = 0.0
    illegal = 0
    accounting = 0
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            chosen, _detail = choose_assignment("max_rate_independent", env, np.random.default_rng(0), geometry(env))
            option = make_option(env, chosen)
            pending = HOLD
        assert option is not None
        association = np.asarray(option.association, dtype=np.int64).copy()
        movement = fly_toward(env, association) if actor is None else _actor_move(env, actor, option, last)[0]
        info = env.step(movement, association)
        if not np.array_equal(option.association, association):
            illegal += 1
        if float(np.max(np.abs(env.conservation_error_bits()))) > float(env.tol_bits):
            accounting += 1
        sinr.append(float(np.mean(info["sinr"])))
        rate.append(float(np.mean(info["rate_bps"])))
        path += float(np.sum(np.linalg.norm(movement, axis=-1)) * env.config.step_length_m)
        pending -= 1
        option.remaining_hold = int(pending)
        last = movement.copy()
    return {
        "scenario_id": int(scenario),
        "cumulative_payload_bits": float(env.payload_bits),
        "residual_queue_bits": float(np.sum(env.q_bits)),
        "dropped_bits": float(np.sum(env.cumulative_dropped_bits)),
        "mean_sinr": float(np.mean(sinr)),
        "mean_rate": float(np.mean(rate)),
        "path_length": path,
        "illegal": int(illegal),
        "accounting_failures": int(accounting),
    }


def write_case_figures(cases: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    panels = (("sinr", "SINR"), ("rate", "rate"), ("served_bits", "served bits"), ("interference", "interference"), ("distance", "distance"))
    for case in cases:
        figure, axes = plt.subplots(2, 3, figsize=(10, 6))
        for axis, (key, title) in zip(axes.ravel(), panels):
            for name in ("fly", "mappo_v2"):
                axis.plot([row[key] for row in case["series"][name]], label=name)
            axis.set(title=title)
            axis.legend(fontsize=7)
        trajectory = axes.ravel()[5]
        for name in ("fly", "mappo_v2"):
            positions = np.stack([row["position"][0] for row in case["series"][name]])
            trajectory.plot(positions[:, 0], positions[:, 1], label=name)
        trajectory.set(title="UAV 0 trajectory")
        trajectory.legend(fontsize=7)
        figure.suptitle(case["label"])
        figure.tight_layout()
        figure.savefig(directory / f"{case['label']}.png")
        plt.close(figure)


def safety_regressed(mappo_rate: float, fly_rate: float) -> bool:
    return float(mappo_rate) > float(fly_rate) + float(SAFETY_RATE_MARGIN)


def positive_fraction(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64) > POSITIVE_BIT))
