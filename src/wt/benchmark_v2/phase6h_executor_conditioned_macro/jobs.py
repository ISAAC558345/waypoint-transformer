"""Scenario jobs. Each worker loads the frozen actor and does not update it."""
from __future__ import annotations

import copy
import time

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import geometry
from wt.benchmark_v2.phase6c.rollout import choose_assignment
from wt.benchmark_v2.phase6e.generators import build_pools
from wt.benchmark_v2.phase6e.value import argmax_value, batch_j15
from wt.benchmark_v2.phase6f_system_bringup.episode import _contamination, _fingerprint
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import (
    deterministic_move,
    limit_threads,
    load_actor,
    load_primary,
    parameter_token,
)
from wt.benchmark_v2.phase6h_executor_conditioned_macro.labels import batch_j15_mappo, edge_ranks, ordered_keys, score_pool
from wt.benchmark_v2.phase6h_executor_conditioned_macro.metrics import ranking_shift
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import HOLD, ROBUSTNESS_SEEDS, assert_allowed
from wt.l3b.policies import fly_toward


def _pool(env):
    pools, _packed, illegal = build_pools(env, (("G1_RATE", 5),))
    if int(illegal) != 0:
        raise RuntimeError("ILLEGAL")
    pool = pools[("G1_RATE", 5)]
    if not pool:
        raise RuntimeError("ILLEGAL")
    return pool


def _guard(env, actor, option, call):
    before = _fingerprint(env, option)
    token = None if actor is None else parameter_token(actor)
    result = call()
    if _contamination(before, _fingerprint(env, option)) != 0:
        raise RuntimeError("CONTAMINATION")
    if token is not None and parameter_token(actor) != token:
        raise RuntimeError("CONTAMINATION")
    return result


def _execute_shadow(env, actor, assignment, horizon: int) -> float:
    scratch = copy.deepcopy(env)
    install_light_info(scratch)
    option = make_option(scratch, np.asarray(assignment, dtype=np.int64))
    last = np.zeros((scratch.config.num_uavs, 2), dtype=np.float64)
    total = 0.0
    steps = 0
    while steps < int(horizon) and int(scratch.t) < int(scratch.config.episode_horizon):
        movement = deterministic_move(scratch, actor, option, last)
        info = scratch.step(movement, option.association)
        total += float(np.sum(info["served_bits"]))
        option.remaining_hold -= 1
        last = movement
        steps += 1
    if float(np.max(np.abs(scratch.conservation_error_bits()))) > float(scratch.tol_bits):
        raise RuntimeError("ACCOUNTING")
    return float(total)


def _live_step(env, actor, option, use_fly: bool):
    association = np.asarray(option.association, dtype=np.int64).copy()
    movement = fly_toward(env, association) if use_fly else deterministic_move(env, actor, option, env._phase6h_last)
    info = env.step(movement, association)
    if not np.array_equal(option.association, association):
        raise RuntimeError("ILLEGAL")
    if float(np.max(np.abs(env.conservation_error_bits()))) > float(env.tol_bits):
        raise RuntimeError("ACCOUNTING")
    if not np.isfinite(float(env.payload_bits)):
        raise RuntimeError("NONFINITE")
    env._phase6h_last = movement
    option.remaining_hold -= 1
    return info


def label_shift_scenario(scenario: int) -> dict:
    limit_threads()
    started = time.perf_counter()
    assert_allowed(int(scenario), split="label")
    actor = load_primary()
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario))
    env._phase6h_last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    option = None
    pending = 0
    rows = []
    isolation = []
    runtime = []
    boundary = []
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            pool = _pool(env)
            horizon = min(HOLD, int(env.config.episode_horizon) - int(env.t))
            scored = _guard(env, actor, option, lambda: score_pool(env, actor, pool, horizon))
            if scored["fly_error"] > float(env.tol_bits) or scored["mappo_error"] > float(env.tol_bits):
                raise RuntimeError("ACCOUNTING")
            shift = ranking_shift(scored["fly"], scored["mappo"])
            shift["top1_agreement"] = int(scored["fly_best"][0] == scored["mappo_best"][0])
            shift["argmax_disagreement"] = int(scored["fly_best"][0] != scored["mappo_best"][0])
            shift["fly_top1"] = ",".join(str(item) for item in scored["fly_best"][0])
            shift["mappo_top1"] = ",".join(str(item) for item in scored["mappo_best"][0])
            shift["mappo_top1_fly_rank"] = ordered_keys(scored["fly"]).index(scored["mappo_best"][0]) + 1
            ranks = edge_ranks(env, scored["mappo_best"][0])
            rows.append({"scenario_id": int(scenario), "decision_time": int(env.t), "candidate_count": len(pool), **shift})
            isolation.append(
                {
                    "scenario_id": int(scenario),
                    "decision_time": int(env.t),
                    "shadow_state_contamination": 0,
                    "actor_stateless": 1,
                    "actor_parameter_change": 0,
                    "mappo_conservation_error": scored["mappo_error"],
                    "fly_conservation_error": scored["fly_error"],
                }
            )
            runtime.append(
                {
                    "scenario_id": int(scenario),
                    "stage": "label_shift",
                    "decision_time": int(env.t),
                    "candidate_count": len(pool),
                    "fly_pool_seconds": scored["fly_seconds"],
                    "mappo_pool_seconds": scored["mappo_seconds"],
                    "mappo_first_candidate_seconds": scored["mappo_first_seconds"],
                    "fly_per_candidate_seconds": scored["fly_seconds"] / max(len(pool), 1),
                    "mappo_per_candidate_seconds": scored["mappo_seconds"] / max(len(pool), 1),
                }
            )
            for rank in ranks:
                boundary.append({"scenario_id": int(scenario), "split": "label_shift", "decision_time": int(env.t), "edge_rank": int(rank)})
            chosen, _detail = choose_assignment("max_rate_independent", env, np.random.default_rng(0), geometry(env))
            option = make_option(env, chosen)
            pending = HOLD
        _live_step(env, actor, option, use_fly=True)
        pending -= 1
    runtime.append(
        {
            "scenario_id": int(scenario),
            "stage": "label_shift_episode",
            "decision_time": -1,
            "candidate_count": 0,
            "episode_seconds": time.perf_counter() - started,
        }
    )
    print(f"label {scenario}", flush=True)
    return {"rows": rows, "isolation": isolation, "runtime": runtime, "boundary": boundary}


def _select(env, actor, controller: str, option):
    if controller in {"c0", "c3"}:
        chosen, _detail = choose_assignment("max_rate_independent", env, np.random.default_rng(0), geometry(env))
        key = tuple(int(item) for item in np.asarray(chosen, dtype=np.int64).reshape(-1))
        return key, None, {"candidate_count": 1, "seconds": 0.0, "first_seconds": 0.0, "error": 0.0, "ranks": edge_ranks(env, key)}
    pool = _pool(env)
    horizon = min(HOLD, int(env.config.episode_horizon) - int(env.t))
    if controller == "c1":
        values, seconds, error = _guard(env, actor, option, lambda: batch_j15(env, pool, horizon, blind_interference=False))
        first = seconds / max(len(pool), 1)
    else:
        values, seconds, first, error = _guard(env, actor, option, lambda: batch_j15_mappo(env, actor, pool, horizon))
    if float(error) > float(env.tol_bits):
        raise RuntimeError("ACCOUNTING")
    best, _value = argmax_value(pool, values)
    return best, float(values[best]), {
        "candidate_count": len(pool),
        "seconds": float(seconds),
        "first_seconds": float(first),
        "error": float(error),
        "ranks": edge_ranks(env, best),
        "ordered": ordered_keys(values),
    }


def closed_loop_scenario(job: tuple[int, str]) -> dict:
    limit_threads()
    started = time.perf_counter()
    scenario, controller = int(job[0]), str(job[1])
    assert_allowed(scenario, split="closed")
    actor = None if controller == "c3" else load_primary()
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=scenario)
    env._phase6h_last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    option = None
    pending = 0
    decision_id = -1
    current = None
    decisions = []
    runtime = []
    boundary = []
    isolation = []
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            if current is not None:
                current["position_after"] = ";".join(f"{x:.3f},{y:.3f}" for x, y in env.uav_positions)
                current["truncated_window"] = int(current["steps"] < HOLD)
                decisions.append(current)
            decision_id += 1
            started_t = int(env.t)
            key, predicted, meta = _select(env, actor, controller, option)
            option = make_option(env, np.asarray(key, dtype=np.int64))
            pending = HOLD
            current = {
                "scenario_id": scenario,
                "controller": controller,
                "decision_id": decision_id,
                "decision_time": started_t,
                "assignment": ",".join(str(item) for item in key),
                "predicted_j15": predicted,
                "q_bits": float(np.sum(env.q_bits)),
                "position": ";".join(f"{x:.3f},{y:.3f}" for x, y in env.uav_positions),
                "served_bits": 0.0,
                "steps": 0,
                "truncated_window": 0,
            }
            if controller == "c2":
                for rank in meta["ranks"]:
                    boundary.append({"scenario_id": scenario, "split": "closed_loop_c2", "decision_time": started_t, "edge_rank": int(rank)})
            runtime.append(
                {
                    "scenario_id": scenario,
                    "stage": controller,
                    "decision_time": started_t,
                    "candidate_count": meta["candidate_count"],
                    "pool_seconds": meta["seconds"],
                    "first_candidate_seconds": meta["first_seconds"],
                    "per_candidate_seconds": meta["seconds"] / max(meta["candidate_count"], 1),
                }
            )
            isolation.append(
                {
                    "scenario_id": scenario,
                    "controller": controller,
                    "decision_time": started_t,
                    "shadow_state_contamination": 0,
                    "conservation_error": meta["error"],
                    "actor_stateless": 1,
                }
            )
        info = _live_step(env, actor, option, use_fly=controller == "c3")
        current["served_bits"] += float(np.sum(info["served_bits"]))
        current["steps"] += 1
        pending -= 1
    current["truncated_window"] = int(current["steps"] < HOLD)
    current["position_after"] = ";".join(f"{x:.3f},{y:.3f}" for x, y in env.uav_positions)
    decisions.append(current)
    runtime.append(
        {
            "scenario_id": scenario,
            "stage": f"{controller}_episode",
            "decision_time": -1,
            "candidate_count": 0,
            "episode_seconds": time.perf_counter() - started,
            "pool_seconds": float(sum(row["pool_seconds"] for row in runtime)),
        }
    )
    print(f"closed {controller} {scenario} payload={float(env.payload_bits):.1f}", flush=True)
    return {
        "scenario_id": scenario,
        "controller": controller,
        "payload_bits": float(env.payload_bits),
        "residual_queue_bits": float(np.sum(env.q_bits)),
        "dropped_bits": float(np.sum(env.cumulative_dropped_bits)),
        "decisions": decisions,
        "runtime": runtime,
        "boundary": boundary,
        "isolation": isolation,
    }


def mechanism_scenario(scenario: int) -> dict:
    limit_threads()
    started = time.perf_counter()
    assert_allowed(int(scenario), split="label")
    fly = closed_loop_like(scenario, "c3", "label")
    mappo = closed_loop_like(scenario, "c0", "label")
    return {"fly": fly, "mappo": mappo, "episode_seconds": time.perf_counter() - started}


def closed_loop_like(scenario: int, controller: str, split: str) -> dict:
    assert_allowed(int(scenario), split=split)
    actor = None if controller == "c3" else load_primary()
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario))
    env._phase6h_last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    option = None
    pending = 0
    current = None
    decisions = []
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            if current is not None:
                current["position_after"] = ";".join(f"{x:.3f},{y:.3f}" for x, y in env.uav_positions)
                decisions.append(current)
            key, _predicted, meta = _select(env, actor, controller, option)
            option = make_option(env, np.asarray(key, dtype=np.int64))
            pending = HOLD
            current = {
                "scenario_id": int(scenario),
                "executor": "fly_toward" if controller == "c3" else "mappo_v2_seed0",
                "decision_time": int(env.t),
                "assignment": ",".join(str(item) for item in key),
                "edge_ranks": ",".join(str(item) for item in meta["ranks"]),
                "q_bits": float(np.sum(env.q_bits)),
                "position": ";".join(f"{x:.3f},{y:.3f}" for x, y in env.uav_positions),
                "served_bits": 0.0,
            }
        info = _live_step(env, actor, option, use_fly=controller == "c3")
        current["served_bits"] += float(np.sum(info["served_bits"]))
        pending -= 1
    current["position_after"] = ";".join(f"{x:.3f},{y:.3f}" for x, y in env.uav_positions)
    decisions.append(current)
    return {"payload_bits": float(env.payload_bits), "decisions": decisions}


def robustness_scenario(scenario: int) -> dict:
    limit_threads()
    started = time.perf_counter()
    assert_allowed(int(scenario), split="robust")
    actors = {seed: load_actor(seed) for seed in ROBUSTNESS_SEEDS}
    primary = actors[0]
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario))
    env._phase6h_last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    option = None
    pending = 0
    rows = []
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            pool = _pool(env)
            horizon = min(HOLD, int(env.config.episode_horizon) - int(env.t))
            scored = _guard(env, primary, option, lambda: score_pool(env, primary, pool, horizon))
            if scored["mappo_error"] > float(env.tol_bits):
                raise RuntimeError("ACCOUNTING")
            mappo_order = ordered_keys(scored["mappo"])
            slots = {
                "seed0_j15": mappo_order[0],
                "max_rate": tuple(int(item) for item in np.asarray(choose_assignment("max_rate_independent", env, np.random.default_rng(0), geometry(env))[0], dtype=np.int64).reshape(-1)),
                "fly_j15": scored["fly_best"][0],
                "seed0_top2": mappo_order[1],
                "seed0_top3": mappo_order[2],
            }
            returns = {name: {} for name in slots}
            for name, assignment in slots.items():
                for seed, actor in actors.items():
                    returns[name][seed] = _guard(
                        env,
                        actor,
                        option,
                        lambda assignment=assignment, actor=actor: _execute_shadow(env, actor, assignment, horizon),
                    )
            reference = np.asarray([returns[name][0] for name in slots], dtype=np.float64)
            for seed in ROBUSTNESS_SEEDS:
                if seed == 0:
                    continue
                order = sorted(slots, key=lambda name: (-returns[name][seed], name))
                rank = order.index("seed0_j15") + 1
                rows.append(
                    {
                        "scenario_id": int(scenario),
                        "decision_time": int(env.t),
                        "model_seed": int(seed),
                        "seed0_assignment_rank": int(rank),
                        "seed0_bits": float(returns["seed0_j15"][0]),
                        "seed_bits_on_seed0_assignment": float(returns["seed0_j15"][seed]),
                        "slot_cv_mean": _slot_cv(returns),
                        "top2_for_other_seed": int(rank <= 2),
                    }
                )
            _ = reference
            option = make_option(env, np.asarray(mappo_order[0], dtype=np.int64))
            pending = HOLD
        _live_step(env, primary, option, use_fly=False)
        pending -= 1
    print(f"robust {scenario}", flush=True)
    return {"rows": rows, "episode_seconds": time.perf_counter() - started}


def _slot_cv(returns: dict) -> float:
    values = []
    for name, by_seed in returns.items():
        samples = np.asarray([by_seed[seed] for seed in ROBUSTNESS_SEEDS], dtype=np.float64)
        mean = float(np.mean(samples))
        if abs(mean) <= 1.0:
            values.append(0.0)
        else:
            values.append(float(np.std(samples) / abs(mean)))
        _ = name
    return float(np.mean(values))
