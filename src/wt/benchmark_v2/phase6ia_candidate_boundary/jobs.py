"""Scenario jobs. Workers load the frozen actor and never update it."""
from __future__ import annotations

import ctypes
from ctypes import wintypes

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import geometry, link_rates
from wt.benchmark_v2.phase6c.rollout import choose_assignment
from wt.benchmark_v2.phase6e.generators import build_pools, feature_matrices, proposal_scores
from wt.benchmark_v2.phase6e.value import _prepare, _zero_arrivals, argmax_value
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import deterministic_move, limit_threads, load_actor, load_primary
from wt.benchmark_v2.phase6h_executor_conditioned_macro.jobs import _execute_shadow, _guard, _live_step
from wt.benchmark_v2.phase6h_executor_conditioned_macro.labels import batch_j15_mappo
from wt.benchmark_v2.phase6ia_candidate_boundary.protocol import (
    CONFIRM_POOL,
    DESIGN_POOL,
    EPS_BITS,
    HOLD,
    KS,
    PRIMARY_TIMES,
    ROBUSTNESS_SEEDS,
    assert_allowed,
)


def _peak_rss_bytes() -> int:
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        return 0
    return int(counters.PeakWorkingSetSize)


def _key(assignment) -> tuple[int, ...]:
    return tuple(int(item) for item in np.asarray(assignment, dtype=np.int64).reshape(-1))


def _text(assignment: tuple[int, ...]) -> str:
    return ",".join(str(item) for item in assignment)


def _pools(env) -> tuple[dict[int, list[np.ndarray]], int]:
    pools, _packed, illegal = build_pools(env, tuple(("G1_RATE", k) for k in KS))
    if int(illegal) != 0:
        raise RuntimeError("ILLEGAL")
    return {k: pools[("G1_RATE", k)] for k in KS}, int(illegal)


def _score_nested(env, actor, pools: dict[int, list[np.ndarray]], horizon: int) -> dict:
    keys = {k: [] for k in KS}
    seen = {k: set() for k in KS}
    for k in KS:
        for assignment in pools[k]:
            key = _key(assignment)
            if key not in seen[k]:
                seen[k].add(key)
                keys[k].append(key)
    extra = {5: keys[5]}
    if 6 in keys:
        extra[6] = [key for key in keys[6] if key not in seen[5]]
    if 7 in keys:
        extra[7] = [key for key in keys[7] if key not in seen[6]]
    values: dict[tuple[int, ...], float] = {}
    timing = {}
    worst = 0.0
    for k in KS:
        part, seconds, first, error = batch_j15_mappo(env, actor, [np.asarray(key, dtype=np.int64) for key in extra[k]], horizon)
        if float(error) > float(env.tol_bits):
            raise RuntimeError("ACCOUNTING")
        values.update(part)
        timing[k] = {"seconds": float(seconds), "first_seconds": float(first), "incremental_count": len(extra[k])}
        worst = max(worst, float(error))
    if not np.all(np.isfinite(np.fromiter(values.values(), dtype=np.float64, count=len(values)))):
        raise RuntimeError("NONFINITE")
    return {"values": values, "keys": keys, "timing": timing, "error": worst}


def _best(pool: list[np.ndarray], values: dict[tuple[int, ...], float], keys: list[tuple[int, ...]]) -> tuple[tuple[int, ...], float, float]:
    chosen, chosen_value = argmax_value(pool, values)
    maximum = max(float(values[key]) for key in keys)
    return chosen, float(chosen_value), float(maximum)


def _edge_rows(scores: np.ndarray, assignment: tuple[int, ...]) -> list[tuple[int, int, int]]:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), axis=1, kind="mergesort")
    rows = []
    for uav, buoy in enumerate(assignment):
        if int(buoy) < 0:
            continue
        rank = int(np.where(order[uav] == int(buoy))[0][0]) + 1
        rows.append((int(uav), int(buoy), rank))
    return rows


def _covered(edges: list[tuple[int, int, int]], k: int) -> int:
    if not edges:
        return 1
    return int(all(rank <= int(k) for _uav, _buoy, rank in edges))


def _added_interference(env, assignment: tuple[int, ...], uav: int, buoy: int) -> float:
    geo = geometry(env)
    active = []
    counts: dict[int, int] = {}
    for index, target in enumerate(assignment):
        target_id = int(target)
        if target_id < 0 or not bool(geo["in_range"][index, target_id]):
            continue
        if float(env.queue[target_id]) <= 1.0e-9:
            continue
        counts[target_id] = counts.get(target_id, 0) + 1
        active.append((index, target_id))
    active = [(index, target) for index, target in active if counts[target] == 1]
    total = 0.0
    buoy_cluster = 0 if int(buoy) < 6 else 1
    for other, other_buoy in active:
        if other == int(uav):
            continue
        kappa = env.config.kappa_same_cluster if (0 if other_buoy < 6 else 1) == buoy_cluster else env.config.kappa_cross_cluster
        total += float(geo["gain"][other, int(buoy)]) * float(env.config.tx_power) * float(kappa)
    return float(total)


def _causal_service(env, actor, assignment: tuple[int, ...], horizon: int) -> tuple[np.ndarray, np.ndarray, float]:
    scratch = _prepare(env, blind_interference=False)
    _zero_arrivals(scratch)
    option = make_option(scratch, np.asarray(assignment, dtype=np.int64))
    last = np.zeros((scratch.config.num_uavs, 2), dtype=np.float64)
    served = np.zeros(scratch.config.num_buoys, dtype=np.float64)
    first = np.full(scratch.config.num_uavs, -1, dtype=np.int64)
    steps = 0
    while steps < int(horizon) and int(scratch.t) < int(scratch.config.episode_horizon):
        movement = deterministic_move(scratch, actor, option, last)
        info = scratch.step(movement, option.association)
        bits = np.asarray(info["served_bits"], dtype=np.float64)
        for uav, buoy in enumerate(assignment):
            if int(buoy) >= 0 and int(first[uav]) < 0 and float(bits[int(buoy)]) > float(scratch.tol_bits):
                first[uav] = int(steps)
        served += bits
        option.remaining_hold -= 1
        last = movement
        steps += 1
    error = float(np.max(np.abs(scratch.conservation_error_bits())))
    if error > float(scratch.tol_bits):
        raise RuntimeError("ACCOUNTING")
    return first, served, error


def _state_interference(env, assignment: tuple[int, ...]) -> float:
    geo = geometry(env)
    _sinr, _rate, mean_interference = link_rates(
        geo["gain"],
        np.asarray(assignment, dtype=np.int64),
        env.queue,
        geo["in_range"],
        bandwidth_hz=float(env.config.bandwidth_hz),
        tx_power=float(env.config.tx_power),
        noise_power=float(env.config.noise_power),
        kappa_same=float(env.config.kappa_same_cluster),
        kappa_cross=float(env.config.kappa_cross_cluster),
    )
    return float(mean_interference)


def _cv(samples: list[float]) -> float:
    array = np.asarray(samples, dtype=np.float64)
    mean = float(np.mean(array))
    if abs(mean) <= EPS_BITS:
        return 0.0
    return float(np.std(array) / abs(mean))


def audit_scenario(scenario: int) -> dict:
    limit_threads()
    scenario = int(scenario)
    split = "design" if scenario in DESIGN_POOL else "confirm"
    assert_allowed(scenario, split=split)
    actor = load_primary()
    others = {seed: load_actor(seed) for seed in ROBUSTNESS_SEEDS} if split == "confirm" else {}
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=scenario)
    env._phase6h_last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    option = None
    pending = 0
    optimum, edges, mechanism, robust, isolation, runtime = [], [], [], [], [], []
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            pools, _illegal = _pools(env)
            horizon = min(HOLD, int(env.config.episode_horizon) - int(env.t))
            print(f"score {scenario} t={int(env.t)} n6={len(pools[6])}", flush=True)
            scored = _guard(env, actor, option, lambda: _score_nested(env, actor, pools, horizon))
            features = feature_matrices(env)
            scores = proposal_scores(features)["G1_RATE"]
            chosen = {}
            maxima = {}
            for k in KS:
                assignment, _tied, maximum = _best(pools[k], scored["values"], scored["keys"][k])
                chosen[k] = assignment
                maxima[k] = maximum
            gain_6_5 = maxima[6] - maxima[5]
            has_k7 = 7 in maxima
            gain_7_6 = (maxima[7] - maxima[6]) if has_k7 else ""
            gain_7_5 = (maxima[7] - maxima[5]) if has_k7 else ""
            scale = max(maxima[7] if has_k7 else maxima[6], EPS_BITS)
            edge_info = _edge_rows(scores, chosen[7] if has_k7 else chosen[6])
            primary = int(int(env.t) in PRIMARY_TIMES)
            max_rate, _detail = choose_assignment("max_rate_independent", env, np.random.default_rng(0), geometry(env))
            max_rate_key = _key(max_rate)
            optimum.append(
                {
                    "scenario_id": scenario,
                    "split": split,
                    "decision_time": int(env.t),
                    "primary": primary,
                    "horizon": int(horizon),
                    "candidate_count_5": len(pools[5]),
                    "candidate_count_6": len(pools[6]),
                    "candidate_count_7": len(pools[7]) if has_k7 else "",
                    "j5": maxima[5],
                    "j6": maxima[6],
                    "j7": maxima[7] if has_k7 else "",
                    "gain_6_5": gain_6_5,
                    "gain_7_6": gain_7_6,
                    "gain_7_5": gain_7_5,
                    "ngain_6_5": gain_6_5 / scale,
                    "ngain_7_6": (gain_7_6 / scale) if has_k7 else "",
                    "a5": _text(chosen[5]),
                    "a6": _text(chosen[6]),
                    "a7": _text(chosen[7]) if has_k7 else "",
                    "disagree_6_5": int(chosen[6] != chosen[5]),
                    "disagree_7_6": int(chosen[7] != chosen[6]) if has_k7 else "",
                    "disagree_7_5": int(chosen[7] != chosen[5]) if has_k7 else "",
                    "covered_k5": _covered(edge_info, 5),
                    "covered_k6": _covered(edge_info, 6),
                    "covered_k7": _covered(edge_info, 7) if has_k7 else "",
                    "contains_rank6": int(any(rank == 6 for _uav, _buoy, rank in edge_info)),
                    "contains_rank7": int(any(rank == 7 for _uav, _buoy, rank in edge_info)) if has_k7 else "",
                    "mean_interference": _state_interference(env, max_rate_key),
                    "conservation_error": scored["error"],
                }
            )
            for uav, buoy, rank in edge_info:
                edges.append(
                    {
                        "scenario_id": scenario,
                        "split": split,
                        "decision_time": int(env.t),
                        "primary": primary,
                        "uav": uav,
                        "buoy": buoy,
                        "edge_rank": rank,
                    }
                )
            full_seconds = {k: sum(scored["timing"][item]["seconds"] for item in KS if item <= k) for k in KS}
            for k in KS:
                runtime.append(
                    {
                        "scenario_id": scenario,
                        "split": split,
                        "decision_time": int(env.t),
                        "primary": primary,
                        "k": k,
                        "candidate_count": len(pools[k]),
                        "full_pool_seconds": full_seconds[k],
                        "single_eval_seconds": scored["timing"][k]["first_seconds"],
                        "peak_rss_bytes": _peak_rss_bytes(),
                    }
                )
            isolation.append(
                {
                    "scenario_id": scenario,
                    "decision_time": int(env.t),
                    "shadow_state_contamination": 0,
                    "conservation_error": scored["error"],
                    "actor_stateless": 1,
                }
            )
            if primary and chosen[6] != chosen[5]:
                new_edges = _edge_rows(scores, chosen[6])
                new_edges = [item for item in new_edges if item[2] > 5]
                if not new_edges:
                    mechanism.append(
                        {
                            "scenario_id": scenario,
                            "split": split,
                            "decision_time": int(env.t),
                            "kind": "recombination_inside_k5",
                            "uav": -1,
                            "buoy": -1,
                            "edge_rank": -1,
                            "distance_m": "",
                            "q_bits": "",
                            "instantaneous_rate": "",
                            "interference": "",
                            "first_service_step": "",
                            "h15_served_bits": "",
                            "gain_6_5": gain_6_5,
                        }
                    )
                else:
                    first, served, _error = _guard(
                        env,
                        actor,
                        option,
                        lambda: _causal_service(env, actor, chosen[6], horizon),
                    )
                    for uav, buoy, rank in new_edges:
                        mechanism.append(
                            {
                                "scenario_id": scenario,
                                "split": split,
                                "decision_time": int(env.t),
                                "kind": "new_rank_edge",
                                "uav": uav,
                                "buoy": buoy,
                                "edge_rank": rank,
                                "distance_m": float(features["distance"][uav, buoy]),
                                "q_bits": float(env.q_bits[buoy]),
                                "instantaneous_rate": float(features["rate"][uav, buoy]),
                                "interference": _added_interference(env, chosen[6], uav, buoy),
                                "first_service_step": int(first[uav]),
                                "h15_served_bits": float(served[buoy]),
                                "gain_6_5": gain_6_5,
                            }
                        )
            if split == "confirm" and primary:
                returns = {}
                slots = (("a5", chosen[5]), ("a6", chosen[6]))
                if has_k7:
                    slots = slots + (("a7", chosen[7]),)
                for name, assignment in slots:
                    returns[name] = {}
                    for seed, seed_actor in others.items():
                        returns[name][seed] = _guard(
                            env,
                            seed_actor,
                            option,
                            lambda assignment=assignment, seed_actor=seed_actor: _execute_shadow(env, seed_actor, assignment, horizon),
                        )
                for seed in ROBUSTNESS_SEEDS:
                    robust.append(
                        {
                            "scenario_id": scenario,
                            "decision_time": int(env.t),
                            "model_seed": int(seed),
                            "a5_bits": float(returns["a5"][seed]),
                            "a6_bits": float(returns["a6"][seed]),
                            "a7_bits": float(returns["a7"][seed]) if "a7" in returns else "",
                            "gain_6_5": float(returns["a6"][seed] - returns["a5"][seed]),
                            "gain_7_6": float(returns["a7"][seed] - returns["a6"][seed]) if "a7" in returns else "",
                            "cv_a5": _cv(list(returns["a5"].values())),
                            "cv_a6": _cv(list(returns["a6"].values())),
                            "cv_a7": _cv(list(returns["a7"].values())) if "a7" in returns else "",
                        }
                    )
            option = make_option(env, np.asarray(max_rate_key, dtype=np.int64))
            pending = HOLD
        _live_step(env, actor, option, use_fly=False)
        pending -= 1
    print(f"audit {scenario}", flush=True)
    return {
        "scenario_id": scenario,
        "split": split,
        "optimum": optimum,
        "edges": edges,
        "mechanism": mechanism,
        "robust": robust,
        "isolation": isolation,
        "runtime": runtime,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def closed_loop_scenario(job: tuple[int, int]) -> dict:
    limit_threads()
    scenario, k = int(job[0]), int(job[1])
    assert_allowed(scenario, split="sanity")
    if k not in KS:
        raise RuntimeError("ILLEGAL")
    actor = load_primary()
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=scenario)
    env._phase6h_last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    option = None
    pending = 0
    worst = 0.0
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            pools, _packed, illegal = build_pools(env, (("G1_RATE", k),))
            if int(illegal) != 0:
                raise RuntimeError("ILLEGAL")
            pool = pools[("G1_RATE", k)]
            horizon = min(HOLD, int(env.config.episode_horizon) - int(env.t))

            def _evaluate(pool=pool, horizon=horizon):
                values, _seconds, _first, error = batch_j15_mappo(env, actor, pool, horizon)
                if float(error) > float(env.tol_bits):
                    raise RuntimeError("ACCOUNTING")
                return values, float(error)

            values, error = _guard(env, actor, option, _evaluate)
            worst = max(worst, error)
            best, _value = argmax_value(pool, values)
            option = make_option(env, np.asarray(best, dtype=np.int64))
            pending = HOLD
        info = _live_step(env, actor, option, use_fly=False)
        if not np.isfinite(float(np.sum(info["served_bits"]))):
            raise RuntimeError("NONFINITE")
        pending -= 1
    if float(np.max(np.abs(env.conservation_error_bits()))) > float(env.tol_bits):
        raise RuntimeError("ACCOUNTING")
    print(f"closed k={k} {scenario} payload={float(env.payload_bits):.1f}", flush=True)
    return {
        "scenario_id": scenario,
        "k": k,
        "controller": f"c{k}",
        "payload_bits": float(env.payload_bits),
        "residual_queue_bits": float(np.sum(env.q_bits)),
        "dropped_bits": float(np.sum(env.cumulative_dropped_bits)),
        "conservation_error": worst,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
