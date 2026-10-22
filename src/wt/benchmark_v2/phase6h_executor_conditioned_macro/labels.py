"""Causal macro labels. MAPPO scoring never reads realized future arrivals."""
from __future__ import annotations

import time

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
from wt.benchmark_v2.phase6e.generators import feature_matrices, proposal_scores
from wt.benchmark_v2.phase6e.value import _prepare, _zero_arrivals, argmax_value, batch_j15
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import deterministic_move, parameter_token
from wt.lowlevel_v2.residual_policy import ResidualActor


def batch_j15_mappo(
    env: L3BBitQueueEnvV2,
    actor: ResidualActor,
    assignments: list[np.ndarray],
    horizon: int,
) -> tuple[dict[tuple[int, ...], float], float, float, float]:
    """Return causal MAPPO macro returns, full seconds, first-candidate seconds, and max conservation error."""
    if not assignments:
        return {}, 0.0, 0.0, 0.0
    scratch = _prepare(env, blind_interference=False)
    snap = capture_state(scratch)
    values: dict[tuple[int, ...], float] = {}
    worst = 0.0
    unique: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for assignment in assignments:
        key = tuple(int(item) for item in np.asarray(assignment, dtype=np.int64).reshape(-1))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    before = parameter_token(actor)
    started = time.perf_counter()
    first_seconds = 0.0
    for index, key in enumerate(unique):
        one = time.perf_counter()
        restore_state(scratch, snap)
        _zero_arrivals(scratch)
        option = make_option(scratch, np.asarray(key, dtype=np.int64))
        last = np.zeros((scratch.config.num_uavs, 2), dtype=np.float64)
        total = 0.0
        steps = 0
        while steps < int(horizon) and int(scratch.t) < int(scratch.config.episode_horizon):
            movement = deterministic_move(scratch, actor, option, last)
            info = scratch.step(movement, option.association)
            if not np.array_equal(option.association, np.asarray(key, dtype=np.int64)):
                raise RuntimeError("ILLEGAL")
            total += float(info["payload_increment_bits"])
            option.remaining_hold -= 1
            last = movement
            steps += 1
        values[key] = float(total)
        worst = max(worst, float(np.max(np.abs(scratch.conservation_error_bits()))))
        if index == 0:
            first_seconds = time.perf_counter() - one
    if parameter_token(actor) != before:
        raise RuntimeError("CONTAMINATION")
    if not np.all(np.isfinite(np.fromiter(values.values(), dtype=np.float64, count=len(values)))):
        raise RuntimeError("NONFINITE")
    return values, float(time.perf_counter() - started), float(first_seconds), float(worst)


def score_pool(env, actor: ResidualActor, pool: list[np.ndarray], horizon: int) -> dict:
    fly_values, fly_seconds, fly_error = batch_j15(env, pool, horizon, blind_interference=False)
    mappo_values, mappo_seconds, first_seconds, mappo_error = batch_j15_mappo(env, actor, pool, horizon)
    return {
        "fly": fly_values,
        "mappo": mappo_values,
        "fly_seconds": float(fly_seconds),
        "mappo_seconds": float(mappo_seconds),
        "mappo_first_seconds": float(first_seconds),
        "fly_error": float(fly_error),
        "mappo_error": float(mappo_error),
        "fly_best": argmax_value(pool, fly_values),
        "mappo_best": argmax_value(pool, mappo_values),
    }


def edge_ranks(env, assignment: tuple[int, ...]) -> list[int]:
    scores = proposal_scores(feature_matrices(env))["G1_RATE"]
    ranks = []
    for uav, buoy in enumerate(assignment):
        if int(buoy) < 0:
            continue
        order = np.argsort(-scores[uav], kind="mergesort")
        position = int(np.where(order == int(buoy))[0][0]) + 1
        ranks.append(position)
    return ranks


def ordered_keys(values: dict[tuple[int, ...], float]) -> list[tuple[int, ...]]:
    return sorted(values, key=lambda key: (-float(values[key]), key))
