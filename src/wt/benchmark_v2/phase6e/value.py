"""J15_CURRENT evaluation. Future arrivals stay at zero and never enter the score."""
from __future__ import annotations

import copy
import time
from dataclasses import replace

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
from wt.l3b.policies import fly_toward


def _quiet(env: L3BBitQueueEnvV2) -> None:
    env.info = lambda: {"sinr": env.last_sinr, "rate_bps": env.last_rate}


def _zero_arrivals(env: L3BBitQueueEnvV2) -> None:
    zeros = np.zeros(env.config.num_buoys, dtype=np.float64)
    env.arrivals = lambda: zeros.copy()
    env.arrivals_at = lambda time_step: zeros.copy()


def _prepare(env: L3BBitQueueEnvV2, *, blind_interference: bool) -> L3BBitQueueEnvV2:
    scratch = copy.deepcopy(env)
    if blind_interference:
        scratch.config = replace(scratch.config, kappa_same_cluster=0.0, kappa_cross_cluster=0.0)
    _quiet(scratch)
    _zero_arrivals(scratch)
    return scratch


def batch_j15(
    env: L3BBitQueueEnvV2,
    assignments: list[np.ndarray],
    horizon: int,
    *,
    blind_interference: bool = False,
) -> tuple[dict[tuple[int, ...], float], float, float]:
    """Return causal macro returns, elapsed seconds, and the max conservation error."""
    if not assignments:
        return {}, 0.0, 0.0
    scratch = _prepare(env, blind_interference=blind_interference)
    snap = capture_state(scratch)
    values: dict[tuple[int, ...], float] = {}
    worst = 0.0
    started = time.perf_counter()
    unique: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for assignment in assignments:
        key = tuple(int(item) for item in np.asarray(assignment, dtype=np.int64).reshape(-1))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    for key in unique:
        restore_state(scratch, snap)
        _zero_arrivals(scratch)
        chosen = np.asarray(key, dtype=np.int64)
        total = 0.0
        steps = 0
        while steps < int(horizon) and int(scratch.t) < int(scratch.config.episode_horizon):
            info = scratch.step(fly_toward(scratch, chosen), chosen)
            total += float(info["payload_increment_bits"])
            steps += 1
        values[key] = float(total)
        worst = max(worst, float(np.max(np.abs(scratch.conservation_error_bits()))))
    return values, float(time.perf_counter() - started), float(worst)


def argmax_value(pool: list[np.ndarray], values: dict[tuple[int, ...], float]) -> tuple[tuple[int, ...], float]:
    best_key: tuple[int, ...] | None = None
    best_value = -1.0
    for assignment in pool:
        key = tuple(int(item) for item in np.asarray(assignment, dtype=np.int64).reshape(-1))
        value = float(values[key])
        if best_key is None or value > best_value + 1.0 or (abs(value - best_value) <= 1.0 and key < best_key):
            best_key = key
            best_value = value
    if best_key is None:
        raise RuntimeError("candidate pool is empty")
    return best_key, float(best_value)


def rollout_edges(
    env: L3BBitQueueEnvV2,
    assignment: tuple[int, ...],
    horizon: int,
) -> dict:
    """One causal rollout used only to describe service timing."""
    scratch = _prepare(env, blind_interference=False)
    chosen = np.asarray(assignment, dtype=np.int64)
    first = np.full(scratch.config.num_uavs, -1, dtype=np.int64)
    lifetime = np.full(scratch.config.num_buoys, int(horizon), dtype=np.int64)
    alive = scratch.q_bits > scratch.tol_bits
    steps = 0
    while steps < int(horizon) and int(scratch.t) < int(scratch.config.episode_horizon):
        q_before = scratch.q_bits.copy()
        info = scratch.step(fly_toward(scratch, chosen), chosen)
        served = np.asarray(info["served_bits"], dtype=np.float64)
        for uav, buoy in enumerate(chosen):
            buoy_id = int(buoy)
            if buoy_id < 0 or int(first[uav]) >= 0:
                continue
            if float(served[buoy_id]) > float(scratch.tol_bits):
                first[uav] = int(steps)
        for buoy in range(scratch.config.num_buoys):
            if bool(alive[buoy]) and float(q_before[buoy]) > float(scratch.tol_bits) and float(scratch.q_bits[buoy]) <= float(scratch.tol_bits):
                if int(lifetime[buoy]) == int(horizon):
                    lifetime[buoy] = int(steps)
        steps += 1
    return {
        "first_service": first,
        "lifetime": lifetime,
        "conservation_error": float(np.max(np.abs(scratch.conservation_error_bits()))),
    }


__all__ = ["argmax_value", "batch_j15", "rollout_edges"]
