"""Non-learning association/movement scripts for Level-2-A causal checks."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .env import L2AEnv, project_exclusive


def fly_toward(env: L2AEnv, assignment: np.ndarray) -> np.ndarray:
    move = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0:
            continue
        delta = env.buoy_positions[int(buoy)] - env.uav_positions[uav]
        norm = float(np.linalg.norm(delta))
        if norm > 1.0e-9:
            move[uav] = delta / norm
    return move


def myopic_assignment(env: L2AEnv) -> np.ndarray:
    """Nearest positive-queue buoy, exclusive. Analog of L1 greedy matching."""
    requested = np.full(env.config.num_uavs, -1, dtype=np.int64)
    taken = set()
    order = list(range(env.config.num_uavs))
    for uav in order:
        best, best_dist = -1, 1.0e18
        for buoy in range(env.config.num_buoys):
            if buoy in taken or env.queue[buoy] <= 1.0e-9:
                continue
            dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[buoy]))
            if dist < best_dist:
                best, best_dist = buoy, dist
        requested[uav] = best
        if best >= 0:
            taken.add(int(best))
    return project_exclusive(requested, env.uav_positions, env.buoy_positions)


def _cluster_work(env: L2AEnv) -> tuple[float, float]:
    leftover = env.remaining_arrivals()
    work = env.queue + leftover
    return float(np.sum(work[:6])), float(np.sum(work[6:]))


def assign_split(env: L2AEnv, n_far: int) -> np.ndarray:
    """Send n_far UAVs to the far cluster; others stay near. Exclusive nearest queue."""
    n_far = int(np.clip(n_far, 0, env.config.num_uavs))
    far_center = np.mean(env.buoy_positions[6:], axis=0)
    dist_far = np.linalg.norm(env.uav_positions - far_center, axis=-1)
    far_uavs = set(np.argsort(dist_far, kind="stable")[:n_far].tolist())
    requested = np.full(env.config.num_uavs, -1, dtype=np.int64)
    taken: set[int] = set()
    leftover = env.remaining_arrivals()
    for uav in range(env.config.num_uavs):
        cluster = range(6, 12) if uav in far_uavs else range(0, 6)
        best, best_dist = -1, 1.0e18
        for buoy in cluster:
            if buoy in taken:
                continue
            if env.queue[buoy] <= 1.0e-9 and float(leftover[buoy]) <= 1.0e-9:
                continue
            dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[buoy]))
            if dist < best_dist:
                best, best_dist = buoy, dist
        requested[uav] = best
        if best >= 0:
            taken.add(int(best))
    return project_exclusive(requested, env.uav_positions, env.buoy_positions)


def _shadow_payload(env: L2AEnv, n_far: int, horizon: int | None = None) -> float:
    import copy

    clone = copy.deepcopy(env)
    start = float(clone.payload_bits)
    remaining = int(clone.config.episode_horizon - clone.t)
    steps = remaining if horizon is None else min(int(horizon), remaining)
    for _ in range(max(0, steps)):
        chosen = assign_split(clone, n_far)
        clone.step(fly_toward(clone, chosen), chosen)
        if clone.t >= int(clone.config.episode_horizon):
            break
    return float(clone.payload_bits - start)


def oracle_option_assignment(env: L2AEnv) -> np.ndarray:
    """Hold a cluster option chosen by a single full-horizon shadow search at t=0."""
    near_work, far_work = _cluster_work(env)
    if not hasattr(env, "_oracle_n_far") or int(getattr(env, "_oracle_n_t", -1)) > int(env.t):
        env._oracle_n_far = 0
        env._oracle_n_t = int(env.t)
    if int(env.t) == 0 or (far_work <= 1.0e-9 or near_work <= 1.0e-9):
        candidates = list(range(env.config.num_uavs + 1))
        if far_work <= 1.0e-9:
            candidates = [0]
        if near_work <= 1.0e-9:
            candidates = [env.config.num_uavs]
        best_n = candidates[0]
        best_val = -1.0e18
        for n_far in candidates:
            value = _shadow_payload(env, n_far)
            if value > best_val + 1.0e-9 or (abs(value - best_val) <= 1.0e-9 and n_far < best_n):
                best_val = value
                best_n = n_far
        env._oracle_n_far = int(best_n)
        env._oracle_n_t = int(env.t)
    return assign_split(env, int(env._oracle_n_far))


def hold_fixed(assignment: np.ndarray) -> Callable[[L2AEnv], np.ndarray]:
    frozen = np.asarray(assignment, dtype=np.int64).copy()

    def policy(_env: L2AEnv) -> np.ndarray:
        return frozen.copy()

    return policy


def rollout(env: L2AEnv, assign_fn: Callable[[L2AEnv], np.ndarray], seed: int) -> dict[str, float]:
    info = env.reset(seed=seed)
    agrees = []
    while not info.get("terminated", False):
        option = oracle_option_assignment(env)
        chosen = assign_fn(env)
        agrees.append(float(np.mean(np.asarray(chosen) == np.asarray(option))))
        info = env.step(fly_toward(env, chosen), chosen)
    return {
        "payload_bits": float(info["cumulative_payload_bits"]),
        "dropped_blocks": float(info["dropped_blocks"]),
        "residual_queue": float(np.sum(info["queue_blocks"])),
        "oracle_agreement": float(np.mean(agrees)) if agrees else float("nan"),
    }


__all__ = ["fly_toward", "hold_fixed", "myopic_assignment", "oracle_option_assignment", "rollout"]
