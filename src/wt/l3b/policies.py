"""Non-learning association scripts for Level-3-B SINR probes."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .env import L3BEnv, conflict_mask


def fly_toward(env: L3BEnv, assignment: np.ndarray) -> np.ndarray:
    move = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0:
            continue
        delta = env.buoy_positions[int(buoy)] - env.uav_positions[uav]
        norm = float(np.linalg.norm(delta))
        if norm > 1.0e-9:
            move[uav] = delta / norm
    return move


def myopic_assignment(env: L3BEnv) -> np.ndarray:
    requested = np.full(env.config.num_uavs, -1, dtype=np.int64)
    for uav in range(env.config.num_uavs):
        best, best_score = -1, -1.0e18
        for buoy in range(env.config.num_buoys):
            if env.queue[buoy] <= 1.0e-9:
                continue
            dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[buoy]))
            score = float(env.queue[buoy]) / (dist + 1.0)
            if score > best_score:
                best, best_score = buoy, score
        requested[uav] = best
    return requested


def exclusive_matching(env: L3BEnv) -> np.ndarray:
    """Sequential exclusive Q/distance matching. Ignores SINR."""
    requested = np.full(env.config.num_uavs, -1, dtype=np.int64)
    taken: set[int] = set()
    for uav in range(env.config.num_uavs):
        best, best_score = -1, -1.0e18
        for buoy in range(env.config.num_buoys):
            if buoy in taken or env.queue[buoy] <= 1.0e-9:
                continue
            dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[buoy]))
            score = float(env.queue[buoy]) / (dist + 1.0)
            if score > best_score:
                best, best_score = buoy, score
        requested[uav] = best
        if best >= 0:
            taken.add(int(best))
    return requested


def _cluster_work(env: L3BEnv) -> tuple[float, float]:
    leftover = env.remaining_arrivals()
    work = env.queue + leftover
    return float(np.sum(work[:6])), float(np.sum(work[6:]))


def assign_split(env: L3BEnv, n_near: int, n_far: int) -> np.ndarray:
    n_near = int(np.clip(n_near, 0, env.config.num_uavs))
    n_far = int(np.clip(n_far, 0, env.config.num_uavs - n_near))
    far_center = np.mean(env.buoy_positions[6:], axis=0)
    near_center = np.mean(env.buoy_positions[:6], axis=0)
    dist_far = np.linalg.norm(env.uav_positions - far_center, axis=-1)
    far_uavs = set(np.argsort(dist_far, kind="stable")[:n_far].tolist())
    rest = [uav for uav in range(env.config.num_uavs) if uav not in far_uavs]
    dist_near = np.array([float(np.linalg.norm(env.uav_positions[uav] - near_center)) for uav in rest], dtype=np.float64)
    near_order = np.argsort(dist_near, kind="stable")[:n_near]
    near_uavs = {int(rest[int(index)]) for index in near_order}
    requested = np.full(env.config.num_uavs, -1, dtype=np.int64)
    taken: set[int] = set()
    leftover = env.remaining_arrivals()
    for uav in range(env.config.num_uavs):
        if uav in far_uavs:
            cluster = range(6, 12)
        elif uav in near_uavs:
            cluster = range(0, 6)
        else:
            continue
        best, best_score = -1, -1.0e18
        for buoy in cluster:
            if buoy in taken:
                continue
            if env.queue[buoy] <= 1.0e-9 and float(leftover[buoy]) <= 1.0e-9:
                continue
            dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[buoy]))
            score = float(max(env.queue[buoy], leftover[buoy])) / (dist + 1.0)
            if score > best_score:
                best, best_score = buoy, score
        requested[uav] = best
        if best >= 0:
            taken.add(int(best))
    return requested


def _shadow_payload(env: L3BEnv, n_near: int, n_far: int) -> float:
    import copy

    clone = copy.deepcopy(env)
    clone._skip_graph = True
    start = float(clone.payload_bits)
    remaining = int(clone.config.episode_horizon - clone.t)
    for _ in range(max(0, remaining)):
        chosen = assign_split(clone, n_near, n_far)
        clone.step(fly_toward(clone, chosen), chosen)
        if clone.t >= int(clone.config.episode_horizon):
            break
    return float(clone.payload_bits - start)


def oracle_option_assignment(env: L3BEnv) -> np.ndarray:
    """Hold a (n_near, n_far, idle) option chosen by full-horizon SINR-aware shadow search."""
    near_work, far_work = _cluster_work(env)
    if not hasattr(env, "_oracle_split") or int(getattr(env, "_oracle_n_t", -1)) > int(env.t):
        env._oracle_split = (1, 1)
        env._oracle_n_t = int(env.t)
    if int(env.t) == 0 or (far_work <= 1.0e-9 or near_work <= 1.0e-9):
        best_split = (1, 0)
        best_val = -1.0e18
        for n_near in range(env.config.num_uavs + 1):
            for n_far in range(env.config.num_uavs - n_near + 1):
                if far_work <= 1.0e-9 and n_far > 0:
                    continue
                if near_work <= 1.0e-9 and n_near > 0:
                    continue
                value = _shadow_payload(env, n_near, n_far)
                split = (n_near, n_far)
                if value > best_val + 1.0e-9 or (abs(value - best_val) <= 1.0e-9 and split < best_split):
                    best_val = value
                    best_split = split
        env._oracle_split = best_split
        env._oracle_n_t = int(env.t)
        env._oracle_shadow_payload = float(best_val)
    n_near, n_far = env._oracle_split
    return assign_split(env, int(n_near), int(n_far))


def _held_fn(assign_fn: Callable[[L3BEnv], np.ndarray], hold: int) -> Callable[[L3BEnv], np.ndarray]:
    state: dict[str, np.ndarray | int] = {"z": np.array([], dtype=np.int64), "until": -1}

    def policy(env: L3BEnv) -> np.ndarray:
        if int(env.t) >= int(state["until"]):
            state["z"] = np.asarray(assign_fn(env), dtype=np.int64).copy()
            state["until"] = int(env.t) + int(hold)
        return np.asarray(state["z"], dtype=np.int64).copy()

    return policy


def rollout(
    env: L3BEnv,
    assign_fn: Callable[[L3BEnv], np.ndarray],
    seed: int,
    hold: int | None = None,
    compare_oracle: bool = True,
) -> dict[str, float]:
    info = env.reset(seed=seed)
    env._skip_graph = True
    hold_steps = int(env.config.option_hold if hold is None else hold)
    policy = assign_fn if hold_steps <= 1 else _held_fn(assign_fn, hold_steps)
    agrees = []
    conflict_rates = []
    mean_rates = []
    n_active = []
    while not info.get("terminated", False):
        chosen = policy(env)
        if compare_oracle:
            option = oracle_option_assignment(env)
            agrees.append(float(np.mean(np.asarray(chosen) == np.asarray(option))))
        conflict_rates.append(float(np.mean(conflict_mask(chosen, env.config.num_buoys))))
        n_active.append(float(np.mean(np.asarray(chosen) >= 0)))
        info = env.step(fly_toward(env, chosen), chosen)
        serving = np.asarray(info["rate_bps"], dtype=np.float64)
        mean_rates.append(float(np.mean(serving)))
    return {
        "payload_bits": float(info["cumulative_payload_bits"]),
        "dropped_blocks": float(info["dropped_blocks"]),
        "residual_queue": float(np.sum(info["queue_blocks"])),
        "oracle_agreement": float(np.mean(agrees)) if agrees else float("nan"),
        "mean_conflict_rate": float(np.mean(conflict_rates)) if conflict_rates else 0.0,
        "mean_rate_bps": float(np.mean(mean_rates)) if mean_rates else 0.0,
        "mean_active_frac": float(np.mean(n_active)) if n_active else 0.0,
        "hold": float(hold_steps),
        "oracle_n_near": float(getattr(env, "_oracle_split", (-1, -1))[0]),
        "oracle_n_far": float(getattr(env, "_oracle_split", (-1, -1))[1]),
    }


__all__ = [
    "assign_split",
    "exclusive_matching",
    "fly_toward",
    "myopic_assignment",
    "oracle_option_assignment",
    "rollout",
]
