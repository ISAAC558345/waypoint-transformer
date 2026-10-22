"""Stage-1 style candidate pools. Optima inside a pool are not global oracles."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import geometry, hypothetical_rate


def top_targets(env: L3BBitQueueEnvV2, k: int, ranking: str = "instantaneous_rate") -> np.ndarray:
    geo = geometry(env)
    if ranking == "instantaneous_rate":
        score = hypothetical_rate(env, geo)
    elif ranking == "queue":
        score = np.repeat(env.q_bits[None, :], env.config.num_uavs, axis=0)
    else:
        raise RuntimeError(f"unknown ranking {ranking}")
    order = np.argsort(-np.asarray(score, dtype=np.float64), axis=1, kind="mergesort")
    return order[:, : int(k)]


def enumerate_legal(options: np.ndarray) -> list[np.ndarray]:
    """Partial injections. Idle is always legal. Duplicate buoys are not."""
    n_uavs = int(options.shape[0])
    current = np.full(n_uavs, -1, dtype=np.int64)
    used = np.zeros(int(np.max(options)) + 1 if options.size else 1, dtype=bool)
    found: list[np.ndarray] = []

    def rec(uav: int) -> None:
        if uav == n_uavs:
            found.append(current.copy())
            return
        current[uav] = -1
        rec(uav + 1)
        for buoy in options[uav]:
            buoy_id = int(buoy)
            if buoy_id < 0 or bool(used[buoy_id]):
                continue
            used[buoy_id] = True
            current[uav] = buoy_id
            rec(uav + 1)
            used[buoy_id] = False

    rec(0)
    return found


def candidate_pool(env: L3BBitQueueEnvV2, k: int, ranking: str = "instantaneous_rate") -> tuple[list[np.ndarray], int]:
    assignments = enumerate_legal(top_targets(env, k, ranking))
    return assignments, len(assignments)


__all__ = ["candidate_pool", "enumerate_legal", "top_targets"]
