"""Pre-registered Stage-1 scores. J15 is not an input."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import geometry, hypothetical_rate
from wt.benchmark_v2.phase6d.pool import enumerate_legal

from .protocol import GENERATORS


def _unit(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(values)) if values.size else 0.0
    if scale <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return np.asarray(values, dtype=np.float64) / scale


def feature_matrices(env: L3BBitQueueEnvV2) -> dict[str, np.ndarray]:
    """Current observables only. Arrivals and future queues are not read."""
    geo = geometry(env)
    rate = hypothetical_rate(env, geo)
    distance = np.asarray(geo["distance"], dtype=np.float64)
    radius = float(env.config.service_radius_m)
    step = max(float(env.config.step_length_m), 1.0e-9)
    eta = np.maximum(0.0, (distance - radius) / step)
    return {
        "rate": rate,
        "distance": distance,
        "eta": eta,
        "queue": np.asarray(env.q_bits, dtype=np.float64),
        "in_range": np.asarray(geo["in_range"], dtype=bool),
        "gain": geo["gain"],
    }


def proposal_scores(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rate_hat = _unit(features["rate"])
    distance_hat = _unit(features["distance"])
    queue_hat = _unit(features["queue"])
    eta = np.asarray(features["eta"], dtype=np.float64)
    queue_grid = np.broadcast_to(queue_hat, rate_hat.shape)
    reachable = 1.0 + eta
    scores = {
        "G1_RATE": rate_hat,
        "G2_DISTANCE": -distance_hat,
        "G3_ETA": -eta,
        "G4_QUEUE_RATE": queue_grid * rate_hat,
        "G5_REACHABLE_RATE": rate_hat / reachable,
        "G6_REACHABLE_QUEUE_RATE": queue_grid * rate_hat / reachable,
    }
    return {name: scores[name] for name in GENERATORS}


def top_k(score: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-np.asarray(score, dtype=np.float64), axis=1, kind="mergesort")
    return order[:, : int(k)]


def is_legal(assignment: np.ndarray, n_uavs: int, n_buoys: int) -> bool:
    chosen = np.asarray(assignment, dtype=np.int64).reshape(-1)
    if int(chosen.size) != int(n_uavs):
        return False
    seen: set[int] = set()
    for buoy in chosen:
        buoy_id = int(buoy)
        if buoy_id < -1 or buoy_id >= int(n_buoys):
            return False
        if buoy_id < 0:
            continue
        if buoy_id in seen:
            return False
        seen.add(buoy_id)
    return True


def build_pools(env: L3BBitQueueEnvV2, rules: tuple[tuple[str, int], ...]) -> tuple[dict, dict, int]:
    features = feature_matrices(env)
    scores = proposal_scores(features)
    max_k: dict[str, int] = {}
    wanted: dict[str, set[int]] = {}
    for name, k in rules:
        max_k[name] = max(max_k.get(name, 0), int(k))
        wanted.setdefault(name, set()).add(int(k))
    pools: dict[tuple[str, int], list[np.ndarray]] = {}
    tops: dict[tuple[str, int], np.ndarray] = {}
    illegal = 0
    n_uavs = int(env.config.num_uavs)
    n_buoys = int(env.config.num_buoys)
    for name, limit in max_k.items():
        ranked = top_k(scores[name], limit)
        if np.any(ranked < 0):
            illegal += int(np.sum(ranked < 0))
        for k in sorted(wanted[name]):
            chosen = ranked[:, : int(k)]
            assignments = enumerate_legal(chosen)
            for assignment in assignments:
                if not is_legal(assignment, n_uavs, n_buoys):
                    illegal += 1
            pools[(name, int(k))] = assignments
            tops[(name, int(k))] = chosen
    return pools, {"features": features, "tops": tops}, int(illegal)


__all__ = ["build_pools", "feature_matrices", "is_legal", "proposal_scores", "top_k"]
