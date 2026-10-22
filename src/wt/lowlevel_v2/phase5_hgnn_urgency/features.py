"""Phase-5 graph batch. Urgency is an extra buoy channel, not a future feature."""
from __future__ import annotations

import numpy as np

from wt.htw_phase9.const import MAX_BUOYS
from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.l3b.env import L3BEnv

from .urgency import shared_task_urgency


def observable_exclusive_assignment(env: L3BEnv) -> np.ndarray:
    """Current-time exclusive match. Uses queue, distance, and hypothetical rate only."""
    info = env.info()
    rate = np.asarray(info["rate_if_assigned"], dtype=np.float64)
    dist = np.asarray(info["distance_uav_buoy"], dtype=np.float64)
    queue = np.asarray(env.queue, dtype=np.float64)
    scores = queue[None, :] * rate / (dist + 1.0)
    requested = np.full(env.config.num_uavs, -1, dtype=np.int64)
    taken: set[int] = set()
    order = np.argsort(-np.max(scores, axis=1), kind="stable")
    for uav in order:
        best = -1
        best_score = -1.0e18
        for buoy in range(int(env.config.num_buoys)):
            if buoy in taken or float(queue[buoy]) <= 1.0e-9:
                continue
            score = float(scores[int(uav), buoy])
            if score > best_score:
                best = int(buoy)
                best_score = score
        requested[int(uav)] = best
        if best >= 0:
            taken.add(best)
    return requested


def graph_batch(
    info: dict,
    env: L3BEnv,
    last_arrival: np.ndarray,
    last_positions: np.ndarray,
    *,
    urgency: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    batch = _batch_from_env(info, env, last_arrival, last_positions)
    if urgency is None:
        return batch
    shared = np.asarray(urgency, dtype=np.float32).reshape(-1)
    if shared.shape[0] != int(env.config.num_buoys):
        raise ValueError("urgency length must equal num_buoys")
    extra = np.zeros((1, MAX_BUOYS, 1), dtype=np.float32)
    extra[0, : shared.shape[0], 0] = shared
    batch["buoy"] = np.concatenate((np.asarray(batch["buoy"], dtype=np.float32), extra), axis=-1)
    return batch


def current_urgency(env: L3BEnv) -> np.ndarray:
    return shared_task_urgency(env.queue)


__all__ = ["current_urgency", "graph_batch", "observable_exclusive_assignment"]
