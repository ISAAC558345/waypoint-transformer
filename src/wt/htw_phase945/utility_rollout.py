"""H=15 frozen-physics shadow utility. Not a global Oracle."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from wt.htw_phase9.const import HOLD
from wt.htw_phase9.features import in_range_completion, interference_utility
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward


def shadow_h15(env: L3BEnv, assignment: np.ndarray) -> dict[str, float]:
    clone = copy.deepcopy(env)
    start = float(clone.payload_bits)
    start_queue = float(np.sum(clone.queue))
    rates = []
    chosen = np.asarray(assignment, dtype=np.int64)
    idle = float(np.mean(chosen[: clone.config.num_uavs] < 0))
    for _ in range(int(HOLD)):
        if clone.t >= int(clone.config.episode_horizon):
            break
        after = clone.step(fly_toward(clone, chosen), chosen)
        rates.append(interference_utility(after))
        if after.get("terminated", False):
            break
    payload = float(clone.payload_bits - start)
    queue_red = start_queue - float(np.sum(clone.queue))
    complete = in_range_completion(clone, chosen)
    return {
        "payload_increment": payload,
        "sinr_utility": float(np.mean(rates)) if rates else 0.0,
        "mean_rate": float(np.mean(rates)) if rates else 0.0,
        "queue_reduction": float(queue_red),
        "idle": idle,
        "completion": float(complete),
    }


def shadow_h15_payload(env: L3BEnv, assignment: np.ndarray) -> float:
    return float(shadow_h15(env, assignment)["payload_increment"])


__all__ = ["shadow_h15", "shadow_h15_payload"]
