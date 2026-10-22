"""Official residual reward. Phase-5 does not add an urgency term."""
from __future__ import annotations

import numpy as np

from wt.lowlevel_v2.phase1b.protocol import ACTION_SCALE, LAMBDA_PROJECT, LAMBDA_RESIDUAL, PAYLOAD_SCALE


def unchanged_step_reward(payload_inc: float, a_exec: np.ndarray, a_des: np.ndarray, delta_used: np.ndarray) -> float:
    r_payload = float(payload_inc) / float(PAYLOAD_SCALE)
    c_proj = float(np.mean(np.sum((a_exec - a_des) ** 2, axis=-1) / (ACTION_SCALE ** 2 + 1.0e-8)))
    c_res = float(np.mean(np.sum(delta_used ** 2, axis=-1) / (ACTION_SCALE ** 2 + 1.0e-8)))
    reward = r_payload - LAMBDA_PROJECT * c_proj - LAMBDA_RESIDUAL * c_res
    if not np.isfinite(reward):
        raise RuntimeError("non-finite train reward")
    return float(reward)


__all__ = ["unchanged_step_reward"]
