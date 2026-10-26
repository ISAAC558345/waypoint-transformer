"""L1 calls the formal reward. L2 adds the one pre-registered urgency term."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import low_level_reward

from .protocol import LAMBDA_URGENCY


def l1_reward(served_bits, executed, desired, delta, block_bits) -> dict[str, float]:
    served = np.asarray(served_bits, dtype=np.float64).reshape(-1)
    reward, communication = low_level_reward(float(np.sum(served)), executed, desired, delta, block_bits)
    return {
        "reward": float(reward),
        "communication_reward": float(communication),
        "urgency_reward": 0.0,
        "urgent_service_bits": 0.0,
        "lambda_urgency": 0.0,
    }


def l2_reward(served_bits, urgency, executed, desired, delta, block_bits) -> dict[str, float]:
    served = np.asarray(served_bits, dtype=np.float64).reshape(-1)
    weights = np.asarray(urgency, dtype=np.float64).reshape(-1)
    if served.shape != weights.shape:
        raise RuntimeError("urgency and served bits are not aligned by buoy")
    reward, communication = low_level_reward(float(np.sum(served)), executed, desired, delta, block_bits)
    urgent_bits = float(np.sum(weights * served))
    urgent_term = float(LAMBDA_URGENCY) * urgent_bits / float(block_bits)
    return {
        "reward": float(reward) + float(urgent_term),
        "communication_reward": float(communication),
        "urgency_reward": float(urgent_term),
        "urgent_service_bits": urgent_bits,
        "lambda_urgency": float(LAMBDA_URGENCY),
    }
