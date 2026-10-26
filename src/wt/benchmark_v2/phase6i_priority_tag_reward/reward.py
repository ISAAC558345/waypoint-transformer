"""Bounded urgency bonus on top of the formal low-level reward. The formal function is not edited."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import low_level_reward

from .protocol import LAMBDA_U, RHO


def capped_bonus(urgency_served_bits: float, served_bits: float, lambda_u: float, rho: float) -> float:
    """Bonus uses bits actually served. A zero-service step has a zero cap and a zero bonus."""
    served = float(served_bits)
    raw = float(lambda_u) * float(urgency_served_bits) / 1.0e6
    if served == 0.0:
        return 0.0
    cap = float(rho) * served / 1.0e6
    bonus = float(min(raw, cap))
    if bonus < 0.0 or bonus - cap > 1.0e-12:
        raise RuntimeError("capped bonus left its predeclared interval")
    return bonus


def experiment_reward(
    served_bits: float,
    urgency_served_bits: float,
    executed: np.ndarray,
    desired: np.ndarray,
    delta: np.ndarray,
    block_bits: float,
    *,
    lambda_u: float,
    rho: float,
) -> tuple[float, float, float]:
    base, communication = low_level_reward(float(served_bits), executed, desired, delta, float(block_bits))
    bonus = capped_bonus(float(urgency_served_bits), float(served_bits), float(lambda_u), float(rho))
    return float(base + bonus), float(base), float(bonus)


def l2_reward(served_bits: float, urgency_served_bits: float, executed, desired, delta, block_bits: float) -> tuple[float, float, float]:
    return experiment_reward(
        served_bits,
        urgency_served_bits,
        executed,
        desired,
        delta,
        block_bits,
        lambda_u=LAMBDA_U,
        rho=RHO,
    )
