"""Action maps for the frozen residual baseline and the direct velocity command.

The residual map is the Phase-6G formula. The direct map replaces it.
Neither function reads future arrivals or changes an assignment.
"""
from __future__ import annotations

import math

import numpy as np

from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import clip_unit_step
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL


def residual_limits(alpha: float = ALPHA_RESIDUAL) -> dict[str, float]:
    """Closed-form reachable set of clip(unit_target + alpha * tanh(pre))."""
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise ValueError("residual alpha must lie in (0, 1)")
    heading = math.degrees(math.atan2(value, 1.0 - value))
    speed = math.hypot(1.0 - value, value)
    return {
        "alpha": value,
        "max_heading_deg": float(heading),
        "min_speed_fraction": float(1.0 - value),
        "speed_at_max_heading": float(speed),
        "can_stop": False,
        "can_reverse": False,
        "max_step_fraction": 1.0,
    }


def residual_command(env, association: np.ndarray, pre_tanh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (unit-step command, residual delta) using the frozen Phase-6G map."""
    pre = _as_command(pre_tanh)
    if not np.all(np.isfinite(pre)):
        zeros = np.zeros_like(pre)
        return zeros, zeros
    delta = float(ALPHA_RESIDUAL) * np.tanh(pre)
    command = clip_unit_step(fly_toward(env, association) + delta)
    return command, delta


def direct_command(pre_tanh: np.ndarray) -> np.ndarray:
    """World-frame unit-step velocity. Component 0 is +x, component 1 is +y.

    tanh bounds each axis to (-1, 1). clip_unit_step then caps the displacement
    at one environment step. The zero vector is a legal hover.
    """
    pre = _as_command(pre_tanh)
    if not np.all(np.isfinite(pre)):
        return np.zeros_like(pre)
    return clip_unit_step(np.tanh(np.clip(pre, -20.0, 20.0)))


def direct_reward_terms(commanded: np.ndarray, executed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feed the existing reward slots without adding a new term.

    Projection compares the executed step with the commanded velocity.
    The residual slot receives the commanded velocity, which is the motion
    the actor actually chose.
    """
    command = np.asarray(commanded, dtype=np.float64)
    done = np.asarray(executed, dtype=np.float64)
    return done, command, command


def _as_command(pre_tanh: np.ndarray) -> np.ndarray:
    pre = np.asarray(pre_tanh, dtype=np.float64)
    if pre.ndim != 2 or pre.shape[1] != 2:
        raise ValueError("movement pre-activation must have shape (M, 2)")
    return pre


__all__ = [
    "direct_command",
    "direct_reward_terms",
    "residual_command",
    "residual_limits",
]
