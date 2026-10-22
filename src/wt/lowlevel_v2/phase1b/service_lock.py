"""ServiceResidualLock. Uses existing env in-range / envelope_active. No second radius."""
from __future__ import annotations

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.contracts import clip_unit_step
from wt.lowlevel_v2.service_safe_envelope import envelope_active


def service_lock_mask(env: L3BEnv, option: TaskOption) -> np.ndarray:
    """True iff target valid, not idle, queue>0, UAV currently in service region.

    Evaluated on pre-action geometry. Delegates to envelope_active, which uses env._in_range.
    """
    return np.asarray(envelope_active(env, option), dtype=bool)


def apply_service_residual_lock(
    a_base: np.ndarray,
    delta_rl: np.ndarray,
    lock: np.ndarray,
    alpha_residual: float,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(a_base, dtype=np.float64)
    raw = np.tanh(np.asarray(delta_rl, dtype=np.float64))
    delta_used = float(alpha_residual) * raw
    lock_b = np.asarray(lock, dtype=bool).reshape(-1)
    if lock_b.shape[0] != base.shape[0]:
        raise ValueError("lock mask length must match UAV count")
    delta_used = np.where(lock_b[:, None], 0.0, delta_used)
    a_des = clip_unit_step(base + delta_used)
    return a_des, delta_used


__all__ = ["apply_service_residual_lock", "service_lock_mask"]
