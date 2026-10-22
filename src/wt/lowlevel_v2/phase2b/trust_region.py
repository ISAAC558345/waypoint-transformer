"""Base-relative residual trust region. Magnitude only; does not change residual direction."""
from __future__ import annotations

import numpy as np

from wt.lowlevel_v2.phase1.protocol import ACTION_SCALE

BETA_RESIDUAL = 0.25
EPSILON_BASE_FRACTION = 0.05
NORM_EPS = 1.0e-12


def action_max() -> float:
    """Unit-step action contract used by clip_unit_step / ACTION_SCALE. Not re-defined."""
    return float(ACTION_SCALE)


def epsilon_base() -> float:
    return float(EPSILON_BASE_FRACTION) * action_max()


def apply_base_relative_residual_trust_region(
    a_base: np.ndarray,
    r_after_lock: np.ndarray,
    *,
    beta: float = BETA_RESIDUAL,
    eps_frac: float = EPSILON_BASE_FRACTION,
    a_max: float | None = None,
) -> tuple[np.ndarray, dict]:
    base = np.asarray(a_base, dtype=np.float64)
    r_in = np.asarray(r_after_lock, dtype=np.float64)
    if base.shape != r_in.shape or base.ndim != 2 or base.shape[-1] != 2:
        raise ValueError("base and residual must have shape (M, 2)")
    scale_max = float(action_max() if a_max is None else a_max)
    eps_b = float(eps_frac) * scale_max
    base_norm = np.linalg.norm(base, axis=-1)
    freeze = base_norm <= eps_b
    r = r_in.copy()
    r[freeze] = 0.0
    cap = float(beta) * base_norm
    r_norm = np.linalg.norm(r, axis=-1)
    over = (~freeze) & (r_norm > cap)
    unit = r / np.clip(r_norm[:, None], NORM_EPS, None)
    r_used = np.where(over[:, None], unit * cap[:, None], r)
    r_used[freeze] = 0.0
    return r_used, {
        "r_raw": r_in,
        "cap_active": over,
        "base_zero_lock": freeze,
        "base_norm": base_norm,
        "residual_cap": cap,
        "epsilon_base": eps_b,
        "a_max": scale_max,
        "beta": float(beta),
    }


__all__ = [
    "BETA_RESIDUAL",
    "EPSILON_BASE_FRACTION",
    "action_max",
    "apply_base_relative_residual_trust_region",
    "epsilon_base",
]
