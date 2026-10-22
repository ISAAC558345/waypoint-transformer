"""Residual desired action. Zero residual equals the L0 base action."""
from __future__ import annotations

import numpy as np

from .contracts import ResidualConfig, clip_unit_step
from .protocol import ALPHA_RESIDUAL_PLACEHOLDER


def residual_desired(
    a_base: np.ndarray,
    delta_rl: np.ndarray,
    config: ResidualConfig | None = None,
) -> np.ndarray:
    cfg = config or ResidualConfig(alpha_residual=float(ALPHA_RESIDUAL_PLACEHOLDER), placeholder=True)
    base = np.asarray(a_base, dtype=np.float64)
    delta = np.tanh(np.asarray(delta_rl, dtype=np.float64))
    if delta.shape != base.shape:
        raise ValueError("residual shape must match base action")
    return clip_unit_step(base + float(cfg.alpha_residual) * delta)


__all__ = ["residual_desired"]
