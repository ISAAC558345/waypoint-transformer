"""Torch copy of ServiceResidualLock + BaseZeroResidualFreeze + residual cap. Formula frozen."""
from __future__ import annotations

import torch

from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL
from wt.lowlevel_v2.phase2b.trust_region import BETA_RESIDUAL, EPSILON_BASE_FRACTION, action_max

NORM_EPS = 1.0e-12


def residual_mean_used(
    mu: torch.Tensor,
    a_base: torch.Tensor,
    service_lock: torch.Tensor,
    base_zero: torch.Tensor,
    *,
    alpha: float = ALPHA_RESIDUAL,
    beta: float = BETA_RESIDUAL,
    eps_frac: float = EPSILON_BASE_FRACTION,
) -> torch.Tensor:
    r = float(alpha) * torch.tanh(mu)
    lock = service_lock.reshape(-1, 1)
    freeze = base_zero.reshape(-1, 1)
    r = r * (1.0 - lock) * (1.0 - freeze)
    base_norm = torch.linalg.norm(a_base, dim=-1)
    cap = float(beta) * base_norm
    r_norm = torch.linalg.norm(r, dim=-1)
    over = (r_norm > cap) & (service_lock < 0.5) & (base_zero < 0.5)
    unit = r / r_norm.clamp(min=NORM_EPS).unsqueeze(-1)
    r = torch.where(over.unsqueeze(-1), unit * cap.unsqueeze(-1), r)
    eps_b = float(eps_frac) * float(action_max())
    freeze_now = (base_norm <= eps_b).unsqueeze(-1)
    r = torch.where(freeze_now, torch.zeros_like(r), r)
    return r


__all__ = ["residual_mean_used"]
