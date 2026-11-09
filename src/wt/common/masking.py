"""Masked reductions shared by the current models."""
from __future__ import annotations

from torch import Tensor


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)