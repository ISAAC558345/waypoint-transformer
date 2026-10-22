"""Chunked copy of the Phase-6I-D pairwise rank loss."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import softplus

from wt.benchmark_v2.phase6is_assignment_head.protocol import PAIR_CHUNK


def pairwise_rank_loss(scores: Tensor, labels: Tensor, chunk: int = PAIR_CHUNK) -> Tensor:
    """Mean of -log sigmoid(s_i - s_j) over pairs with J_i > J_j.

    Labels choose the pair direction and are detached. They are not features.
    """
    if scores.shape != labels.shape or scores.dim() != 1:
        raise ValueError("scores and J15 labels must be aligned vectors")
    compared = labels.detach()
    total = scores.new_zeros(())
    count = 0
    width = int(scores.shape[0])
    for start in range(0, width, int(chunk)):
        stop = min(width, start + int(chunk))
        gap = scores[start:stop, None] - scores[None, :]
        weight = (compared[start:stop, None] > compared[None, :]).to(dtype=scores.dtype)
        total = total + (softplus(-gap) * weight).sum()
        count += int(weight.sum().detach())
    if count == 0:
        return scores.sum() * 0.0
    return total / float(count)


__all__ = ["pairwise_rank_loss"]
