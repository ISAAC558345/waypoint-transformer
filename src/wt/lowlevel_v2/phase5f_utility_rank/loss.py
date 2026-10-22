"""Pairwise logistic ranking on utility_rank. Lower rank is the better target."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import logsigmoid


def pairwise_rank_loss(scores: Tensor, ranks: list[int]) -> Tensor:
    """Mean of -log sigmoid(s_better - s_worse) over ordered utility pairs."""
    if int(scores.shape[0]) != len(ranks):
        raise ValueError("scores and ranks must describe the same candidates")
    terms = []
    for left, left_rank in enumerate(ranks):
        for right, right_rank in enumerate(ranks):
            if int(left_rank) < int(right_rank):
                terms.append(-logsigmoid(scores[left] - scores[right]))
    if not terms:
        return scores.sum() * 0.0
    return torch.stack(terms).mean()


__all__ = ["pairwise_rank_loss"]
