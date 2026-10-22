"""Pairwise ranking on decoder scores. The order comes from J(A)."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import logsigmoid


def joint_rank_loss(scores: Tensor, joints: Tensor) -> Tensor:
    """Mean of -log sigmoid(s(A) - s(B)) over pairs with J(A) > J(B).

    The written difference of the labels is constant with respect to the
    decoder. The loss compares decoder scores and uses J only to choose
    which assignment must rank higher.
    """
    if scores.shape != joints.shape or scores.dim() != 1:
        raise ValueError("scores and joints must be aligned vectors")
    better = joints[:, None] > joints[None, :]
    gap = scores[:, None] - scores[None, :]
    terms = -logsigmoid(gap)[better]
    if int(terms.numel()) == 0:
        return scores.sum() * 0.0
    return terms.mean()


__all__ = ["joint_rank_loss"]
