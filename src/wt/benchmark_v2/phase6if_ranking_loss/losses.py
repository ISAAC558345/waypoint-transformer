"""Phase-6I-F losses. They read scores and J15 only, never the graph."""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn.functional import log_softmax, softplus, softmax


GAP_SCALE_BITS = 1.0e6
TOLERANCE_BITS = 1.0e4
MARGIN_CAP = 1.0
LISTNET_BETA = 30.0
CHUNK = 1024


def _pairs(scores: Tensor, labels: Tensor, start: int, stop: int) -> tuple[Tensor, Tensor]:
    block = scores[start:stop]
    block_labels = labels[start:stop]
    return block[:, None] - scores[None, :], block_labels[:, None] - labels[None, :]


def gap_weighted_loss(scores: Tensor, labels: Tensor) -> Tensor:
    """Scheme A. Weight is log1p(J15 gap / 1e6), then normalized inside the state."""
    labels = labels.detach().to(dtype=scores.dtype)
    total_weight = scores.new_zeros(())
    total_loss = scores.new_zeros(())
    count = int(scores.shape[0])
    for start in range(0, count, CHUNK):
        gap_s, gap_y = _pairs(scores, labels, start, min(count, start + CHUNK))
        positive = gap_y > 0
        weight = torch.log1p(gap_y.clamp_min(0) / GAP_SCALE_BITS) * positive.to(dtype=scores.dtype)
        total_weight = total_weight + weight.sum()
        total_loss = total_loss + (weight.detach() * softplus(-gap_s)).sum()
    return total_loss / total_weight.clamp_min(1.0e-12)


def tolerance_margin_loss(scores: Tensor, labels: Tensor) -> Tensor:
    """Scheme B. Gaps at or below 1e4 bit are ties. Larger gaps use a score-unit margin."""
    labels = labels.detach().to(dtype=scores.dtype)
    total_count = scores.new_zeros(())
    total_loss = scores.new_zeros(())
    normalizer = math.log1p(20.0)
    count = int(scores.shape[0])
    for start in range(0, count, CHUNK):
        gap_s, gap_y = _pairs(scores, labels, start, min(count, start + CHUNK))
        active = gap_y > TOLERANCE_BITS
        margin = (torch.log1p(gap_y.clamp_min(0) / GAP_SCALE_BITS) / normalizer).clamp(0.0, MARGIN_CAP)
        terms = softplus(margin.detach() - gap_s)
        chosen = active.to(dtype=scores.dtype)
        total_count = total_count + chosen.sum()
        total_loss = total_loss + (chosen.detach() * terms).sum()
    return total_loss / total_count.clamp_min(1.0)


def top_regret_loss(scores: Tensor, labels: Tensor) -> Tensor:
    """Scheme C. Match a label-derived target. The target is not a model input."""
    labels = labels.detach().to(dtype=scores.dtype)
    span = (labels.max() - labels.min()).clamp_min(1.0e-6)
    quality = (labels - labels.min()) / span
    target = softmax(LISTNET_BETA * (quality - quality.max()), dim=0)
    return -(target.detach() * log_softmax(scores, dim=0)).sum()


LOSSES = {
    "a": gap_weighted_loss,
    "b": tolerance_margin_loss,
    "c": top_regret_loss,
}


__all__ = [
    "CHUNK",
    "GAP_SCALE_BITS",
    "LISTNET_BETA",
    "LOSSES",
    "MARGIN_CAP",
    "TOLERANCE_BITS",
    "gap_weighted_loss",
    "tolerance_margin_loss",
    "top_regret_loss",
]
