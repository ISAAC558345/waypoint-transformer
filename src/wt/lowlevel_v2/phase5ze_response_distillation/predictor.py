"""Set-equivariant one-step response predictor. The value head is not in this module."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .protocol import BINARY_FEATURES, HIDDEN, HUBER_DELTA, RESPONSE_FEATURES


def _indexes(names: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(RESPONSE_FEATURES.index(name) for name in names)


CONTINUOUS_INDEX = _indexes(tuple(name for name in RESPONSE_FEATURES if name not in BINARY_FEATURES))
BINARY_INDEX = _indexes(BINARY_FEATURES)


class OneStepExecutorResponsePredictor(nn.Module):
    """Local MLP, mean over UAVs, then a response MLP. Hidden width stays 128."""

    def __init__(self, input_dim: int, width: int = HIDDEN, response_dim: int = len(RESPONSE_FEATURES)) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.response_dim = int(response_dim)
        self.local = nn.Sequential(
            nn.Linear(self.input_dim, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
            nn.SiLU(),
        )
        self.response = nn.Sequential(
            nn.Linear(self.width * 2, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.response_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.dim() != 3:
            raise ValueError("tokens must be assignment by UAV by feature")
        local = self.local(tokens)
        context = local.mean(dim=1, keepdim=True).expand_as(local)
        return self.response(torch.cat((local, context), dim=-1))


def response_loss(prediction: Tensor, normalized_target: Tensor, binary_target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Equal-weight mean Huber on normalized continuous channels and mean BCE on binary labels."""
    if prediction.shape != normalized_target.shape:
        raise ValueError("prediction and response target must share assignment, UAV, and feature axes")
    continuous = prediction[..., list(CONTINUOUS_INDEX)]
    continuous_target = normalized_target[..., list(CONTINUOUS_INDEX)]
    binary = prediction[..., list(BINARY_INDEX)]
    huber = F.huber_loss(continuous, continuous_target, reduction="mean", delta=float(HUBER_DELTA))
    bce = F.binary_cross_entropy_with_logits(binary, binary_target, reduction="mean")
    return huber + bce, huber, bce


__all__ = [
    "BINARY_INDEX",
    "CONTINUOUS_INDEX",
    "OneStepExecutorResponsePredictor",
    "response_loss",
]
