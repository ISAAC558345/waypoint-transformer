"""Pairwise interaction added to a frozen unary assignment score.

S(A) = S_unary(A) + lambda_pair * sum_{i<j} g(e_i, e_j).
g reads the symmetric triple [e_i + e_j, |e_i - e_j|, e_i * e_j].
Idle UAVs are left out of every pair. lambda_pair is not a parameter.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wt.benchmark_v2.phase6iu_pairwise_head.protocol import CANDIDATE_CHUNK, HIDDEN, LAMBDA_PAIR, WIDTH


def symmetric_pair_features(left: Tensor, right: Tensor) -> Tensor:
    """Order-invariant features of two selected edge states."""
    return torch.cat((left + right, (left - right).abs(), left * right), dim=-1)


class PairwiseInteractionHead(nn.Module):
    def __init__(self, width: int = WIDTH, hidden: int = HIDDEN, chunk: int = CANDIDATE_CHUNK) -> None:
        super().__init__()
        self.width = int(width)
        self.chunk = int(chunk)
        self.g = nn.Sequential(
            nn.Linear(int(width) * 3, int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), 1),
        )
        left, right = [], []
        for first in range(6):
            for second in range(first + 1, 6):
                left.append(first)
                right.append(second)
        self.register_buffer("pair_left", torch.tensor(left, dtype=torch.long), persistent=False)
        self.register_buffer("pair_right", torch.tensor(right, dtype=torch.long), persistent=False)
        self.register_buffer("lambda_pair", torch.tensor(float(LAMBDA_PAIR)), persistent=True)

    def pair_score(self, edge_h: Tensor, assignments: Tensor) -> Tensor:
        """Sum of symmetric pair scores. edge_h is [6, buoys, width]."""
        if edge_h.dim() != 3 or int(edge_h.shape[0]) != 6 or int(edge_h.shape[-1]) != self.width:
            raise ValueError("edge states must be [6, buoys, width] for one decision")
        if assignments.dim() != 2 or int(assignments.shape[1]) != 6:
            raise ValueError("assignments must be [count, 6]")
        pieces = []
        count = int(assignments.shape[0])
        for start in range(0, count, self.chunk):
            pieces.append(self._chunk(edge_h, assignments[start : start + self.chunk]))
        return torch.cat(pieces, dim=0)

    def forward(self, edge_h: Tensor, assignments: Tensor, unary: Tensor) -> Tensor:
        if unary.shape != assignments.shape[:1]:
            raise ValueError("unary scores must align with assignments")
        return unary.detach() + self.lambda_pair * self.pair_score(edge_h, assignments)

    def _chunk(self, edge_h: Tensor, assignments: Tensor) -> Tensor:
        valid = assignments >= 0
        safe = assignments.clamp(min=0)
        batch, uav_count = int(assignments.shape[0]), int(assignments.shape[1])
        uav_index = torch.arange(uav_count, device=assignments.device).unsqueeze(0).expand(batch, uav_count)
        edges = edge_h[uav_index, safe] * valid.unsqueeze(-1).to(edge_h.dtype)
        left = edges[:, self.pair_left]
        right = edges[:, self.pair_right]
        features = symmetric_pair_features(left, right)
        values = self.g(features.reshape(batch * int(self.pair_left.shape[0]), -1)).reshape(batch, -1)
        mask = valid[:, self.pair_left] & valid[:, self.pair_right]
        return (values * mask.to(values.dtype)).sum(dim=1)


__all__ = ["PairwiseInteractionHead", "symmetric_pair_features"]
