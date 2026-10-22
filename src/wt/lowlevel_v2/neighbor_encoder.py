"""Permutation-invariant neighbor set encoder. Schema/forward only; not trained."""
from __future__ import annotations

import torch
from torch import nn

from .observation import NEIGHBOR_EDGE_DIM


class NeighborSetEncoder(nn.Module):
    def __init__(self, edge_dim: int = NEIGHBOR_EDGE_DIM, hidden: int = 32, out_dim: int = 16) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(int(edge_dim), int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), int(out_dim)),
        )
        self.out_dim = int(out_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        """edges: (..., N_neighbors, edge_dim) -> (..., out_dim) mean pool."""
        encoded = self.phi(edges)
        return encoded.mean(dim=-2)


def permutation_invariance_error(encoder: NeighborSetEncoder, edges: torch.Tensor, perm: torch.Tensor) -> float:
    with torch.no_grad():
        left = encoder(edges)
        shuffled = edges.index_select(-2, perm)
        right = encoder(shuffled)
        return float(torch.max(torch.abs(left - right)))


__all__ = ["NeighborSetEncoder", "permutation_invariance_error"]
