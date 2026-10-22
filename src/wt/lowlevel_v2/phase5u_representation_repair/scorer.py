"""Phase-5S scorer with a fixed relational block concatenated to z_A."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wt.lowlevel_v2.phase5l_joint_encoder.protocol import WIDTH
from wt.lowlevel_v2.phase5t_scorer_failure_audit.represent import assignment_latents

from .protocol import REL_DIM


class RelationalSetScorer(nn.Module):
    def __init__(self, rel_dim: int = REL_DIM, width: int = WIDTH) -> None:
        super().__init__()
        self.width = int(width)
        self.rel_dim = int(rel_dim)
        self.interact = nn.Sequential(nn.Linear(self.width * 2, self.width), nn.Tanh(), nn.Linear(self.width, self.width))
        pooled = self.width * 6
        self.mlp = nn.Sequential(nn.Linear(pooled + self.rel_dim, self.width), nn.Tanh(), nn.Linear(self.width, 1))

    def latent(self, features: dict[str, Tensor], assignments: Tensor) -> Tensor:
        base = assignment_latents(self, features, assignments)
        relational = features["z_rel"]
        if tuple(relational.shape) != (int(assignments.shape[0]), self.rel_dim):
            raise ValueError("z_rel must align with the candidate assignments")
        return torch.cat((base, relational), dim=-1)

    def forward(self, features: dict[str, Tensor], assignments: Tensor) -> Tensor:
        return self.mlp(self.latent(features, assignments)).squeeze(-1)


__all__ = ["RelationalSetScorer"]
