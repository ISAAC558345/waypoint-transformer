"""Urgency-aware residual actor. Action dimension stays 2."""
from __future__ import annotations

import torch
from torch import nn

from wt.lowlevel_v2.neighbor_encoder import NeighborSetEncoder
from wt.lowlevel_v2.observation import COMM_DIM, SELF_DIM, TASK_DIM
from wt.lowlevel_v2.residual_policy import ResidualActor

ORIGINAL_OBS_DIM = int(SELF_DIM + TASK_DIM + COMM_DIM + 16 + 1)
URGENCY_OBS_DIM = ORIGINAL_OBS_DIM + 1
URGENCY_INDEX = ORIGINAL_OBS_DIM


class UrgencyResidualActor(nn.Module):
    """Same trunk as ResidualActor, with the shared assigned-target urgency appended."""

    def __init__(self, hidden: int = 64, neighbor_out: int = 16, log_std_init: float = -1.0) -> None:
        super().__init__()
        self.neighbor = NeighborSetEncoder(out_dim=int(neighbor_out))
        self.in_dim = int(SELF_DIM + TASK_DIM + COMM_DIM + neighbor_out + 1 + 1)
        self.body = nn.Sequential(nn.Linear(self.in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.full((2,), float(log_std_init)))
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def _pack(self, self_feat, task_feat, comm_feat, neighbor_edges, d_nearest, urgency):
        neigh = self.neighbor(neighbor_edges)
        if d_nearest is None:
            extra = torch.zeros(*self_feat.shape[:-1], 1, dtype=self_feat.dtype, device=self_feat.device)
        else:
            extra = d_nearest if d_nearest.shape[-1:] == (1,) else d_nearest.unsqueeze(-1)
        if urgency.shape[-1:] != (1,):
            urgency = urgency.unsqueeze(-1)
        return torch.cat([self_feat, task_feat, comm_feat, neigh, extra, urgency], dim=-1)

    def forward(self, self_feat, task_feat, comm_feat, neighbor_edges, d_nearest=None, urgency=None):
        if urgency is None:
            urgency = torch.zeros(*self_feat.shape[:-1], 1, dtype=self_feat.dtype, device=self_feat.device)
        hidden = self.body(self._pack(self_feat, task_feat, comm_feat, neighbor_edges, d_nearest, urgency))
        mu = self.mu(hidden)
        return mu, self.log_std.expand_as(mu)


def original_actor(log_std_init: float) -> ResidualActor:
    return ResidualActor(log_std_init=float(log_std_init))


__all__ = ["ORIGINAL_OBS_DIM", "URGENCY_INDEX", "URGENCY_OBS_DIM", "UrgencyResidualActor", "original_actor"]
