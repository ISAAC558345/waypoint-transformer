"""Shared residual actor / centralized critic skeletons."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from .neighbor_encoder import NeighborSetEncoder
from .observation import COMM_DIM, SELF_DIM, TASK_DIM


class ResidualActor(nn.Module):
    def __init__(self, hidden: int = 64, neighbor_out: int = 16, log_std_init: float = -1.0) -> None:
        super().__init__()
        self.neighbor = NeighborSetEncoder(out_dim=int(neighbor_out))
        in_dim = int(SELF_DIM + TASK_DIM + COMM_DIM + neighbor_out + 1)
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.full((2,), float(log_std_init)))
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def _pack(self, self_feat, task_feat, comm_feat, neighbor_edges, d_nearest):
        neigh = self.neighbor(neighbor_edges)
        if d_nearest is None:
            extra = torch.zeros(*self_feat.shape[:-1], 1, dtype=self_feat.dtype, device=self_feat.device)
        else:
            extra = d_nearest if d_nearest.shape[-1:] == (1,) else d_nearest.unsqueeze(-1)
        return torch.cat([self_feat, task_feat, comm_feat, neigh, extra], dim=-1)

    def forward(self, self_feat, task_feat, comm_feat, neighbor_edges, d_nearest=None):
        hidden = self.body(self._pack(self_feat, task_feat, comm_feat, neighbor_edges, d_nearest))
        mu = self.mu(hidden)
        return mu, self.log_std.expand_as(mu)

    def zero_residual_mean(self, self_feat, task_feat, comm_feat, neighbor_edges, d_nearest=None) -> torch.Tensor:
        mu, _ = self.forward(self_feat, task_feat, comm_feat, neighbor_edges, d_nearest)
        return mu


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(int(state_dim), hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


def gaussian_residual(mu: torch.Tensor, log_std: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
    if deterministic:
        return torch.tanh(mu)
    pre = Normal(mu, log_std.exp()).sample()
    return torch.tanh(pre)


__all__ = ["CentralizedCritic", "ResidualActor", "gaussian_residual"]
