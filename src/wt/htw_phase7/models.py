"""High-level option encoders: MLP, DeepSets, and bipartite HGNN. Low-level is untouched."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wt.htw_phase61.features import GLOBAL_DIM, MAX_BUOYS, MAX_UAVS, OPTION_DIM, UAV_DIM
from wt.htw_phase64.models import HighLevelOptionModel
from wt.round30.models import masked_mean

from .graph import EDGE_DIM


def _mlp(inputs: int, outputs: int, width: int = 128) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))


class MLPOptionModel(nn.Module):
    """Flat MLP high-level policy. Concatenates padded UAV/buoy sets."""

    def __init__(self, buoy_dim: int) -> None:
        super().__init__()
        self.buoy_dim = int(buoy_dim)
        flat = MAX_UAVS * UAV_DIM + MAX_BUOYS * self.buoy_dim + GLOBAL_DIM
        self.backbone = _mlp(flat, 128)
        self.actor = nn.Linear(128, MAX_UAVS * OPTION_DIM)
        self.critic = nn.Linear(128, 1)

    def forward(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor, edge: Tensor | None = None) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(torch.cat((uav.flatten(1), buoy.flatten(1), global_), dim=-1))
        logits = self.actor(hidden).view(-1, MAX_UAVS, OPTION_DIM)
        logits = logits.masked_fill(~uav_mask.unsqueeze(-1), -1.0e9)
        value = self.critic(hidden).squeeze(-1)
        return logits, value


class DeepSetsOptionModel(HighLevelOptionModel):
    def forward(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor, edge: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return super().forward(uav, buoy, uav_mask, buoy_mask, global_)


class HeteroOptionHGNN(nn.Module):
    """Two-layer UAV–Buoy message passing. HGNN is high-level only."""

    def __init__(self, buoy_dim: int, edge_dim: int = EDGE_DIM, width: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.width = width
        self.uav_enc = _mlp(UAV_DIM, width)
        self.buoy_enc = _mlp(int(buoy_dim), width)
        self.edge_enc = _mlp(int(edge_dim), width)
        self.ub_msg = nn.ModuleList([_mlp(3 * width, width) for _ in range(layers)])
        self.u_upd = nn.ModuleList([_mlp(width, width) for _ in range(layers)])
        self.b_upd = nn.ModuleList([_mlp(width, width) for _ in range(layers)])
        self.pair_score = _mlp(3 * width, 1)
        self.idle_score = _mlp(width, 1)
        self.value = _mlp(2 * width + GLOBAL_DIM, 1)

    def forward(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor, edge: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if edge is None:
            raise ValueError("HGNN high-level encoder requires UAV–Buoy edge features")
        h_u, h_b, e = self.uav_enc(uav), self.buoy_enc(buoy), self.edge_enc(edge)
        pair_mask = uav_mask[:, :, None, None] & buoy_mask[:, None, :, None]
        k_count = int(buoy.shape[1])
        m_count = int(uav.shape[1])
        for msg, u_upd, b_upd in zip(self.ub_msg, self.u_upd, self.b_upd):
            h_u_exp = h_u.unsqueeze(2).expand(-1, -1, k_count, -1)
            h_b_exp = h_b.unsqueeze(1).expand(-1, m_count, -1, -1)
            messages = msg(torch.cat((h_u_exp, h_b_exp, e), dim=-1)) * pair_mask.to(h_u.dtype)
            agg_u = messages.sum(dim=2) / pair_mask.to(h_u.dtype).sum(dim=2).clamp_min(1.0)
            agg_b = messages.sum(dim=1) / pair_mask.to(h_u.dtype).sum(dim=1).clamp_min(1.0)
            h_u = h_u + u_upd(agg_u)
            h_b = h_b + b_upd(agg_b)
        h_u_exp = h_u.unsqueeze(2).expand(-1, -1, k_count, -1)
        h_b_exp = h_b.unsqueeze(1).expand(-1, m_count, -1, -1)
        scores = self.pair_score(torch.cat((h_u_exp, h_b_exp, e), dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~buoy_mask.unsqueeze(1), -1.0e9)
        idle = self.idle_score(h_u).squeeze(-1)
        logits = torch.cat((scores, idle.unsqueeze(-1)), dim=-1)
        logits = logits.masked_fill(~uav_mask.unsqueeze(-1), -1.0e9)
        value = self.value(torch.cat((masked_mean(h_u, uav_mask, 1), masked_mean(h_b, buoy_mask, 1), global_), dim=-1)).squeeze(-1)
        return logits, value


__all__ = ["DeepSetsOptionModel", "HeteroOptionHGNN", "MLPOptionModel"]
