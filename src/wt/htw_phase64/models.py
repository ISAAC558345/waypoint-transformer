"""High-level option model with configurable buoy width. Frozen low-level is unchanged."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wt.htw_phase61.features import GLOBAL_DIM, MAX_UAVS, OPTION_DIM, UAV_DIM
from wt.round30.models import CentralDeepSetsCritic, DeepSetsPPOShape, masked_mean


def _mlp(inputs: int, outputs: int, width: int = 128) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))


class HighLevelOptionModel(nn.Module):
    def __init__(self, buoy_dim: int) -> None:
        super().__init__()
        self.shape = DeepSetsPPOShape(uav_dim=UAV_DIM, buoy_dim=int(buoy_dim), global_dim=GLOBAL_DIM, max_uavs=MAX_UAVS)
        self.uav_encoder = _mlp(self.shape.uav_dim, self.shape.width)
        self.buoy_encoder = _mlp(self.shape.buoy_dim, self.shape.width)
        self.actor_head = _mlp(3 * self.shape.width + self.shape.global_dim, OPTION_DIM, self.shape.width)
        self.critic = CentralDeepSetsCritic(self.shape)

    def forward(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor) -> tuple[Tensor, Tensor]:
        h_u, h_b = self.uav_encoder(uav), self.buoy_encoder(buoy)
        g_u, g_b = masked_mean(h_u, uav_mask, 1), masked_mean(h_b, buoy_mask, 1)
        count = uav.shape[1]
        token = torch.cat((h_u, g_u[:, None].expand(-1, count, -1), g_b[:, None].expand(-1, count, -1), global_[:, None].expand(-1, count, -1)), dim=-1)
        logits = self.actor_head(token)
        logits = logits.masked_fill(~uav_mask.unsqueeze(-1), -1.0e9)
        value = self.critic(uav, buoy, uav_mask, buoy_mask, global_)
        return logits, value


__all__ = ["HighLevelOptionModel"]
