"""Phase-9 high-level encoders. HGNN lives only here; Phase-7 files are not edited."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wt.round30.models import CentralDeepSetsCritic, DeepSetsPPOShape, masked_mean

from .const import BUOY_DIM, EDGE_UB_DIM, EDGE_UU_DIM, GLOBAL_DIM, MAX_BUOYS, MAX_UAVS, OPTION_DIM, UAV_DIM


def _mlp(inputs: int, outputs: int, width: int = 128) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))


def _mask_options(logits: Tensor, uav_mask: Tensor, buoy_mask: Tensor) -> Tensor:
    logits = logits.masked_fill(~uav_mask.unsqueeze(-1), -1.0e9)
    option_mask = torch.ones(logits.shape[0], logits.shape[-1], dtype=torch.bool, device=logits.device)
    option_mask[:, :MAX_BUOYS] = buoy_mask
    return logits.masked_fill(~option_mask.unsqueeze(1), -1.0e9)


class MLPOptionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        flat = MAX_UAVS * UAV_DIM + MAX_BUOYS * BUOY_DIM + GLOBAL_DIM
        self.backbone = _mlp(flat, 128)
        self.actor = nn.Linear(128, MAX_UAVS * OPTION_DIM)
        self.critic = nn.Linear(128, 1)

    def forward(
        self,
        uav: Tensor,
        buoy: Tensor,
        uav_mask: Tensor,
        buoy_mask: Tensor,
        global_: Tensor,
        edge: Tensor | None = None,
        edge_uu: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(torch.cat((uav.flatten(1), buoy.flatten(1), global_), dim=-1))
        logits = self.actor(hidden).view(-1, MAX_UAVS, OPTION_DIM)
        value = self.critic(hidden).squeeze(-1)
        return _mask_options(logits, uav_mask, buoy_mask), value


class DeepSetsOptionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shape = DeepSetsPPOShape(uav_dim=UAV_DIM, buoy_dim=BUOY_DIM, global_dim=GLOBAL_DIM, max_uavs=MAX_UAVS)
        self.uav_encoder = _mlp(self.shape.uav_dim, self.shape.width)
        self.buoy_encoder = _mlp(self.shape.buoy_dim, self.shape.width)
        self.actor_head = _mlp(3 * self.shape.width + self.shape.global_dim, OPTION_DIM, self.shape.width)
        self.critic = CentralDeepSetsCritic(self.shape)

    def forward(
        self,
        uav: Tensor,
        buoy: Tensor,
        uav_mask: Tensor,
        buoy_mask: Tensor,
        global_: Tensor,
        edge: Tensor | None = None,
        edge_uu: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        h_u, h_b = self.uav_encoder(uav), self.buoy_encoder(buoy)
        g_u, g_b = masked_mean(h_u, uav_mask, 1), masked_mean(h_b, buoy_mask, 1)
        count = uav.shape[1]
        token = torch.cat((h_u, g_u[:, None].expand(-1, count, -1), g_b[:, None].expand(-1, count, -1), global_[:, None].expand(-1, count, -1)), dim=-1)
        logits = self.actor_head(token)
        value = self.critic(uav, buoy, uav_mask, buoy_mask, global_)
        return _mask_options(logits, uav_mask, buoy_mask), value


class HeteroOptionHGNN(nn.Module):
    """Bipartite UAV–Buoy MPNN plus UAV–UAV interference messages. High-level only."""

    def __init__(self, width: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.uav_enc = _mlp(UAV_DIM, width)
        self.buoy_enc = _mlp(BUOY_DIM, width)
        self.edge_enc = _mlp(EDGE_UB_DIM, width)
        self.uu_enc = _mlp(EDGE_UU_DIM, width)
        self.ub_msg = nn.ModuleList([_mlp(3 * width, width) for _ in range(layers)])
        self.u_upd = nn.ModuleList([_mlp(width, width) for _ in range(layers)])
        self.b_upd = nn.ModuleList([_mlp(width, width) for _ in range(layers)])
        self.uu_msg = _mlp(3 * width, width)
        self.uu_upd = _mlp(width, width)
        self.pair_score = _mlp(3 * width, 1)
        self.idle_score = _mlp(width, 1)
        self.value = _mlp(2 * width + GLOBAL_DIM, 1)

    def forward(
        self,
        uav: Tensor,
        buoy: Tensor,
        uav_mask: Tensor,
        buoy_mask: Tensor,
        global_: Tensor,
        edge: Tensor | None = None,
        edge_uu: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if edge is None or edge_uu is None:
            raise ValueError("HGNN requires UAV-Buoy and UAV-UAV edges")
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
        e_uu = self.uu_enc(edge_uu)
        uu_mask = uav_mask[:, :, None, None] & uav_mask[:, None, :, None]
        eye = torch.eye(m_count, device=uav.device, dtype=torch.bool)[None, :, :, None]
        uu_mask = uu_mask & (~eye)
        h_src = h_u.unsqueeze(1).expand(-1, m_count, -1, -1)
        h_dst = h_u.unsqueeze(2).expand(-1, -1, m_count, -1)
        uu_messages = self.uu_msg(torch.cat((h_dst, h_src, e_uu), dim=-1)) * uu_mask.to(h_u.dtype)
        agg_uu = uu_messages.sum(dim=2) / uu_mask.to(h_u.dtype).sum(dim=2).clamp_min(1.0)
        h_u = h_u + self.uu_upd(agg_uu)
        h_u_exp = h_u.unsqueeze(2).expand(-1, -1, k_count, -1)
        h_b_exp = h_b.unsqueeze(1).expand(-1, m_count, -1, -1)
        scores = self.pair_score(torch.cat((h_u_exp, h_b_exp, e), dim=-1)).squeeze(-1)
        idle = self.idle_score(h_u).squeeze(-1)
        logits = torch.cat((scores, idle.unsqueeze(-1)), dim=-1)
        value = self.value(torch.cat((masked_mean(h_u, uav_mask, 1), masked_mean(h_b, buoy_mask, 1), global_), dim=-1)).squeeze(-1)
        return _mask_options(logits, uav_mask, buoy_mask), value


__all__ = ["DeepSetsOptionModel", "HeteroOptionHGNN", "MLPOptionModel"]
