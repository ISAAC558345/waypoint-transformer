"""Phase-9.2 assignment-head models. HGNN outputs edge scores, not independent categoricals."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wt.htw_phase9.const import BUOY_DIM, EDGE_UB_DIM, EDGE_UU_DIM, GLOBAL_DIM, MAX_BUOYS, MAX_UAVS, UAV_DIM
from wt.round30.models import masked_mean

IDLE_INDEX = MAX_BUOYS
MLP_OPTION_DIM = MAX_BUOYS + 1


def _mlp(inputs: int, outputs: int, width: int = 128) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))


class MLPOptionClassifier(nn.Module):
    """Independent buoy/idle classification. Baseline without matching."""

    def __init__(self) -> None:
        super().__init__()
        flat = MAX_UAVS * UAV_DIM + MAX_BUOYS * BUOY_DIM + GLOBAL_DIM
        self.backbone = _mlp(flat, 128)
        self.actor = nn.Linear(128, MAX_UAVS * MLP_OPTION_DIM)

    def forward(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor, edge: Tensor, edge_uu: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(torch.cat((uav.flatten(1), buoy.flatten(1), global_), dim=-1))
        logits = self.actor(hidden).view(-1, MAX_UAVS, MLP_OPTION_DIM)
        logits = logits.masked_fill(~uav_mask.unsqueeze(-1), -1.0e9)
        option_mask = torch.ones(logits.shape[0], MLP_OPTION_DIM, dtype=torch.bool, device=logits.device)
        option_mask[:, :MAX_BUOYS] = buoy_mask
        logits = logits.masked_fill(~option_mask.unsqueeze(1), -1.0e9)
        idle = logits[:, :, IDLE_INDEX]
        return logits[:, :, :MAX_BUOYS], idle


class DeepSetsAssignmentHead(nn.Module):
    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.uav_enc = _mlp(UAV_DIM, width)
        self.buoy_enc = _mlp(BUOY_DIM, width)
        self.edge_enc = _mlp(EDGE_UB_DIM, width)
        self.pair_score = _mlp(3 * width, 1)
        self.idle_score = _mlp(width, 1)

    def forward(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor, edge: Tensor, edge_uu: Tensor) -> tuple[Tensor, Tensor]:
        h_u, h_b, e = self.uav_enc(uav), self.buoy_enc(buoy), self.edge_enc(edge)
        m_count, k_count = int(uav.shape[1]), int(buoy.shape[1])
        h_u_exp = h_u.unsqueeze(2).expand(-1, -1, k_count, -1)
        h_b_exp = h_b.unsqueeze(1).expand(-1, m_count, -1, -1)
        scores = self.pair_score(torch.cat((h_u_exp, h_b_exp, e), dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~uav_mask.unsqueeze(-1), -1.0e9)
        scores = scores.masked_fill(~buoy_mask.unsqueeze(1), -1.0e9)
        idle = self.idle_score(h_u).squeeze(-1)
        idle = idle.masked_fill(~uav_mask, -1.0e9)
        return scores, idle


class HGNNAssignmentHead(nn.Module):
    """UAV-Buoy MPNN + UAV-UAV interference; edge scores go to matching, not softmax-per-UAV."""

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

    def forward(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor, edge: Tensor, edge_uu: Tensor) -> tuple[Tensor, Tensor]:
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
        scores = scores.masked_fill(~uav_mask.unsqueeze(-1), -1.0e9)
        scores = scores.masked_fill(~buoy_mask.unsqueeze(1), -1.0e9)
        idle = self.idle_score(h_u).squeeze(-1)
        idle = idle.masked_fill(~uav_mask, -1.0e9)
        _ = global_
        return scores, idle


__all__ = ["DeepSetsAssignmentHead", "HGNNAssignmentHead", "IDLE_INDEX", "MLPOptionClassifier", "MLP_OPTION_DIM"]
