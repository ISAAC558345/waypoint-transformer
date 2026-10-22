"""Phase-9.3 relational edge scorer. Encoder matches Phase-9.2 HGNN; scoring is relation-aware."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wt.htw_phase9.const import BUOY_DIM, EDGE_UB_DIM, EDGE_UU_DIM, GLOBAL_DIM, UAV_DIM
from wt.round30.models import masked_mean


def _mlp(inputs: int, outputs: int, width: int = 128) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))


class RelationalHGNNAssignmentHead(nn.Module):
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
        # h_u + h_b + e_ub + agg_uu + graph + raw global
        self.pair_score = _mlp(5 * width + GLOBAL_DIM, 1)
        self.idle_score = _mlp(3 * width + GLOBAL_DIM, 1)

    def _encode(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, edge: Tensor, edge_uu: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
        return h_u, h_b, e, agg_uu

    def forward(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor, edge: Tensor, edge_uu: Tensor) -> tuple[Tensor, Tensor]:
        h_u, h_b, e, agg_uu = self._encode(uav, buoy, uav_mask, buoy_mask, edge, edge_uu)
        graph = 0.5 * (masked_mean(h_u, uav_mask, 1) + masked_mean(h_b, buoy_mask, 1))
        k_count = int(buoy.shape[1])
        m_count = int(uav.shape[1])
        h_u_exp = h_u.unsqueeze(2).expand(-1, -1, k_count, -1)
        h_b_exp = h_b.unsqueeze(1).expand(-1, m_count, -1, -1)
        agg_exp = agg_uu.unsqueeze(2).expand(-1, -1, k_count, -1)
        graph_exp = graph[:, None, None, :].expand(-1, m_count, k_count, -1)
        glob_exp = global_[:, None, None, :].expand(-1, m_count, k_count, -1)
        scores = self.pair_score(torch.cat((h_u_exp, h_b_exp, e, agg_exp, graph_exp, glob_exp), dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~uav_mask.unsqueeze(-1), -1.0e9)
        scores = scores.masked_fill(~buoy_mask.unsqueeze(1), -1.0e9)
        idle = self.idle_score(torch.cat((h_u, agg_uu, graph[:, None].expand(-1, m_count, -1), global_[:, None].expand(-1, m_count, -1)), dim=-1)).squeeze(-1)
        idle = idle.masked_fill(~uav_mask, -1.0e9)
        return scores, idle


__all__ = ["RelationalHGNNAssignmentHead"]
