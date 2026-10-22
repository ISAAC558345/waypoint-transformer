"""HGNN encoder for the unchanged order-free decoder. UAV-UAV and UAV-Buoy only."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wt.htw_phase9.const import EDGE_UB_DIM, EDGE_UU_DIM, UAV_DIM
from wt.htw_phase943.order_free_decoder import OrderFreeSetDecoder
from wt.round30.models import masked_mean


def _mlp(inputs: int, outputs: int, width: int = 128) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))


class Phase5HGNNEncoder(nn.Module):
    """Heterogeneous message passing. No buoy-buoy edges."""

    def __init__(self, buoy_dim: int, width: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.buoy_dim = int(buoy_dim)
        self.uav_enc = _mlp(UAV_DIM, width)
        self.buoy_enc = _mlp(self.buoy_dim, width)
        self.edge_enc = _mlp(EDGE_UB_DIM, width)
        self.uu_enc = _mlp(EDGE_UU_DIM, width)
        self.ub_msg = nn.ModuleList([_mlp(3 * width, width) for _ in range(layers)])
        self.u_upd = nn.ModuleList([_mlp(width, width) for _ in range(layers)])
        self.b_upd = nn.ModuleList([_mlp(width, width) for _ in range(layers)])
        self.uu_msg = _mlp(3 * width, width)
        self.uu_upd = _mlp(width, width)

    def encode_nodes(
        self,
        uav: Tensor,
        buoy: Tensor,
        uav_mask: Tensor,
        buoy_mask: Tensor,
        edge: Tensor,
        edge_uu: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if int(buoy.shape[-1]) != self.buoy_dim:
            raise ValueError(f"buoy dim {int(buoy.shape[-1])} != encoder {self.buoy_dim}")
        h_u, h_b, edge_h = self.uav_enc(uav), self.buoy_enc(buoy), self.edge_enc(edge)
        pair_mask = uav_mask[:, :, None, None] & buoy_mask[:, None, :, None]
        k_count = int(buoy.shape[1])
        m_count = int(uav.shape[1])
        for msg, u_upd, b_upd in zip(self.ub_msg, self.u_upd, self.b_upd):
            h_u_exp = h_u.unsqueeze(2).expand(-1, -1, k_count, -1)
            h_b_exp = h_b.unsqueeze(1).expand(-1, m_count, -1, -1)
            messages = msg(torch.cat((h_u_exp, h_b_exp, edge_h), dim=-1)) * pair_mask.to(h_u.dtype)
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
        return h_u, h_b, edge_h


class Phase5OrderFreeModel(nn.Module):
    """New HGNN encoder plus a fresh OrderFreeSetDecoder of the existing class."""

    def __init__(self, buoy_dim: int) -> None:
        super().__init__()
        self.encoder = Phase5HGNNEncoder(buoy_dim=int(buoy_dim))
        self.decoder = OrderFreeSetDecoder()

    def encode(
        self,
        uav: Tensor,
        buoy: Tensor,
        uav_mask: Tensor,
        buoy_mask: Tensor,
        global_: Tensor,
        edge: Tensor,
        edge_uu: Tensor,
    ) -> dict[str, Tensor]:
        h_u, h_b, e_ub = self.encoder.encode_nodes(uav, buoy, uav_mask, buoy_mask, edge, edge_uu)
        graph = 0.5 * (masked_mean(h_u, uav_mask, 1) + masked_mean(h_b, buoy_mask, 1))
        return {
            "h_u": h_u,
            "h_b": h_b,
            "e_ub": e_ub,
            "graph": graph,
            "global": global_,
            "uav_xy": uav[..., :2],
            "buoy_xy": buoy[..., :2],
        }


__all__ = ["Phase5HGNNEncoder", "Phase5OrderFreeModel"]
