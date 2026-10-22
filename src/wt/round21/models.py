"""Permutation-equivariant single-step Round-21 spatial graph models."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def mlp(input_dim: int, output_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),
    )


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    return (values * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


class GraphLayer(nn.Module):
    """One relation-specific B→U, U→B, U→U message-passing layer."""

    def __init__(self, hidden_dim: int, ub_dim: int, uu_dim: int, dropout: float, use_uu: bool = True) -> None:
        super().__init__()
        self.use_uu = use_uu
        self.msg_bu = mlp(2 * hidden_dim + ub_dim, hidden_dim, hidden_dim, dropout)
        self.msg_ub = mlp(2 * hidden_dim + ub_dim, hidden_dim, hidden_dim, dropout)
        self.msg_uu = mlp(2 * hidden_dim + uu_dim, hidden_dim, hidden_dim, dropout) if use_uu else None
        self.update_u = mlp(3 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.update_b = mlp(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.norm_u = nn.LayerNorm(hidden_dim)
        self.norm_b = nn.LayerNorm(hidden_dim)

    def forward(self, h_u: Tensor, h_b: Tensor, ub: Tensor, uu: Tensor, u_mask: Tensor, b_mask: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        batch, m_count, hidden = h_u.shape
        k_count = h_b.shape[1]
        u_here = h_u[:, :, None, :].expand(-1, -1, k_count, -1)
        b_here = h_b[:, None, :, :].expand(-1, m_count, -1, -1)
        pair = torch.cat((b_here, u_here, ub), dim=-1)
        messages_bu = self.msg_bu(pair)
        messages_ub = self.msg_ub(torch.cat((u_here, b_here, ub), dim=-1))
        ub_mask = u_mask[:, :, None] & b_mask[:, None, :]
        agg_bu = masked_mean(messages_bu, ub_mask, dim=2)
        agg_ub = masked_mean(messages_ub, ub_mask, dim=1)
        if self.use_uu and self.msg_uu is not None:
            # Axis order is [target UAV m, source UAV j]; message passing is
            # explicitly directed j→m and never assumes a particular UAV order.
            target = h_u[:, :, None, :].expand(-1, -1, m_count, -1)
            source = h_u[:, None, :, :].expand(-1, m_count, -1, -1)
            messages_uu = self.msg_uu(torch.cat((source, target, uu), dim=-1))
            uu_mask = u_mask[:, :, None] & u_mask[:, None, :]
            eye = torch.eye(m_count, dtype=torch.bool, device=h_u.device)[None, :, :]
            uu_mask = uu_mask & ~eye
            agg_uu = masked_mean(messages_uu, uu_mask, dim=2)
        else:
            messages_uu = h_u.new_zeros((batch, m_count, m_count, hidden))
            agg_uu = h_u.new_zeros((batch, m_count, hidden))
        next_u = self.norm_u(h_u + self.update_u(torch.cat((h_u, agg_bu, agg_uu), dim=-1)))
        next_b = self.norm_b(h_b + self.update_b(torch.cat((h_b, agg_ub), dim=-1)))
        next_u = next_u * u_mask.unsqueeze(-1).to(next_u.dtype)
        next_b = next_b * b_mask.unsqueeze(-1).to(next_b.dtype)
        return next_u, next_b, {"bu": messages_bu, "ub": messages_ub, "uu": messages_uu}


class _TaskHeads(nn.Module):
    def __init__(self, hidden_dim: int, global_dim: int, ub_dim: int, dropout: float) -> None:
        super().__init__()
        pair_dim = 3 * hidden_dim + ub_dim
        self.global_mlp = mlp(2 * hidden_dim + global_dim, hidden_dim, hidden_dim, dropout)
        self.service = mlp(pair_dim, 1, hidden_dim, dropout)
        self.progress = mlp(pair_dim, 1, hidden_dim, dropout)
        self.team_return = mlp(hidden_dim, 1, hidden_dim, dropout)

    def forward(self, h_u: Tensor, h_b: Tensor, global_features: Tensor, u_mask: Tensor, b_mask: Tensor, ub_edges: Tensor | None = None) -> dict[str, Tensor]:
        m_count, k_count = h_u.shape[1], h_b.shape[1]
        pooled_u = masked_mean(h_u, u_mask, dim=1)
        pooled_b = masked_mean(h_b, b_mask, dim=1)
        h_g = self.global_mlp(torch.cat((pooled_u, pooled_b, global_features), dim=-1))
        u = h_u[:, :, None, :].expand(-1, -1, k_count, -1)
        b = h_b[:, None, :, :].expand(-1, m_count, -1, -1)
        global_pair = h_g[:, None, None, :].expand(-1, m_count, k_count, -1)
        edge = ub_edges if ub_edges is not None else h_u.new_zeros((*u.shape[:-1], 0))
        pair = torch.cat((u, b, edge, global_pair), dim=-1)
        valid = u_mask[:, :, None] & b_mask[:, None, :]
        scores = self.service(pair).squeeze(-1).masked_fill(~valid, -1.0e9)
        service = torch.softmax(scores, dim=-1) * valid.to(scores.dtype)
        progress = F.softplus(self.progress(pair).squeeze(-1)) * valid.to(scores.dtype)
        return {"service": service, "progress": progress, "team_return": self.team_return(h_g).squeeze(-1), "h_u": h_u, "h_b": h_b, "h_g": h_g}


class HeterogeneousTaskWaypointPredictor(nn.Module):
    """Two-layer HGNN with shared node encoders and relation-specific messages."""

    def __init__(self, uav_dim: int, buoy_dim: int, ub_dim: int, uu_dim: int, global_dim: int, hidden_dim: int = 128, graph_layers: int = 2, dropout: float = 0.1, use_uu: bool = True) -> None:
        super().__init__()
        if graph_layers != 2:
            raise ValueError("Round-21 graph_layers is frozen at two")
        self.family = "full_hgnn" if use_uu else "hgnn_no_uu"
        self.hidden_dim = hidden_dim
        self.use_uu = use_uu
        self.uav_encoder = mlp(uav_dim, hidden_dim, hidden_dim, dropout)
        self.buoy_encoder = mlp(buoy_dim, hidden_dim, hidden_dim, dropout)
        self.layers = nn.ModuleList([GraphLayer(hidden_dim, ub_dim, uu_dim, dropout, use_uu) for _ in range(graph_layers)])
        self.heads = _TaskHeads(hidden_dim, global_dim, ub_dim, dropout)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        h_u = self.uav_encoder(batch["uav_nodes"]) * batch["uav_mask"].unsqueeze(-1).to(batch["uav_nodes"].dtype)
        h_b = self.buoy_encoder(batch["buoy_nodes"]) * batch["buoy_mask"].unsqueeze(-1).to(batch["buoy_nodes"].dtype)
        relation_messages: list[dict[str, Tensor]] = []
        for layer in self.layers:
            h_u, h_b, messages = layer(h_u, h_b, batch["ub_edges"], batch["uu_edges"], batch["uav_mask"], batch["buoy_mask"])
            relation_messages.append(messages)
        output = self.heads(h_u, h_b, batch["global_features"], batch["uav_mask"], batch["buoy_mask"], batch["ub_edges"])
        output["relation_messages"] = relation_messages
        return output


class DeepSetsTaskWaypointPredictor(nn.Module):
    """Parameter-comparable set baseline with no relation messages or edges."""

    def __init__(self, uav_dim: int, buoy_dim: int, global_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.family = "deepsets"
        self.hidden_dim = hidden_dim
        self.uav_encoder = mlp(uav_dim, hidden_dim, hidden_dim, dropout)
        self.buoy_encoder = mlp(buoy_dim, hidden_dim, hidden_dim, dropout)
        # Extra shared transforms retain broadly comparable depth/capacity but
        # do not receive UB/UU edge features or messages.
        self.uav_set_transform = mlp(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.buoy_set_transform = mlp(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.heads = _TaskHeads(hidden_dim, global_dim, 0, dropout)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        h_u = self.uav_set_transform(self.uav_encoder(batch["uav_nodes"])) * batch["uav_mask"].unsqueeze(-1).to(batch["uav_nodes"].dtype)
        h_b = self.buoy_set_transform(self.buoy_encoder(batch["buoy_nodes"])) * batch["buoy_mask"].unsqueeze(-1).to(batch["buoy_nodes"].dtype)
        output = self.heads(h_u, h_b, batch["global_features"], batch["uav_mask"], batch["buoy_mask"], None)
        output["relation_messages"] = []
        return output


def build_model(family: str, dimensions: dict[str, int], hidden_dim: int = 128, graph_layers: int = 2, dropout: float = 0.1) -> nn.Module:
    if family == "deepsets":
        return DeepSetsTaskWaypointPredictor(dimensions["uav"], dimensions["buoy"], dimensions["global"], hidden_dim, dropout)
    if family in {"full_hgnn", "hgnn_no_interference"}:
        return HeterogeneousTaskWaypointPredictor(dimensions["uav"], dimensions["buoy"], dimensions["ub"], dimensions["uu"], dimensions["global"], hidden_dim, graph_layers, dropout, True)
    if family == "hgnn_no_uu":
        return HeterogeneousTaskWaypointPredictor(dimensions["uav"], dimensions["buoy"], dimensions["ub"], dimensions["uu"], dimensions["global"], hidden_dim, graph_layers, dropout, False)
    raise ValueError(f"unknown Round-21 family: {family}")


__all__ = ["DeepSetsTaskWaypointPredictor", "HeterogeneousTaskWaypointPredictor", "build_model", "masked_mean"]
