"""Pair each UAV with its target, then average. UAV and buoy are not pooled apart."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wt.lowlevel_v2.phase5za_correspondence_pooling.protocol import (
    ASSIGNMENT_DIM,
    FINAL_REL_DIM,
    GLOBAL_DIM,
    HEAD_INPUT_DIM,
    REL_DIM,
    TEMP_INPUT_DIM,
    WIDTH,
)


class CorrespondencePreservingPairPool(nn.Module):
    """Shared phi on (UAV, target) tokens. The mean runs after the binding."""

    def __init__(self) -> None:
        super().__init__()
        self.width = int(WIDTH)
        self.rel_dim = int(REL_DIM)
        self.temp_dim = int(TEMP_INPUT_DIM)
        self.input_dim = int(self.width * 3 + self.rel_dim + self.temp_dim + 1)
        self.hidden_dim = int(WIDTH)
        self.output_dim = int(ASSIGNMENT_DIM)
        self.phi = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def pair_tokens(self, features: dict[str, Tensor], assignments: Tensor, relational: Tensor) -> Tensor:
        if assignments.dim() != 2:
            raise ValueError("assignments must be a candidate by UAV matrix")
        if tuple(relational.shape) != (int(assignments.shape[0]), int(FINAL_REL_DIM)):
            raise ValueError("REP and TEMP must align with the candidate assignments")
        h_u = features["h_u"].detach()[0]
        h_b = features["h_b"].detach()[0]
        h_edge = features["h_edge"].detach()[0]
        count, uavs = int(assignments.shape[0]), int(assignments.shape[1])
        buoy = assignments.clamp(min=0)
        selected = assignments >= 0
        uav_index = torch.arange(uavs, device=assignments.device)
        chosen_u = h_u.unsqueeze(0).expand(count, -1, -1)
        chosen_b = h_b[buoy] * selected.unsqueeze(-1).to(h_b.dtype)
        chosen_e = h_edge[uav_index.view(1, uavs), buoy] * selected.unsqueeze(-1).to(h_edge.dtype)
        rep = relational[:, : self.rel_dim].unsqueeze(1).expand(-1, uavs, -1)
        temp = relational[:, self.rel_dim :].unsqueeze(1).expand(-1, uavs, -1)
        idle = (~selected).unsqueeze(-1).to(chosen_u.dtype)
        return torch.cat((chosen_u, chosen_b, chosen_e, rep, temp, idle), dim=-1)

    def forward(self, features: dict[str, Tensor], assignments: Tensor, relational: Tensor) -> Tensor:
        tokens = self.pair_tokens(features, assignments, relational)
        return self.phi(tokens).mean(dim=1)


class CorrespondencePreservingValue(nn.Module):
    """Phase-5Y head dimensions. Only phi and this head are trainable."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = CorrespondencePreservingPairPool()
        self.width = int(WIDTH)
        self.rel_dim = int(FINAL_REL_DIM)
        if int(ASSIGNMENT_DIM + GLOBAL_DIM + FINAL_REL_DIM) != int(HEAD_INPUT_DIM):
            raise RuntimeError("value-head input width drifted from Phase-5Y")
        self.mlp = nn.Sequential(nn.Linear(int(HEAD_INPUT_DIM), self.width), nn.Tanh(), nn.Linear(self.width, 1))

    def assignment_embedding(self, features: dict[str, Tensor], assignments: Tensor) -> Tensor:
        relational = features["z_rel"]
        return self.pool(features, assignments, relational)

    def latent(self, features: dict[str, Tensor], assignments: Tensor) -> Tensor:
        pooled = self.assignment_embedding(features, assignments)
        global_state = features["z_global"].detach().reshape(-1)
        relational = features["z_rel"]
        return torch.cat((pooled, global_state.expand(pooled.shape[0], -1), relational), dim=-1)

    def forward(self, features: dict[str, Tensor], assignments: Tensor) -> Tensor:
        return self.mlp(self.latent(features, assignments)).squeeze(-1)


def permute_uavs(features: dict[str, Tensor], assignments: Tensor, perm: Tensor) -> tuple[dict[str, Tensor], Tensor]:
    order = perm.to(dtype=torch.long)
    updated = {key: value.detach().clone() for key, value in features.items()}
    updated["h_u"] = features["h_u"].detach()[:, order].clone()
    updated["h_edge"] = features["h_edge"].detach()[:, order].clone()
    if "h_uu" in features:
        updated["h_uu"] = features["h_uu"].detach()[:, order][:, :, order].clone()
    updated["z_rel"] = features["z_rel"].detach().clone()
    return updated, assignments[:, order].detach().clone()


def permute_buoys(features: dict[str, Tensor], assignments: Tensor, perm: Tensor) -> tuple[dict[str, Tensor], Tensor]:
    order = perm.to(dtype=torch.long)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    updated = {key: value.detach().clone() for key, value in features.items()}
    updated["h_b"] = features["h_b"].detach()[:, order].clone()
    updated["h_edge"] = features["h_edge"].detach()[:, :, order].clone()
    updated["z_rel"] = features["z_rel"].detach().clone()
    relabeled = assignments.detach().clone()
    active = relabeled >= 0
    relabeled[active] = inverse[relabeled[active]]
    return updated, relabeled


__all__ = [
    "CorrespondencePreservingPairPool",
    "CorrespondencePreservingValue",
    "permute_buoys",
    "permute_uavs",
]
