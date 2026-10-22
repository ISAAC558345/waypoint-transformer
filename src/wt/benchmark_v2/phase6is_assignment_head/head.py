"""Small assignment-conditioned scorer on frozen encoder states.

The score is a permutation-invariant function of the six selected UAV tokens.
Each token contains the encoder state of that UAV, its assigned buoy, the
selected edge, and decision-time interference from the other assigned UAVs.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wt.benchmark_v2.phase6is_assignment_head.protocol import HIDDEN, RAW_NAMES
from wt.lowlevel_v2.phase5l_joint_encoder.protocol import WIDTH


class AssignmentConditionedHead(nn.Module):
    def __init__(self, width: int = WIDTH, raw_dim: int = len(RAW_NAMES), hidden: int = HIDDEN) -> None:
        super().__init__()
        token_dim = int(width) * 3 + int(raw_dim)
        self.width = int(width)
        self.raw_dim = int(raw_dim)
        self.phi = nn.Sequential(
            nn.Linear(token_dim, int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), int(hidden)),
            nn.Tanh(),
        )
        self.rho = nn.Sequential(
            nn.Linear(int(hidden) * 2, int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), 1),
        )

    def tokens(self, h_u: Tensor, h_b: Tensor, edge_h: Tensor, raw: Tensor, assignments: Tensor) -> Tensor:
        if h_u.dim() != 2 or h_b.dim() != 2 or edge_h.dim() != 3:
            raise ValueError("encoder states must be one decision, without a batch axis")
        if assignments.dim() != 2 or int(assignments.shape[1]) != int(h_u.shape[0]):
            raise ValueError("assignments must be [count, num_uavs]")
        if int(raw.shape[-1]) != self.raw_dim:
            raise ValueError("raw feature width does not match the head")
        valid = assignments >= 0
        safe = assignments.clamp(min=0)
        count, uav_count = int(assignments.shape[0]), int(assignments.shape[1])
        uav_index = torch.arange(uav_count, device=assignments.device).unsqueeze(0).expand(count, uav_count)
        selected_buoy = h_b[safe] * valid.unsqueeze(-1).to(h_b.dtype)
        selected_edge = edge_h[uav_index, safe] * valid.unsqueeze(-1).to(edge_h.dtype)
        selected_uav = h_u.unsqueeze(0).expand(count, uav_count, -1)
        selected_raw = raw.to(dtype=h_u.dtype)
        idle = (~valid).unsqueeze(-1).to(h_u.dtype)
        selected_raw = torch.cat((selected_raw[..., :-1] * valid.unsqueeze(-1).to(h_u.dtype), idle), dim=-1)
        return torch.cat((selected_uav, selected_buoy, selected_edge, selected_raw), dim=-1)

    def forward(self, h_u: Tensor, h_b: Tensor, edge_h: Tensor, raw: Tensor, assignments: Tensor) -> Tensor:
        hidden = self.phi(self.tokens(h_u, h_b, edge_h, raw, assignments))
        pooled = torch.cat((hidden.sum(dim=1), hidden.amax(dim=1)), dim=-1)
        return self.rho(pooled).squeeze(-1)


__all__ = ["AssignmentConditionedHead"]
