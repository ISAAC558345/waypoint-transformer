"""Matched static and H=1 response scorers. The encoder is not in this module."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wt.lowlevel_v2.phase5l_joint_encoder.protocol import WIDTH
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import FINAL_REL_DIM

from .protocol import HUBER_DELTA, RESPONSE_FEATURES


class ExecutorConditionedJointScorer(nn.Module):
    """AssignmentConditionedSetScorer with a response block on each selected UAV token."""

    def __init__(self, rel_dim: int = FINAL_REL_DIM, width: int = WIDTH, response_dim: int = len(RESPONSE_FEATURES)) -> None:
        super().__init__()
        self.width = int(width)
        self.rel_dim = int(rel_dim)
        self.response_dim = int(response_dim)
        self.interact = nn.Sequential(nn.Linear(self.width * 2, self.width), nn.Tanh(), nn.Linear(self.width, self.width))
        pooled = self.width * 3 + self.response_dim + self.width * 2 + self.width
        self.mlp = nn.Sequential(nn.Linear(pooled + self.rel_dim, self.width), nn.Tanh(), nn.Linear(self.width, 1))

    def forward(self, features: dict[str, Tensor], assignments: Tensor) -> Tensor:
        if assignments.dim() != 2:
            raise ValueError("assignments must be a candidate by UAV matrix")
        h_u = features["h_u"].detach()[0]
        h_b = features["h_b"].detach()[0]
        h_edge = features["h_edge"].detach()[0]
        h_uu = features["h_uu"].detach()[0]
        global_state = features["z_global"].detach().reshape(-1)
        relational = features["z_rel"]
        response = features["response"]
        count, uavs = int(assignments.shape[0]), int(assignments.shape[1])
        if tuple(response.shape) != (count, uavs, self.response_dim):
            raise ValueError("H1 response must align with the candidate assignments")
        if tuple(relational.shape) != (count, self.rel_dim):
            raise ValueError("z_rel must align with the candidate assignments")
        buoy = assignments.clamp(min=0)
        selected = assignments >= 0
        uav_index = torch.arange(uavs, device=assignments.device)
        chosen_u = h_u.unsqueeze(0).expand(count, -1, -1)
        chosen_b = h_b[buoy]
        chosen_e = h_edge[uav_index.view(1, uavs), buoy]
        tokens = torch.cat((chosen_u, chosen_b, chosen_e, response), dim=-1)
        selected_mask = selected.unsqueeze(-1).to(tokens.dtype)
        pooled_selected = (tokens * selected_mask).sum(dim=1) / selected_mask.sum(dim=1).clamp_min(1.0)
        pair_values = []
        pair_mask = []
        for left in range(uavs):
            for right in range(left + 1, uavs):
                both = selected[:, left] & selected[:, right]
                interaction = self.interact(torch.cat((chosen_e[:, left], chosen_e[:, right]), dim=-1))
                relation = 0.5 * (h_uu[left, right] + h_uu[right, left])
                pair_values.append(torch.cat((relation.expand(count, -1), interaction), dim=-1))
                pair_mask.append(both)
        stacked = torch.stack(pair_values, dim=1)
        active = torch.stack(pair_mask, dim=1).unsqueeze(-1).to(stacked.dtype)
        pooled_pairs = (stacked * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
        latent = torch.cat((pooled_selected, pooled_pairs, global_state.expand(count, -1), relational), dim=-1)
        return self.mlp(latent).squeeze(-1)


def state_huber(prediction: Tensor, target_blocks: Tensor) -> Tensor:
    if prediction.shape != target_blocks.shape:
        raise ValueError("prediction and J15 blocks must describe the same assignments")
    return F.huber_loss(prediction, target_blocks, reduction="mean", delta=float(HUBER_DELTA))


__all__ = ["ExecutorConditionedJointScorer", "state_huber"]
