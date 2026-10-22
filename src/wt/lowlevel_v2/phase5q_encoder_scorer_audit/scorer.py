"""Non-additive score of one complete candidate assignment.

The score is an MLP over the selected UAV-buoy set and the selected UAV-UAV
interactions. It is not a sum of per-edge scores, and it does not read J_MAPPO.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wt.lowlevel_v2.phase5l_joint_encoder.protocol import WIDTH


class AssignmentConditionedSetScorer(nn.Module):
    def __init__(self, width: int = WIDTH) -> None:
        super().__init__()
        self.width = int(width)
        self.interact = nn.Sequential(nn.Linear(self.width * 2, self.width), nn.Tanh(), nn.Linear(self.width, self.width))
        pooled = self.width * 3 + self.width * 2 + self.width
        self.mlp = nn.Sequential(nn.Linear(pooled, self.width), nn.Tanh(), nn.Linear(self.width, 1))

    def forward(self, features: dict[str, Tensor], assignments: Tensor) -> Tensor:
        if assignments.dim() != 2:
            raise ValueError("assignments must be a candidate by UAV matrix")
        h_u = features["h_u"][0]
        h_b = features["h_b"][0]
        h_edge = features["h_edge"][0]
        h_uu = features["h_uu"][0]
        global_state = features["z_global"].reshape(-1)
        if int(h_u.shape[-1]) != self.width or int(global_state.shape[0]) != self.width:
            raise ValueError("scorer width does not match the frozen embeddings")
        count, uavs = int(assignments.shape[0]), int(assignments.shape[1])
        buoy = assignments.clamp(min=0)
        selected = assignments >= 0
        uav_index = torch.arange(uavs)
        chosen_u = h_u.unsqueeze(0).expand(count, -1, -1)
        chosen_b = h_b[buoy]
        chosen_e = h_edge[uav_index.view(1, uavs), buoy]
        selected_mask = selected.unsqueeze(-1).to(chosen_u.dtype)
        triples = torch.cat((chosen_u, chosen_b, chosen_e), dim=-1)
        pooled_selected = (triples * selected_mask).sum(dim=1) / selected_mask.sum(dim=1).clamp_min(1.0)
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
        latent = torch.cat((pooled_selected, pooled_pairs, global_state.expand(count, -1)), dim=-1)
        return self.mlp(latent).squeeze(-1)


__all__ = ["AssignmentConditionedSetScorer"]
