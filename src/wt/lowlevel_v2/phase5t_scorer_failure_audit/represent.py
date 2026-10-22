"""Latent z_A entering the frozen scorer MLP. This does not change the module."""
from __future__ import annotations

import torch
from torch import Tensor

from wt.lowlevel_v2.phase5q_encoder_scorer_audit.scorer import AssignmentConditionedSetScorer


def assignment_latents(scorer: AssignmentConditionedSetScorer, features: dict[str, Tensor], assignments: Tensor) -> Tensor:
    """Same pooling as AssignmentConditionedSetScorer.forward, stopped before mlp."""
    if assignments.dim() != 2:
        raise ValueError("assignments must be a candidate by UAV matrix")
    h_u = features["h_u"][0]
    h_b = features["h_b"][0]
    h_edge = features["h_edge"][0]
    h_uu = features["h_uu"][0]
    global_state = features["z_global"].reshape(-1)
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
            interaction = scorer.interact(torch.cat((chosen_e[:, left], chosen_e[:, right]), dim=-1))
            relation = 0.5 * (h_uu[left, right] + h_uu[right, left])
            pair_values.append(torch.cat((relation.expand(count, -1), interaction), dim=-1))
            pair_mask.append(both)
    stacked = torch.stack(pair_values, dim=1)
    active = torch.stack(pair_mask, dim=1).unsqueeze(-1).to(stacked.dtype)
    pooled_pairs = (stacked * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
    return torch.cat((pooled_selected, pooled_pairs, global_state.expand(count, -1)), dim=-1)


__all__ = ["assignment_latents"]
