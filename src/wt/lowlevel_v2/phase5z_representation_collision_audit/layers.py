"""Read the frozen pooling intermediates. This does not change the scorer."""
from __future__ import annotations

import torch
from torch import Tensor

from wt.lowlevel_v2.phase5t_scorer_failure_audit.represent import assignment_latents


def layer_tensors(scorer, features: dict[str, Tensor], assignments: Tensor) -> dict[str, Tensor]:
    """Same arithmetic as assignment_latents, with the pre-pool tensors kept."""
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
    post = torch.cat((pooled_selected, pooled_pairs, global_state.expand(count, -1)), dim=-1)
    reference = assignment_latents(scorer, features, assignments)
    if not torch.allclose(post, reference, atol=1.0e-5, rtol=1.0e-5):
        raise RuntimeError("layer trace drifted from the frozen assignment latent")
    return {
        "edge": chosen_e * selected_mask,
        "prepool": triples * selected_mask,
        "pairs": stacked * active,
        "postpool": post,
    }


__all__ = ["layer_tensors"]
