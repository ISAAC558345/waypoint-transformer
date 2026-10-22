"""Assignment-level joint head. Pair scores are coupled by Sinkhorn before decoding."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from wt.lowlevel_v2.phase5k_joint_decoder.decoders import capacity_plan, gather_assignment_score, hungarian_assign
from wt.round30.models import masked_mean


class JointAssignmentHead(nn.Module):
    """Score UAV-buoy pairs from individual embeddings and one global embedding."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width * 3, width), nn.Tanh(), nn.Linear(width, 1))

    def pair_scores(self, h_u: Tensor, h_b: Tensor, z_global: Tensor) -> Tensor:
        if h_u.dim() != 3 or h_b.dim() != 3:
            raise ValueError("embeddings must include a batch axis")
        if int(h_u.shape[0]) != 1:
            raise ValueError("joint head scores one decision at a time")
        uav_count = int(h_u.shape[1])
        buoy_count = int(h_b.shape[1])
        left = h_u[0].unsqueeze(1).expand(uav_count, buoy_count, -1)
        right = h_b[0].unsqueeze(0).expand(uav_count, buoy_count, -1)
        context = z_global.reshape(1, 1, -1).expand(uav_count, buoy_count, -1)
        return self.net(torch.cat((left, right, context), dim=-1)).squeeze(-1)


def global_embedding(h_u: Tensor, h_b: Tensor, uav_mask: Tensor, buoy_mask: Tensor) -> Tensor:
    return 0.5 * (masked_mean(h_u, uav_mask, 1) + masked_mean(h_b, buoy_mask, 1))


def masked_plan(logits: Tensor, candidate_mask: Tensor) -> Tensor:
    """Sinkhorn plan. Non-candidates receive no mass."""
    masked = logits.masked_fill(~candidate_mask, -1.0e9)
    return capacity_plan(masked)


def assignment_scores(plan: Tensor, assignments: Tensor) -> Tensor:
    """Sum of coupled soft-assignment mass. Hungarian maximizes this score."""
    return gather_assignment_score(plan, assignments, logarithmic=False)


def hungarian_from_plan(plan: Tensor, candidate_mask: Tensor) -> np.ndarray:
    """Decode the soft plan. A UAV with no free candidate stays idle.

    linear_sum_assignment still returns a column for every row. Columns outside
    the candidate mask are not assignments; those UAVs are set back to idle so
    the decoded tuple stays inside the supervised matching set.
    """
    mask = candidate_mask.detach().cpu().numpy().astype(bool)
    matrix = plan.detach().cpu().numpy().copy()
    matrix[~mask] = -1.0
    assignment = hungarian_assign(matrix)
    for uav, buoy in enumerate(assignment):
        if int(buoy) < 0 or not bool(mask[int(uav), int(buoy)]):
            assignment[int(uav)] = -1
    return assignment


__all__ = [
    "JointAssignmentHead",
    "assignment_scores",
    "global_embedding",
    "hungarian_from_plan",
    "masked_plan",
]
