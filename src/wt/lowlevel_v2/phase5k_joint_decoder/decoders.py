"""Joint assignment decoders. They read frozen encoder states and do not argmax each UAV alone."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from .protocol import SINKHORN_STEPS


class PairScore(nn.Module):
    """Score every UAV-buoy pair from H_UAV and H_Buoy."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, h_u: Tensor, h_b: Tensor) -> Tensor:
        if h_u.dim() != 3 or h_b.dim() != 3:
            raise ValueError("encoder states must include a batch axis")
        if int(h_u.shape[0]) != 1:
            raise ValueError("joint decoder scores one decision at a time")
        uav_count = int(h_u.shape[1])
        buoy_count = int(h_b.shape[1])
        left = h_u[0].unsqueeze(1).expand(uav_count, buoy_count, -1)
        right = h_b[0].unsqueeze(0).expand(uav_count, buoy_count, -1)
        return self.net(torch.cat((left, right), dim=-1)).squeeze(-1)


def capacity_plan(logits: Tensor, steps: int = SINKHORN_STEPS) -> Tensor:
    """Rectangular plan with row sum 1 and column mass at most 1.

    Dummy UAV rows absorb the unused buoys, so a column is not required to be
    chosen. This is the coupling constraint, not an independent row argmax.
    """
    if logits.dim() != 2:
        raise ValueError("logits must be UAV by buoy")
    uav_count, buoy_count = int(logits.shape[0]), int(logits.shape[1])
    if uav_count > buoy_count:
        raise ValueError("column capacity needs at least one buoy per UAV")
    pad = logits.new_zeros((buoy_count - uav_count, buoy_count))
    full = torch.cat((logits, pad), dim=0)
    plan = torch.exp(full - full.amax(dim=1, keepdim=True))
    for _ in range(int(steps)):
        plan = plan / plan.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        plan = plan / plan.sum(dim=0, keepdim=True).clamp_min(1.0e-8)
    return plan[:uav_count]


def joint_pick(plan: np.ndarray) -> np.ndarray:
    """Pick the largest remaining cell, then remove that UAV and that buoy."""
    matrix = np.asarray(plan, dtype=np.float64)
    uav_count, buoy_count = matrix.shape
    assignment = np.full(uav_count, -1, dtype=np.int64)
    used_uav = np.zeros(uav_count, dtype=bool)
    used_buoy = np.zeros(buoy_count, dtype=bool)
    for _ in range(uav_count):
        best_value = None
        best_pair = None
        for uav in range(uav_count):
            if used_uav[uav]:
                continue
            for buoy in range(buoy_count):
                if used_buoy[buoy]:
                    continue
                value = float(matrix[uav, buoy])
                if best_value is None or value > best_value:
                    best_value = value
                    best_pair = (uav, buoy)
        if best_pair is None:
            break
        assignment[best_pair[0]] = int(best_pair[1])
        used_uav[best_pair[0]] = True
        used_buoy[best_pair[1]] = True
    return assignment


def hungarian_assign(logits: np.ndarray) -> np.ndarray:
    """One-to-one assignment that maximizes the pair scores."""
    matrix = np.asarray(logits, dtype=np.float64)
    uav_count = int(matrix.shape[0])
    rows, cols = linear_sum_assignment(-matrix)
    assignment = np.full(uav_count, -1, dtype=np.int64)
    for uav, buoy in zip(rows, cols):
        assignment[int(uav)] = int(buoy)
    return assignment


def gather_assignment_score(matrix: Tensor, assignments: Tensor, *, logarithmic: bool) -> Tensor:
    """Sum the selected UAV-buoy entries. Idle entries contribute nothing."""
    if assignments.dim() != 2 or int(assignments.shape[1]) != int(matrix.shape[0]):
        raise ValueError("assignments must be [count, num_uavs]")
    chosen = assignments.clamp(min=0)
    valid = assignments >= 0
    rows = torch.arange(int(matrix.shape[0]), device=matrix.device).unsqueeze(0).expand_as(chosen)
    taken = matrix[rows, chosen]
    if logarithmic:
        taken = taken.clamp_min(1.0e-8).log()
    return (taken * valid.to(dtype=taken.dtype)).sum(dim=1)


class JointAssignmentDecoder(nn.Module):
    def __init__(self, mode: str, width: int = 128) -> None:
        super().__init__()
        if mode not in {"global", "matching"}:
            raise ValueError(mode)
        self.mode = mode
        self.pair = PairScore(width)

    def logits(self, h_u: Tensor, h_b: Tensor) -> Tensor:
        return self.pair(h_u, h_b)

    def score_matrix(self, h_u: Tensor, h_b: Tensor) -> Tensor:
        logits = self.logits(h_u, h_b)
        if self.mode == "global":
            return capacity_plan(logits)
        return logits

    def score_assignments(self, h_u: Tensor, h_b: Tensor, assignments: Tensor) -> Tensor:
        matrix = self.score_matrix(h_u, h_b)
        return gather_assignment_score(matrix, assignments, logarithmic=self.mode == "global")

    def decode(self, h_u: Tensor, h_b: Tensor) -> np.ndarray:
        logits = self.logits(h_u, h_b).detach()
        if self.mode == "global":
            return joint_pick(capacity_plan(logits).cpu().numpy())
        return hungarian_assign(logits.cpu().numpy())


__all__ = [
    "JointAssignmentDecoder",
    "capacity_plan",
    "gather_assignment_score",
    "hungarian_assign",
    "joint_pick",
]
