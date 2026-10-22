"""Exact order-marginalized set likelihood: subset DP and 6! enumeration."""
from __future__ import annotations

import itertools

import numpy as np
import torch
from torch.nn import functional as F

from wt.htw_phase9.const import IDLE_INDEX, MAX_BUOYS, MAX_UAVS

from .order_free_decoder import OrderFreeAssignmentModel
from .set_context import UNDECIDED, all_candidate_physics_set, apply_choice, empty_assignment, legal_candidate_mask


def _target_token(assignment: np.ndarray, uav: int) -> int:
    buoy = int(assignment[uav])
    return IDLE_INDEX if buoy < 0 else buoy


def _log_softmax_legal(logits: torch.Tensor, legal: np.ndarray) -> torch.Tensor:
    flat = logits.reshape(-1)
    legal_t = torch.as_tensor(legal.reshape(-1), dtype=torch.bool, device=logits.device)
    masked = torch.where(legal_t, flat, torch.full_like(flat, -1.0e9))
    return F.log_softmax(masked, dim=0).reshape(logits.shape)


def score_state(model: OrderFreeAssignmentModel, encoded: dict, snap, assignment: np.ndarray, buoy_mask: np.ndarray) -> torch.Tensor:
    physics = torch.as_tensor(all_candidate_physics_set(snap, assignment)[None, ...], dtype=torch.float32, device=next(model.parameters()).device)
    return model.decoder(encoded, assignment, physics, buoy_mask)


def subset_dp_logprob(model: OrderFreeAssignmentModel, encoded: dict, snap, target: np.ndarray, buoy_mask: np.ndarray) -> torch.Tensor:
    n_uav = int(target.shape[0])
    n_mask = 1 << n_uav
    assignments = []
    for mask in range(n_mask):
        partial = empty_assignment(n_uav)
        for uav in range(n_uav):
            if mask & (1 << uav):
                partial[uav] = -1 if int(target[uav]) < 0 else int(target[uav])
        assignments.append(partial)
    physics = np.stack([all_candidate_physics_set(snap, item) for item in assignments], axis=0)
    device = next(model.parameters()).device
    phys_t = torch.as_tensor(physics, dtype=torch.float32, device=device)
    logits = model.decoder.forward_many(encoded, assignments, phys_t, buoy_mask)
    cache = []
    for mask in range(n_mask):
        cache.append(_log_softmax_legal(logits[mask], legal_candidate_mask(assignments[mask], buoy_mask)))
    f_vals: list[torch.Tensor | None] = [None] * n_mask
    f_vals[0] = torch.zeros((), dtype=torch.float32, device=device)
    for mask in range(n_mask):
        if f_vals[mask] is None:
            continue
        remaining = [uav for uav in range(n_uav) if not (mask & (1 << uav))]
        if not remaining:
            continue
        logp = cache[mask]
        for uav in remaining:
            token = _target_token(target, uav)
            nxt = mask | (1 << uav)
            term = f_vals[mask] + logp[uav, token]
            if f_vals[nxt] is None:
                f_vals[nxt] = term
            else:
                f_vals[nxt] = torch.logaddexp(f_vals[nxt], term)
    if f_vals[n_mask - 1] is None:
        return torch.tensor(-1.0e9, device=device)
    return f_vals[n_mask - 1]


def exhaustive_logprob(model: OrderFreeAssignmentModel, encoded: dict, snap, target: np.ndarray, buoy_mask: np.ndarray) -> torch.Tensor:
    n_uav = int(target.shape[0])
    total = None
    orderings = itertools.permutations(range(n_uav))
    for order in orderings:
        partial = empty_assignment(n_uav)
        logp_path = torch.zeros((), dtype=torch.float32)
        valid = True
        for uav in order:
            logits = score_state(model, encoded, snap, partial, buoy_mask)
            logp = _log_softmax_legal(logits, legal_candidate_mask(partial, buoy_mask))
            token = _target_token(target, uav)
            if not bool(legal_candidate_mask(partial, buoy_mask)[uav, token]):
                valid = False
                break
            logp_path = logp_path + logp[uav, token]
            partial = apply_choice(partial, uav, token)
        if not valid:
            continue
        total = logp_path if total is None else torch.logaddexp(total, logp_path)
    if total is None:
        return torch.tensor(-1.0e9)
    return total


def set_nll(model: OrderFreeAssignmentModel, encoded: dict, snap, target: np.ndarray, buoy_mask: np.ndarray) -> torch.Tensor:
    return -subset_dp_logprob(model, encoded, snap, target, buoy_mask)


__all__ = ["exhaustive_logprob", "score_state", "set_nll", "subset_dp_logprob"]
