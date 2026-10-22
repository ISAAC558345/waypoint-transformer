"""Absolute set-likelihood utility loss. Ranking and reference unchanged."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn.functional import softplus

from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_likelihood_dp import set_nll
from wt.htw_phase945.protocol import LAMBDA_RANK, LAMBDA_REF, TAU
from wt.htw_phase945.utility_loss import candidate_logprobs
from wt.htw_phase945.utility_targets import ranking_pairs, soft_targets


def utility_losses_abs(
    model: OrderFreeAssignmentModel,
    encoded: dict,
    snap,
    candidates: np.ndarray,
    utilities: np.ndarray,
    reference: np.ndarray,
    buoy_mask: np.ndarray,
) -> dict[str, torch.Tensor]:
    logp = candidate_logprobs(model, encoded, snap, candidates, buoy_mask)
    q = torch.as_tensor(soft_targets(utilities, TAU), dtype=torch.float32, device=logp.device)
    log_pool = torch.logsumexp(logp, dim=0)
    log_cond = logp - log_pool
    l_cond = -(q * log_cond).sum()
    l_abs = -(q * logp).sum()
    support_penalty = -log_pool
    l_rank = torch.zeros((), dtype=torch.float32, device=logp.device)
    for i, j, y, w in ranking_pairs(utilities):
        l_rank = l_rank + float(w) * softplus(-float(y) * (logp[i] - logp[j]))
    l_ref = set_nll(model, encoded, snap, np.asarray(reference, dtype=np.int64), buoy_mask)
    total = l_abs + float(LAMBDA_RANK) * l_rank + float(LAMBDA_REF) * l_ref
    return {
        "total": total,
        "utility_abs": l_abs,
        "utility_cond": l_cond,
        "support_penalty": support_penalty,
        "log_pool_mass": log_pool,
        "rank": l_rank,
        "ref": l_ref,
        "logp": logp,
        "q": q,
    }


def losses_from_logp(logp: torch.Tensor, utilities: np.ndarray) -> dict[str, torch.Tensor]:
    q = torch.as_tensor(soft_targets(utilities, TAU), dtype=torch.float32, device=logp.device)
    log_pool = torch.logsumexp(logp, dim=0)
    l_cond = -(q * (logp - log_pool)).sum()
    l_abs = -(q * logp).sum()
    return {"q": q, "log_pool": log_pool, "l_cond": l_cond, "l_abs": l_abs, "support_penalty": -log_pool}


__all__ = ["losses_from_logp", "utility_losses_abs"]
