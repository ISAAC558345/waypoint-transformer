"""Executor-aware utility loss. Ranking and reference unchanged. No tau / prefix / cond softmax labels."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn.functional import softplus

from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_likelihood_dp import set_nll
from wt.htw_phase945.protocol import LAMBDA_RANK, LAMBDA_REF
from wt.htw_phase945.utility_loss import candidate_logprobs
from wt.htw_phase945.utility_targets import ranking_pairs

from .protocol import Q_EPS


def executor_targets(utilities: np.ndarray, eps: float = Q_EPS) -> np.ndarray:
    u = np.asarray(utilities, dtype=np.float64).reshape(-1)
    if np.any(u <= 0.0):
        u = u - float(np.min(u)) + float(eps)
    total = float(np.sum(u))
    if (not np.isfinite(total)) or total <= 0.0:
        u = np.asarray(utilities, dtype=np.float64).reshape(-1)
        u = u - float(np.min(u)) + float(eps)
        total = float(np.sum(u))
    return u / total


def utility_losses_exec(
    model: OrderFreeAssignmentModel,
    encoded: dict,
    snap,
    candidates: np.ndarray,
    utilities: np.ndarray,
    reference: np.ndarray,
    buoy_mask: np.ndarray,
) -> dict[str, torch.Tensor]:
    logp = candidate_logprobs(model, encoded, snap, candidates, buoy_mask)
    q = torch.as_tensor(executor_targets(utilities), dtype=torch.float32, device=logp.device)
    l_exec = -(q * logp).sum()
    l_rank = torch.zeros((), dtype=torch.float32, device=logp.device)
    for i, j, y, w in ranking_pairs(utilities):
        l_rank = l_rank + float(w) * softplus(-float(y) * (logp[i] - logp[j]))
    l_ref = set_nll(model, encoded, snap, np.asarray(reference, dtype=np.int64), buoy_mask)
    total = l_exec + float(LAMBDA_RANK) * l_rank + float(LAMBDA_REF) * l_ref
    return {
        "total": total,
        "utility_exec": l_exec,
        "rank": l_rank,
        "ref": l_ref,
        "logp": logp,
        "q": q,
    }


__all__ = ["executor_targets", "utility_losses_exec"]
