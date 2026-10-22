"""Utility distribution matching + weighted pairwise ranking + small reference NLL anchor."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn.functional import softplus

from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_likelihood_dp import set_nll, subset_dp_logprob

from .protocol import LAMBDA_RANK, LAMBDA_REF, TAU
from .utility_targets import ranking_pairs, soft_targets


def candidate_logprobs(model: OrderFreeAssignmentModel, encoded: dict, snap, candidates: np.ndarray, buoy_mask: np.ndarray) -> torch.Tensor:
    logs = []
    for row in np.asarray(candidates, dtype=np.int64):
        logs.append(subset_dp_logprob(model, encoded, snap, row, buoy_mask))
    return torch.stack(logs, dim=0)


def utility_losses(
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
    log_p = torch.log_softmax(logp, dim=0)
    l_util = -(q * log_p).sum()
    l_rank = torch.zeros((), dtype=torch.float32, device=logp.device)
    for i, j, y, w in ranking_pairs(utilities):
        l_rank = l_rank + float(w) * softplus(-float(y) * (logp[i] - logp[j]))
    l_ref = set_nll(model, encoded, snap, np.asarray(reference, dtype=np.int64), buoy_mask)
    total = l_util + float(LAMBDA_RANK) * l_rank + float(LAMBDA_REF) * l_ref
    return {"total": total, "utility": l_util, "rank": l_rank, "ref": l_ref, "logp": logp}
