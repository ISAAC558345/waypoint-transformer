"""Prefix-consistent utility loss. Ranking and reference unchanged. Replaces L_abs only."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
from torch.nn.functional import softplus

from wt.htw_phase9.const import IDLE_INDEX, MAX_UAVS
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import UNDECIDED, all_candidate_physics_set, empty_assignment, legal_candidate_mask
from wt.htw_phase943.set_likelihood_dp import _log_softmax_legal, set_nll
from wt.htw_phase945.protocol import LAMBDA_RANK, LAMBDA_REF, TAU
from wt.htw_phase945.utility_loss import candidate_logprobs
from wt.htw_phase945.utility_targets import ranking_pairs, soft_targets


def _token(buoy: int) -> int:
    return IDLE_INDEX if int(buoy) < 0 else int(buoy)


def _partial(gamma: np.ndarray, mask: int, n_uav: int) -> np.ndarray:
    partial = empty_assignment(n_uav)
    for uav in range(n_uav):
        if mask & (1 << uav):
            partial[uav] = -1 if int(gamma[uav]) < 0 else int(gamma[uav])
    return partial


def prefix_weights(
    candidates: np.ndarray,
    q: np.ndarray | torch.Tensor,
    buoy_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[dict[tuple[int, int], float]], dict]:
    """Merged prefix lattice. Weights equal per-candidate exact subset expectation."""
    gammas = np.asarray(candidates, dtype=np.int64)
    q_np = np.asarray(q.detach().cpu() if isinstance(q, torch.Tensor) else q, dtype=np.float64)
    n_uav = int(gammas.shape[1])
    avail = None if buoy_mask is None else np.asarray(buoy_mask, dtype=bool)
    buckets: dict[tuple[int, ...], dict[tuple[int, int], float]] = {}
    illegal = 0
    n_checked = 0
    n_mass_ok = 0
    for j, gamma in enumerate(gammas):
        qj = float(q_np[j])
        if qj <= 0.0:
            continue
        for mask in range(1 << n_uav):
            k = bin(mask).count("1")
            if k >= n_uav:
                continue
            w_s = 1.0 / float(math.comb(n_uav, k))
            t_e = 1.0 / float(n_uav - k)
            partial = _partial(gamma, mask, n_uav)
            key = tuple(int(x) for x in partial.tolist())
            legal = legal_candidate_mask(partial, avail)
            edge_w = buckets.setdefault(key, {})
            remaining = [uav for uav in range(n_uav) if not (mask & (1 << uav))]
            local = 0.0
            for uav in remaining:
                token = _token(gamma[uav])
                n_checked += 1
                if n_uav == MAX_UAVS and not bool(legal[int(uav), int(token)]):
                    illegal += 1
                    continue
                edge_w[(int(uav), int(token))] = edge_w.get((int(uav), int(token)), 0.0) + qj * w_s * t_e
                local += t_e
            if abs(local - 1.0) < 1.0e-8:
                n_mass_ok += 1
    keys = list(buckets.keys())
    partials = [np.asarray(key, dtype=np.int64) for key in keys]
    edges = [buckets[key] for key in keys]
    return partials, edges, {"illegal_target_edges": illegal, "n_prefix": len(partials), "n_checked": n_checked, "n_mass_ok": n_mass_ok}


def prefix_nll_from_logits(logits: torch.Tensor, edges: list[dict[tuple[int, int], float]], buoy_mask: np.ndarray, partials: list[np.ndarray]) -> torch.Tensor:
    loss = torch.zeros((), dtype=torch.float32, device=logits.device)
    for i, edge_w in enumerate(edges):
        legal = legal_candidate_mask(partials[i], buoy_mask)
        logp = _log_softmax_legal(logits[i], legal)
        for (uav, token), weight in edge_w.items():
            if weight <= 0.0:
                continue
            if not bool(legal[int(uav), int(token)]):
                continue
            loss = loss - float(weight) * logp[int(uav), int(token)]
    return loss


def prefix_utility_nll(model: OrderFreeAssignmentModel, encoded: dict, snap, candidates: np.ndarray, q: torch.Tensor, buoy_mask: np.ndarray) -> tuple[torch.Tensor, dict]:
    partials, edges, meta = prefix_weights(candidates, q, buoy_mask)
    if not partials:
        return torch.zeros((), dtype=torch.float32), {"n_prefix": 0, **meta}
    physics = np.stack([all_candidate_physics_set(snap, item) for item in partials], axis=0)
    device = next(model.parameters()).device
    phys_t = torch.as_tensor(physics, dtype=torch.float32, device=device)
    logits = model.decoder.forward_many(encoded, partials, phys_t, buoy_mask)
    loss = prefix_nll_from_logits(logits, edges, buoy_mask, partials)
    info = dict(meta)
    info["n_prefix"] = len(partials)
    return loss, info


def utility_losses_prefix(
    model: OrderFreeAssignmentModel,
    encoded: dict,
    snap,
    candidates: np.ndarray,
    utilities: np.ndarray,
    reference: np.ndarray,
    buoy_mask: np.ndarray,
) -> dict[str, torch.Tensor]:
    q = torch.as_tensor(soft_targets(utilities, TAU), dtype=torch.float32, device=next(model.parameters()).device)
    l_prefix, info = prefix_utility_nll(model, encoded, snap, candidates, q, buoy_mask)
    logp = candidate_logprobs(model, encoded, snap, candidates, buoy_mask)
    l_rank = torch.zeros((), dtype=torch.float32, device=logp.device)
    for i, j, y, w in ranking_pairs(utilities):
        l_rank = l_rank + float(w) * softplus(-float(y) * (logp[i] - logp[j]))
    l_ref = set_nll(model, encoded, snap, np.asarray(reference, dtype=np.int64), buoy_mask)
    total = l_prefix + float(LAMBDA_RANK) * l_rank + float(LAMBDA_REF) * l_ref
    l_abs = -(q.to(logp.device) * logp).sum()
    return {
        "total": total,
        "utility_prefix": l_prefix,
        "utility_abs": l_abs,
        "rank": l_rank,
        "ref": l_ref,
        "logp": logp,
        "q": q,
        "n_prefix": torch.tensor(float(info.get("n_prefix", 0.0))),
    }


def prefix_nll_enum_toy(logp_fn, gamma: np.ndarray, n_uav: int, n_buoy: int) -> torch.Tensor:
    import itertools

    from wt.htw_phase943.set_context import apply_choice

    orders = list(itertools.permutations(range(n_uav)))
    acc = None
    for order in orders:
        partial = np.full(n_uav, UNDECIDED, dtype=np.int64)
        path = torch.zeros(())
        for uav in order:
            logp = logp_fn(partial)
            token = n_buoy if int(gamma[uav]) < 0 else int(gamma[uav])
            path = path + logp[int(uav), int(token)]
            nxt = np.asarray(partial, dtype=np.int64).copy()
            nxt[int(uav)] = -1 if int(token) >= n_buoy else int(token)
            partial = nxt
        acc = path if acc is None else acc + path
    return -(acc / float(len(orders)))


def prefix_nll_subset_toy(logp_fn, gamma: np.ndarray, n_uav: int, n_buoy: int) -> torch.Tensor:
    loss = torch.zeros(())
    for mask in range(1 << n_uav):
        k = bin(mask).count("1")
        if k >= n_uav:
            continue
        w_s = 1.0 / float(math.comb(n_uav, k))
        t_e = 1.0 / float(n_uav - k)
        partial = np.full(n_uav, UNDECIDED, dtype=np.int64)
        for uav in range(n_uav):
            if mask & (1 << uav):
                partial[uav] = -1 if int(gamma[uav]) < 0 else int(gamma[uav])
        logp = logp_fn(partial)
        remaining = [uav for uav in range(n_uav) if not (mask & (1 << uav))]
        for uav in remaining:
            token = n_buoy if int(gamma[uav]) < 0 else int(gamma[uav])
            loss = loss - w_s * t_e * logp[int(uav), int(token)]
    return loss


__all__ = [
    "prefix_nll_enum_toy",
    "prefix_nll_subset_toy",
    "prefix_utility_nll",
    "prefix_weights",
    "utility_losses_prefix",
]
