"""Phase-9.3 ranking losses for relation-aware assignment scores."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from wt.htw_phase92.models import IDLE_INDEX


def ranking_loss(scores: Tensor, idle: Tensor, assignment: Tensor, uav_mask: Tensor, buoy_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Oracle positive edges outrank negatives. Returns (loss, pairwise accuracy)."""
    batch, m_count, k_count = scores.shape
    assigned = (assignment >= 0) & uav_mask
    idle_uav = (~assigned) & uav_mask
    gather_index = assignment.clamp(min=0).unsqueeze(-1)
    pos = torch.gather(scores, 2, gather_index).squeeze(-1)
    pos = torch.where(assigned, pos, torch.zeros_like(pos))
    buoy_ok = buoy_mask[:, None, :].expand(-1, m_count, -1)
    neg_buoy = buoy_ok.clone()
    neg_buoy[assigned] = neg_buoy[assigned] & (torch.arange(k_count, device=scores.device)[None, :] != assignment[assigned][:, None])
    delta_bn = pos[:, :, None] - scores
    pair_bn = assigned[:, :, None] & neg_buoy
    loss_bn = -F.logsigmoid(delta_bn)
    delta_idle_neg = idle[:, :, None] - scores
    pair_idle = idle_uav[:, :, None] & buoy_ok
    loss_idle = -F.logsigmoid(delta_idle_neg)
    delta_pos_idle = pos - idle
    pair_pos_idle = assigned
    loss_pos_idle = -F.logsigmoid(delta_pos_idle)
    other_uav = uav_mask[:, :, None] & uav_mask[:, None, :]
    eye = torch.eye(m_count, device=scores.device, dtype=torch.bool)[None, :, :]
    other_uav = other_uav & (~eye)
    k_for_i = assignment.clamp(min=0)[:, :, None].expand(-1, -1, m_count)
    other_at_k = torch.gather(scores[:, None, :, :].expand(-1, m_count, -1, -1), 3, k_for_i.unsqueeze(-1)).squeeze(-1)
    delta_uav = pos[:, :, None] - other_at_k
    pair_comp = assigned[:, :, None] & other_uav
    loss_comp = -F.logsigmoid(delta_uav)
    parts = []
    acc_hits = []
    acc_tot = []
    if pair_bn.any():
        parts.append(loss_bn[pair_bn].mean())
        acc_hits.append((delta_bn[pair_bn] > 0).float().sum())
        acc_tot.append(pair_bn.to(scores.dtype).sum())
    if pair_idle.any():
        parts.append(loss_idle[pair_idle].mean())
        acc_hits.append((delta_idle_neg[pair_idle] > 0).float().sum())
        acc_tot.append(pair_idle.to(scores.dtype).sum())
    if pair_pos_idle.any():
        parts.append(loss_pos_idle[pair_pos_idle].mean())
        acc_hits.append((delta_pos_idle[pair_pos_idle] > 0).float().sum())
        acc_tot.append(pair_pos_idle.to(scores.dtype).sum())
    if pair_comp.any():
        parts.append(loss_comp[pair_comp].mean())
        acc_hits.append((delta_uav[pair_comp] > 0).float().sum())
        acc_tot.append(pair_comp.to(scores.dtype).sum())
    if not parts:
        zero = scores.sum() * 0.0
        return zero, zero
    loss = torch.stack(parts).mean()
    ranking_acc = torch.stack(acc_hits).sum() / torch.stack(acc_tot).sum().clamp_min(1.0)
    return loss, ranking_acc


def hybrid_loss(scores: Tensor, idle: Tensor, assignment: Tensor, uav_mask: Tensor, buoy_mask: Tensor, ce_weight: float = 0.25) -> tuple[Tensor, Tensor, Tensor]:
    target = assignment.clone()
    target = torch.where(assignment >= 0, assignment, torch.full_like(assignment, IDLE_INDEX))
    logits = torch.cat((scores, idle.unsqueeze(-1)), dim=-1)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    weights = uav_mask.reshape(-1).to(nll.dtype)
    ce = (nll * weights).sum() / weights.sum().clamp_min(1.0)
    rank, acc = ranking_loss(scores, idle, assignment, uav_mask, buoy_mask)
    return ce_weight * ce + rank, ce, acc


__all__ = ["hybrid_loss", "ranking_loss"]
