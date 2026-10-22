"""Diagnostic beam over sequential legal (uav, token) choices. Does not train."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from wt.htw_phase9.const import MAX_UAVS
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import UNDECIDED, all_candidate_physics_set, apply_choice, empty_assignment, legal_candidate_mask


def _step_logp(model: OrderFreeAssignmentModel, encoded: dict, snap, assignment: np.ndarray, buoy_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    physics = torch.as_tensor(all_candidate_physics_set(snap, assignment)[None, ...], dtype=torch.float32)
    logits = model.decoder(encoded, assignment, physics, buoy_mask)
    legal = legal_candidate_mask(assignment, buoy_mask)
    legal_t = torch.as_tensor(legal.reshape(-1), dtype=torch.bool)
    flat = logits.reshape(-1)
    masked = torch.where(legal_t, flat, torch.full_like(flat, -1.0e9))
    logp = F.log_softmax(masked, dim=0).reshape(logits.shape).detach().cpu().numpy()
    return logp, legal


def beam_assignments(model: OrderFreeAssignmentModel, encoded: dict, snap, buoy_mask: np.ndarray, width: int) -> list[tuple[float, np.ndarray]]:
    beam: list[tuple[float, np.ndarray]] = [(0.0, empty_assignment(MAX_UAVS))]
    for _ in range(MAX_UAVS):
        expanded: list[tuple[float, np.ndarray]] = []
        for path_lp, assignment in beam:
            if np.all(assignment != UNDECIDED):
                expanded.append((path_lp, assignment))
                continue
            logp, legal = _step_logp(model, encoded, snap, assignment, buoy_mask)
            uavs, tokens = np.where(legal)
            for uav, token in zip(uavs.tolist(), tokens.tolist()):
                nxt = apply_choice(assignment, int(uav), int(token))
                expanded.append((path_lp + float(logp[int(uav), int(token)]), nxt))
        merged: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
        for path_lp, assignment in expanded:
            key = tuple(int(x) for x in np.asarray(assignment, dtype=np.int64).tolist())
            if key not in merged or path_lp > merged[key][0]:
                merged[key] = (path_lp, assignment)
        beam = sorted(merged.values(), key=lambda item: item[0], reverse=True)[: int(width)]
    out = []
    for path_lp, assignment in beam:
        done = np.where(assignment == UNDECIDED, -1, assignment).astype(np.int64)
        out.append((path_lp, done))
    return out


__all__ = ["beam_assignments"]
