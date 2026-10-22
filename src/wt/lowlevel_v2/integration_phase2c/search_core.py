"""Frozen search diagnostics. No training. No utility rerank of beams."""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from torch.nn import functional as F

from wt.htw_phase9.const import IDLE_INDEX, MAX_UAVS
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel, pick_candidate
from wt.htw_phase943.set_context import (
    UNDECIDED,
    all_candidate_physics_set,
    apply_choice,
    empty_assignment,
    legal_candidate_mask,
)
from wt.htw_phase943.set_likelihood_dp import _log_softmax_legal, _target_token
from wt.lowlevel_v2.integration_phase2a.beam import beam_assignments
from wt.lowlevel_v2.integration_phase2a.coverage import idle_count


def step_logp(model: OrderFreeAssignmentModel, encoded: dict, snap, assignment: np.ndarray, buoy_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    physics = torch.as_tensor(all_candidate_physics_set(snap, assignment)[None, ...], dtype=torch.float32)
    logits = model.decoder(encoded, assignment, physics, buoy_mask)
    legal = legal_candidate_mask(assignment, buoy_mask)
    legal_t = torch.as_tensor(legal.reshape(-1), dtype=torch.bool)
    flat = logits.reshape(-1)
    masked = torch.where(legal_t, flat, torch.full_like(flat, -1.0e9))
    logp = F.log_softmax(masked, dim=0).reshape(logits.shape).detach().cpu().numpy()
    return logp, legal


def greedy_trace(model: OrderFreeAssignmentModel, encoded: dict, snap, buoy_mask: np.ndarray) -> dict:
    assignment = empty_assignment(MAX_UAVS)
    path_lp = 0.0
    steps = []
    for step in range(MAX_UAVS):
        if np.all(assignment != UNDECIDED):
            break
        physics = torch.as_tensor(all_candidate_physics_set(snap, assignment)[None, ...], dtype=torch.float32)
        logits = model.decoder(encoded, assignment, physics, buoy_mask)
        logp, legal = step_logp(model, encoded, snap, assignment, buoy_mask)
        uav, token = pick_candidate(logits, snap, assignment, buoy_mask, rng=None)
        idle_lp = float(logp[:, IDLE_INDEX].max()) if legal[:, IDLE_INDEX].any() else -1.0e9
        nonidle = np.where(legal[:, :IDLE_INDEX], logp[:, :IDLE_INDEX], -1.0e12)
        nonidle_lp = float(nonidle.max())
        chosen = float(logp[int(uav), int(token)])
        steps.append(
            {
                "step": step + 1,
                "uav": int(uav),
                "token": int(token),
                "chose_idle": int(token) >= IDLE_INDEX,
                "logp_chosen": chosen,
                "idle_logp": idle_lp,
                "best_nonidle_logp": nonidle_lp,
                "idle_prob": float(np.exp(np.clip(idle_lp, -50.0, 0.0))),
                "best_nonidle_prob": float(np.exp(np.clip(nonidle_lp, -50.0, 0.0))),
            }
        )
        path_lp += chosen
        assignment = apply_choice(assignment, int(uav), int(token))
    done = np.where(assignment == UNDECIDED, -1, assignment).astype(np.int64)
    return {"assignment": done, "logp_path": float(path_lp), "steps": steps}


def target_cache(model: OrderFreeAssignmentModel, encoded: dict, snap, target: np.ndarray, buoy_mask: np.ndarray) -> list[np.ndarray]:
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
    with torch.no_grad():
        logits = model.decoder.forward_many(encoded, assignments, phys_t, buoy_mask)
    cache = []
    for mask in range(n_mask):
        cache.append(_log_softmax_legal(logits[mask], legal_candidate_mask(assignments[mask], buoy_mask)).detach().cpu().numpy())
    return cache


def set_path_stats(cache: list[np.ndarray], target: np.ndarray) -> dict:
    n_uav = int(target.shape[0])
    paths = []
    for order in itertools.permutations(range(n_uav)):
        mask = 0
        lp = 0.0
        valid = True
        for uav in order:
            token = _target_token(target, int(uav))
            legal_token = cache[mask][int(uav), int(token)]
            if not np.isfinite(legal_token) or legal_token < -1.0e8:
                valid = False
                break
            lp += float(legal_token)
            mask |= 1 << int(uav)
        if valid:
            paths.append(lp)
    if not paths:
        return {
            "logp_set": -1.0e9,
            "max_order_logp_path": -1.0e9,
            "mean_order_logp_path": -1.0e9,
            "omb": float("nan"),
            "order_entropy": float("nan"),
            "n_eff": float("nan"),
            "n_valid_orders": 0,
        }
    arr = np.asarray(paths, dtype=np.float64)
    m = float(np.max(arr))
    log_set = m + math.log(float(np.sum(np.exp(arr - m))))
    rel = np.exp(arr - log_set)
    rel = rel / max(float(rel.sum()), 1.0e-300)
    entropy = float(-np.sum(rel * np.log(np.clip(rel, 1.0e-300, 1.0))))
    return {
        "logp_set": float(log_set),
        "max_order_logp_path": float(np.max(arr)),
        "mean_order_logp_path": float(np.mean(arr)),
        "omb": float(log_set - np.max(arr)),
        "order_entropy": entropy,
        "n_eff": float(math.exp(entropy)),
        "n_valid_orders": int(arr.size),
    }


def compatible_with(prefix: np.ndarray, target: np.ndarray) -> bool:
    occupied_tgt = {int(b) for b in target.tolist() if int(b) >= 0}
    taken = set()
    for uav, val in enumerate(np.asarray(prefix, dtype=np.int64).tolist()):
        if int(val) == UNDECIDED:
            continue
        want = int(target[uav])
        got = int(val)
        if got != want:
            return False
        if got >= 0:
            if got in taken or got not in occupied_tgt:
                return False
            taken.add(got)
    return True


def compatible_edges(logp: np.ndarray, legal: np.ndarray, prefix: np.ndarray, target: np.ndarray) -> list[tuple[int, int, float]]:
    out = []
    for uav in range(int(prefix.shape[0])):
        if int(prefix[uav]) != UNDECIDED:
            continue
        token = IDLE_INDEX if int(target[uav]) < 0 else int(target[uav])
        if not bool(legal[uav, token]):
            continue
        nxt = apply_choice(prefix, uav, token)
        if compatible_with(nxt, target):
            out.append((int(uav), int(token), float(logp[uav, token])))
    return out


def prefix_audit(model, encoded, snap, buoy_mask, greedy_steps: list[dict], greedy_assn: np.ndarray, best_pool: np.ndarray) -> dict:
    assignment = empty_assignment(MAX_UAVS)
    first = None
    margins = []
    masses = []
    for rec in greedy_steps:
        logp, legal = step_logp(model, encoded, snap, assignment, buoy_mask)
        edges = compatible_edges(logp, legal, assignment, best_pool)
        mass = float(sum(math.exp(min(lp, 0.0)) for _, _, lp in edges))
        masses.append({"step": rec["step"], "compatible_prefix_mass": mass, "n_compatible_edges": len(edges)})
        best_comp = max((lp for _, _, lp in edges), default=-1.0e9)
        margins.append(
            {
                "step": rec["step"],
                "greedy_logp": rec["logp_chosen"],
                "best_compatible_logp": float(best_comp),
                "prefix_margin": float(rec["logp_chosen"] - best_comp) if edges else float("nan"),
                "still_compatible": bool(edges),
            }
        )
        if first is None and not compatible_with(apply_choice(assignment, rec["uav"], rec["token"]), best_pool):
            first = int(rec["step"])
        assignment = apply_choice(assignment, rec["uav"], rec["token"])
    if first is None and not np.array_equal(np.asarray(greedy_assn, dtype=np.int64)[:MAX_UAVS], np.asarray(best_pool, dtype=np.int64)[:MAX_UAVS]):
        first = MAX_UAVS
    tf_mass = teacher_forced_compatible_mass(model, encoded, snap, buoy_mask, best_pool)
    return {"first_divergence_step": first, "margins": margins, "greedy_compatible_mass": masses, "teacher_forced_mass": tf_mass}


def teacher_forced_compatible_mass(model, encoded, snap, buoy_mask, target: np.ndarray) -> list[dict]:
    n_uav = int(target.shape[0])
    rows = []
    for order in [tuple(range(n_uav))]:
        partial = empty_assignment(n_uav)
        for step, uav in enumerate(order):
            logp, legal = step_logp(model, encoded, snap, partial, buoy_mask)
            edges = compatible_edges(logp, legal, partial, target)
            mass = float(sum(math.exp(min(lp, 0.0)) for _, _, lp in edges))
            rows.append({"order": "index", "step": step + 1, "compatible_prefix_mass": mass, "n_compatible_edges": len(edges)})
            token = IDLE_INDEX if int(target[uav]) < 0 else int(target[uav])
            partial = apply_choice(partial, int(uav), int(token))
    cache = target_cache(model, encoded, snap, target, buoy_mask)
    acc = [0.0] * n_uav
    n_ord = 0
    for order in itertools.permutations(range(n_uav)):
        mask = 0
        n_ord += 1
        for step, uav in enumerate(order):
            logp = cache[mask]
            mass = 0.0
            for other in range(n_uav):
                if mask & (1 << other):
                    continue
                tok = _target_token(target, other)
                mass += float(math.exp(min(float(logp[other, tok]), 0.0)))
            acc[step] += mass
            mask |= 1 << int(uav)
    for step in range(n_uav):
        rows.append({"order": "mean_all_perms", "step": step + 1, "compatible_prefix_mass": acc[step] / max(n_ord, 1), "n_compatible_edges": -1})
    return rows


def finalize(assignment: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(assignment, dtype=np.int64) == UNDECIDED, -1, assignment).astype(np.int64)


def beam_top(model, encoded, snap, buoy_mask, width: int) -> list[tuple[float, np.ndarray]]:
    return beam_assignments(model, encoded, snap, buoy_mask, int(width))


def idle_active(assignment: np.ndarray) -> tuple[int, int]:
    idle = idle_count(assignment)
    return idle, int(MAX_UAVS - idle)


__all__ = [
    "beam_top",
    "compatible_with",
    "finalize",
    "greedy_trace",
    "idle_active",
    "prefix_audit",
    "set_path_stats",
    "step_logp",
    "target_cache",
    "teacher_forced_compatible_mass",
]
