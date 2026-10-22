"""Exact toy set-MAP vs greedy/beam recovery. Random sequential logits. No training."""
from __future__ import annotations

import numpy as np
import torch

from wt.htw_phase943.set_context import UNDECIDED
from wt.htw_phase943.set_likelihood_dp import _log_softmax_legal
from wt.round30.ppo import seed_everything

from .protocol import TOY_N_BUOY, TOY_N_UAV, TOY_SEEDS


def _toy_legal(partial: np.ndarray, n_uav: int, n_buoy: int) -> np.ndarray:
    mask = np.zeros((n_uav, n_buoy + 1), dtype=bool)
    occupied = np.zeros(n_buoy, dtype=bool)
    for buoy in partial:
        if 0 <= int(buoy) < n_buoy:
            occupied[int(buoy)] = True
    for uav in range(n_uav):
        if int(partial[uav]) != UNDECIDED:
            continue
        mask[uav, :n_buoy] = ~occupied
        mask[uav, n_buoy] = True
    return mask


def _enum(n_uav: int, buoys: list[int]) -> list[np.ndarray]:
    out = []

    def rec(uav: int, taken: set[int], cur: list[int]) -> None:
        if uav == n_uav:
            out.append(np.asarray(cur, dtype=np.int64))
            return
        rec(uav + 1, taken, cur + [-1])
        for buoy in buoys:
            if buoy in taken:
                continue
            rec(uav + 1, taken | {buoy}, cur + [int(buoy)])

    rec(0, set(), [])
    return out


def _toy_dp(cache: list[torch.Tensor], target: np.ndarray, n_uav: int, n_buoy: int) -> torch.Tensor:
    n_mask = 1 << n_uav
    f_vals: list[torch.Tensor | None] = [None] * n_mask
    f_vals[0] = torch.zeros(())
    for mask in range(n_mask):
        if f_vals[mask] is None:
            continue
        remaining = [uav for uav in range(n_uav) if not (mask & (1 << uav))]
        if not remaining:
            continue
        logp = cache[mask]
        for uav in remaining:
            token = n_buoy if int(target[uav]) < 0 else int(target[uav])
            nxt = mask | (1 << uav)
            term = f_vals[mask] + logp[uav, token]
            f_vals[nxt] = term if f_vals[nxt] is None else torch.logaddexp(f_vals[nxt], term)
    return f_vals[n_mask - 1] if f_vals[n_mask - 1] is not None else torch.tensor(-1.0e9)


def _cache_for(store, target, n_uav, n_buoy):
    cache = []
    n_mask = 1 << n_uav
    for mask in range(n_mask):
        partial = np.full(n_uav, UNDECIDED, dtype=np.int64)
        for uav in range(n_uav):
            if mask & (1 << uav):
                partial[uav] = -1 if int(target[uav]) < 0 else int(target[uav])
        key = tuple(int(x) for x in partial.tolist())
        if key not in store:
            store[key] = torch.randn(n_uav, n_buoy + 1)
        cache.append(_log_softmax_legal(store[key], _toy_legal(partial, n_uav, n_buoy)))
    return cache


def _logits(store, partial, n_uav, n_buoy):
    key = tuple(int(x) for x in np.asarray(partial, dtype=np.int64).tolist())
    if key not in store:
        store[key] = torch.randn(n_uav, n_buoy + 1)
    legal = _toy_legal(partial, n_uav, n_buoy)
    return _log_softmax_legal(store[key], legal).detach().cpu().numpy(), legal


def _greedy(store, n_uav, n_buoy):
    partial = np.full(n_uav, UNDECIDED, dtype=np.int64)
    path = 0.0
    for _ in range(n_uav):
        if np.all(partial != UNDECIDED):
            break
        logp, legal = _logits(store, partial, n_uav, n_buoy)
        values = np.where(legal, logp, -1.0e12)
        uav, token = np.unravel_index(int(np.argmax(values)), values.shape)
        path += float(logp[int(uav), int(token)])
        nxt = partial.copy()
        nxt[int(uav)] = -1 if int(token) >= n_buoy else int(token)
        partial = nxt
    return np.where(partial == UNDECIDED, -1, partial).astype(np.int64), path


def _beam(store, n_uav, n_buoy, width: int):
    beam = [(0.0, np.full(n_uav, UNDECIDED, dtype=np.int64))]
    for _ in range(n_uav):
        expanded = []
        for path_lp, assignment in beam:
            if np.all(assignment != UNDECIDED):
                expanded.append((path_lp, assignment))
                continue
            logp, legal = _logits(store, assignment, n_uav, n_buoy)
            uavs, tokens = np.where(legal)
            for uav, token in zip(uavs.tolist(), tokens.tolist()):
                nxt = assignment.copy()
                nxt[int(uav)] = -1 if int(token) >= n_buoy else int(token)
                expanded.append((path_lp + float(logp[int(uav), int(token)]), nxt))
        merged = {}
        for path_lp, assignment in expanded:
            key = tuple(int(x) for x in assignment.tolist())
            if key not in merged or path_lp > merged[key][0]:
                merged[key] = (path_lp, assignment)
        beam = sorted(merged.values(), key=lambda item: item[0], reverse=True)[: int(width)]
    out = []
    for path_lp, assignment in beam:
        done = np.where(assignment == UNDECIDED, -1, assignment).astype(np.int64)
        out.append((path_lp, done))
    return out


def toy_map_recovery() -> dict:
    assignments = _enum(TOY_N_UAV, list(range(TOY_N_BUOY)))
    widths = (1, 2, 4, 8, 16)
    hits = {f"greedy": 0, **{f"beam{w}": 0 for w in widths}}
    rows = []
    for seed in TOY_SEEDS:
        seed_everything(int(seed))
        store: dict[tuple[int, ...], torch.Tensor] = {}
        logs = []
        for row in assignments:
            logs.append(_toy_dp(_cache_for(store, row, TOY_N_UAV, TOY_N_BUOY), row, TOY_N_UAV, TOY_N_BUOY))
        stacked = torch.stack(logs)
        map_i = int(torch.argmax(stacked).cpu())
        map_assn = assignments[map_i]
        greedy, _ = _greedy(store, TOY_N_UAV, TOY_N_BUOY)
        g_hit = bool(np.array_equal(greedy, map_assn))
        hits["greedy"] += int(g_hit)
        rec = {"seed": int(seed), "n_assignments": len(assignments), "greedy_hit": g_hit}
        for w in widths:
            beams = _beam(store, TOY_N_UAV, TOY_N_BUOY, w)
            found = any(np.array_equal(a, map_assn) for _, a in beams)
            set_best = None
            best_lp = -1.0e30
            for _, a in beams:
                lp = float(_toy_dp(_cache_for(store, a, TOY_N_UAV, TOY_N_BUOY), a, TOY_N_UAV, TOY_N_BUOY).cpu())
                if lp > best_lp:
                    best_lp = lp
                    set_best = a
            set_hit = bool(np.array_equal(set_best, map_assn))
            path_hit = bool(np.array_equal(beams[0][1], map_assn))
            hits[f"beam{w}"] += int(found)
            rec[f"beam{w}_contains_map"] = found
            rec[f"beam{w}_path_top1_hit"] = path_hit
            rec[f"beam{w}_setbest_hit"] = set_hit
        rows.append(rec)
    n = len(TOY_SEEDS)
    rates = {key: hits[key] / float(n) for key in hits}
    return {
        "n_uav": TOY_N_UAV,
        "n_buoy": TOY_N_BUOY,
        "n_assignments": len(assignments),
        "n_seeds": n,
        "greedy_map_recovery": rates["greedy"],
        "beam_contains_map": {f"beam{w}": hits[f"beam{w}"] / float(n) for w in widths},
        "rows": rows[:5],
        "pass_note": "exact enumeration of unordered assignments; logits are random prefix-conditional",
    }


__all__ = ["toy_map_recovery"]
