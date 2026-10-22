"""Prefix loss tests. No optimizer.step. Failures block training."""
from __future__ import annotations

import numpy as np
import torch

from wt.htw_phase943.set_context import UNDECIDED
from wt.htw_phase943.set_likelihood_dp import _log_softmax_legal
from wt.htw_phase945.utility_targets import soft_targets
from wt.round30.ppo import seed_everything

from .prefix_loss import prefix_nll_enum_toy, prefix_nll_subset_toy, prefix_weights
from .protocol import ENUM_TOL, PERM_TOL, TAU


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


def _enum_assignments(n_uav: int, buoys: list[int]) -> list[np.ndarray]:
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


def _store_logp(store, n_uav, n_buoy):
    def logp_fn(partial):
        key = tuple(int(x) for x in np.asarray(partial, dtype=np.int64).tolist())
        if key not in store:
            store[key] = torch.randn(n_uav, n_buoy + 1)
        return _log_softmax_legal(store[key], _toy_legal(partial, n_uav, n_buoy))

    return logp_fn


def enum_equivalence(n: int = 100, n_uav: int = 3, n_buoy: int = 4) -> dict:
    assignments = _enum_assignments(n_uav, list(range(n_buoy)))
    rows = []
    ok = True
    worst = 0.0
    for i in range(n):
        seed_everything(1000 + i)
        store = {}
        logp_fn = _store_logp(store, n_uav, n_buoy)
        idx = np.random.choice(len(assignments), size=min(5, len(assignments)), replace=False)
        cands = np.stack([assignments[int(j)] for j in idx], axis=0)
        q = soft_targets(np.random.randn(cands.shape[0]) * 1.0e6, TAU)
        enum = torch.zeros(())
        subset = torch.zeros(())
        for j in range(cands.shape[0]):
            enum = enum + float(q[j]) * prefix_nll_enum_toy(logp_fn, cands[j], n_uav, n_buoy)
            subset = subset + float(q[j]) * prefix_nll_subset_toy(logp_fn, cands[j], n_uav, n_buoy)
        delta = float(abs(enum - subset))
        worst = max(worst, delta)
        passed = delta < ENUM_TOL
        ok = ok and passed
        if i < 5 or not passed:
            rows.append({"i": i, "delta": delta, "pass": passed, "q_sum": float(q.sum())})
    return {"n": n, "pass": ok, "worst_delta": worst, "tolerance": ENUM_TOL, "rows": rows}


def permutation_tests(n: int = 20, n_uav: int = 3, n_buoy: int = 4) -> dict:
    assignments = _enum_assignments(n_uav, list(range(n_buoy)))
    rows = []
    cand_ok = True
    uav_ok = True
    buoy_ok = True
    for i in range(n):
        seed_everything(2000 + i)
        store: dict[tuple[int, ...], torch.Tensor] = {}
        base_fn = _store_logp(store, n_uav, n_buoy)
        idx = np.random.choice(len(assignments), size=4, replace=False)
        cands = np.stack([assignments[int(j)] for j in idx], axis=0)
        qv = soft_targets(np.linspace(0, 3, 4) * 1.0e6, TAU)
        loss = lambda c, q, fn: sum(float(q[j]) * float(prefix_nll_subset_toy(fn, c[j], n_uav, n_buoy)) for j in range(c.shape[0]))
        base = loss(cands, qv, base_fn)
        order = np.random.permutation(4)
        d_c = abs(loss(cands[order], qv[order], base_fn) - base)

        perm_u = np.random.permutation(n_uav)
        inv_u = np.argsort(perm_u)

        def fn_u(partial):
            orig = np.asarray(partial, dtype=np.int64)[inv_u]
            key = tuple(int(x) for x in orig.tolist())
            if key not in store:
                store[key] = torch.randn(n_uav, n_buoy + 1)
            raw = store[key][perm_u]
            return _log_softmax_legal(raw, _toy_legal(partial, n_uav, n_buoy))

        cands_u = cands[:, perm_u]
        d_u = abs(loss(cands_u, qv, fn_u) - base)

        perm_b = np.random.permutation(n_buoy)
        inv_b = np.argsort(perm_b)

        def fn_b(partial):
            orig = np.asarray(partial, dtype=np.int64).copy()
            for uav in range(n_uav):
                if 0 <= int(orig[uav]) < n_buoy:
                    orig[uav] = int(inv_b[int(orig[uav])])
            key = tuple(int(x) for x in orig.tolist())
            if key not in store:
                store[key] = torch.randn(n_uav, n_buoy + 1)
            raw = store[key].clone()
            raw[:, :n_buoy] = store[key][:, inv_b]
            return _log_softmax_legal(raw, _toy_legal(partial, n_uav, n_buoy))

        cands_b = cands.copy()
        for r in range(cands_b.shape[0]):
            for uav in range(n_uav):
                if int(cands_b[r, uav]) >= 0:
                    cands_b[r, uav] = int(perm_b[int(cands_b[r, uav])])
        d_b = abs(loss(cands_b, qv, fn_b) - base)
        cand_ok = cand_ok and d_c < PERM_TOL
        uav_ok = uav_ok and d_u < PERM_TOL
        buoy_ok = buoy_ok and d_b < PERM_TOL
        if i < 5 or d_c >= PERM_TOL or d_u >= PERM_TOL or d_b >= PERM_TOL:
            rows.append({"i": i, "candidate_delta": d_c, "uav_delta": d_u, "buoy_delta": d_b})
    return {
        "pass": bool(cand_ok and uav_ok and buoy_ok),
        "candidate_enumeration_pass": cand_ok,
        "uav_pass": uav_ok,
        "buoy_pass": buoy_ok,
        "n": n,
        "tolerance": PERM_TOL,
        "rows": rows,
    }


def target_audit_on_split(candidates: np.ndarray, utilities: np.ndarray, buoy_mask: np.ndarray) -> dict:
    q = soft_targets(utilities, TAU)
    _, edges, meta = prefix_weights(candidates, q, buoy_mask)
    sums = []
    for edge_w in edges:
        s = float(sum(edge_w.values()))
        sums.append(s)
    return {
        "illegal_target_edges": int(meta["illegal_target_edges"]),
        "n_prefix": int(meta["n_prefix"]),
        "n_checked": int(meta["n_checked"]),
        "all_legal": int(meta["illegal_target_edges"]) == 0,
        "q_sum": float(q.sum()),
        "n_mass_ok": int(meta["n_mass_ok"]),
    }


def gradient_sanity() -> dict:
    n_uav, n_buoy = 3, 4
    seed_everything(7)
    store = {}
    key0 = tuple(int(UNDECIDED) for _ in range(n_uav))
    store[key0] = torch.randn(n_uav, n_buoy + 1, requires_grad=True)
    gamma = np.array([0, 1, -1], dtype=np.int64)
    logp_fn = _store_logp(store, n_uav, n_buoy)
    loss = prefix_nll_subset_toy(logp_fn, gamma, n_uav, n_buoy)
    loss.backward()
    grad = store[key0].grad
    # target root edges: UAV0->0, UAV1->1, UAV2->idle
    signs = {
        "u0_b0": float(grad[0, 0]),
        "u1_b1": float(grad[1, 1]),
        "u2_idle": float(grad[2, n_buoy]),
    }
    # NLL: dL/d logit_target should be negative (increase target logit decreases loss)
    ok = signs["u0_b0"] < 0 and signs["u1_b1"] < 0 and signs["u2_idle"] < 0
    return {"pass": ok, "root_grads": signs, "loss": float(loss.detach()), "optimizer_step": False}


__all__ = ["enum_equivalence", "gradient_sanity", "permutation_tests", "target_audit_on_split"]
