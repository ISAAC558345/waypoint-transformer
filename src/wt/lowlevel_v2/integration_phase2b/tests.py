"""Pre-training tests: uniqueness, global normalization, L_abs identity, shift sensitivity."""
from __future__ import annotations

import numpy as np
import torch

from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import UNDECIDED, make_snap
from wt.htw_phase943.set_likelihood_dp import _log_softmax_legal, subset_dp_logprob
from wt.htw_phase943.training import _encode_one
from wt.htw_phase945.utility_targets import soft_targets
from wt.lowlevel_v2.integration_phase2.train_hl import load_align_split
from wt.round30.ppo import seed_everything

from .abs_loss import losses_from_logp
from .protocol import IDENTITY_N, IDENTITY_TOL, NORM_TOL, OLD_HL_CKPT, POOL_SIZE, SHIFT_CS, TAU


def check_unique_candidates(train: dict, val: dict) -> dict:
    rows = []
    dup_states = 0
    for split, data in (("train", train), ("val", val)):
        for index, hashes in enumerate(data["hashes"]):
            n = len(hashes)
            n_u = len(set(hashes))
            if n_u != n:
                dup_states += 1
            rows.append({"split": split, "index": index, "n": n, "unique": n_u, "duplicate": n_u != n})
    return {
        "candidate_duplicate_weighting_detected": bool(dup_states > 0),
        "n_states": len(rows),
        "n_duplicate_states": dup_states,
        "pool_size": int(POOL_SIZE),
        "items": rows[:8],
    }


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
    if f_vals[n_mask - 1] is None:
        return torch.tensor(-1.0e9)
    return f_vals[n_mask - 1]


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


def toy_global_normalization(n_uav: int = 3, n_buoy: int = 4, seed: int = 2) -> dict:
    seed_everything(int(seed))
    store: dict[tuple[int, ...], torch.Tensor] = {}

    def logits_for(partial: np.ndarray) -> torch.Tensor:
        key = tuple(int(x) for x in np.asarray(partial, dtype=np.int64).tolist())
        if key not in store:
            store[key] = torch.randn(n_uav, n_buoy + 1)
        return store[key]

    def cache_for(target: np.ndarray) -> list[torch.Tensor]:
        cache = []
        n_mask = 1 << n_uav
        for mask in range(n_mask):
            partial = np.full(n_uav, UNDECIDED, dtype=np.int64)
            for uav in range(n_uav):
                if mask & (1 << uav):
                    partial[uav] = -1 if int(target[uav]) < 0 else int(target[uav])
            cache.append(_log_softmax_legal(logits_for(partial), _toy_legal(partial, n_uav, n_buoy)))
        return cache

    assignments = _enum_assignments(n_uav, list(range(n_buoy)))
    logs = [_toy_dp(cache_for(row), row, n_uav, n_buoy) for row in assignments]
    total = torch.logsumexp(torch.stack(logs), dim=0)
    mass = float(torch.exp(total).cpu())
    return {
        "n_uav": n_uav,
        "n_buoy": n_buoy,
        "n_assignments": len(assignments),
        "sum_P": mass,
        "logsumexp": float(total.cpu()),
        "pass": abs(mass - 1.0) < 1.0e-4,
        "recurrence": "identical to subset_dp_logprob: legal log-softmax on partial then order-marginalized DP",
    }


def production_restricted_normalization(n_buoy: int = 3) -> dict:
    from wt.lowlevel_v2.integration_phase2.protocol import OLD_HL_CKPT

    data = load_align_split("hl_rl_align_val")
    model = OrderFreeAssignmentModel("deepsets")
    model.load_state_dict(torch.load(OLD_HL_CKPT, map_location="cpu"))
    model.eval()
    index = 0
    encoded = _encode_one(model, data, index, "cpu")
    snap = make_snap(data["uav_positions"][index], data["buoy_positions"][index])
    buoy_mask = np.zeros_like(data["buoy_mask"][index])
    buoy_mask[:n_buoy] = True
    n_uav = 6
    assignments = _enum_assignments(n_uav, list(range(n_buoy)))
    logs = []
    with torch.no_grad():
        for row in assignments:
            logs.append(subset_dp_logprob(model, encoded, snap, row, buoy_mask))
    total = torch.logsumexp(torch.stack(logs), dim=0)
    mass = float(torch.exp(total).clamp_max(10.0).cpu())
    return {
        "n_uav": n_uav,
        "n_available_buoys": n_buoy,
        "n_assignments": len(assignments),
        "sum_P": mass,
        "logsumexp": float(total.cpu()),
        "pass": abs(mass - 1.0) < 1.0e-4,
        "decoder": "OrderFreeSetDecoder + production subset_dp_logprob",
        "note": "restricted buoy_mask to make enumeration feasible; sequential legal softmax implies partition function 1 on the full space",
    }


def identity_and_shift_tests() -> tuple[dict, dict]:
    seed_everything(7)
    id_rows = []
    id_ok = True
    for i in range(IDENTITY_N):
        logp = torch.randn(int(POOL_SIZE))
        utilities = np.random.randn(int(POOL_SIZE)).astype(np.float64) * 1.0e6
        packed = losses_from_logp(logp, utilities)
        qsum = float(packed["q"].sum().cpu())
        delta = float((packed["l_abs"] - (packed["l_cond"] - packed["log_pool"])).abs().cpu())
        ok = abs(qsum - 1.0) < 1.0e-6 and delta < IDENTITY_TOL
        id_ok = id_ok and ok
        if i < 5 or not ok:
            id_rows.append({"i": i, "q_sum": qsum, "abs_minus_cond_plus_logpool": delta, "pass": ok})
    id_payload = {"n": IDENTITY_N, "pass": id_ok, "tolerance": IDENTITY_TOL, "rows": id_rows, "identity": "L_abs = L_cond - logPoolMass"}

    logp = torch.randn(int(POOL_SIZE))
    utilities = np.linspace(0, 4.0e6, int(POOL_SIZE))
    base = losses_from_logp(logp, utilities)
    shift_rows = []
    shift_ok = True
    for const in (0.0,) + SHIFT_CS:
        packed = losses_from_logp(logp + const, utilities)
        cond_delta = float((packed["l_cond"] - base["l_cond"]).abs().cpu())
        abs_delta = float((packed["l_abs"] - (base["l_abs"] - const)).abs().cpu())
        ok_c = cond_delta < IDENTITY_TOL
        ok_a = abs_delta < IDENTITY_TOL
        shift_ok = shift_ok and ok_c and ok_a
        shift_rows.append(
            {
                "c": const,
                "L_cond": float(packed["l_cond"].cpu()),
                "L_abs": float(packed["l_abs"].cpu()),
                "cond_delta_vs_0": cond_delta,
                "abs_matches_minus_c": ok_a,
                "cond_invariant": ok_c,
            }
        )
    shift_payload = {
        "pass": shift_ok,
        "rows": shift_rows,
        "theory": "L_cond invariant; L_abs(new)=L_abs(old)-c",
        "tau": TAU,
    }
    return id_payload, shift_payload


__all__ = [
    "check_unique_candidates",
    "identity_and_shift_tests",
    "production_restricted_normalization",
    "toy_global_normalization",
]
