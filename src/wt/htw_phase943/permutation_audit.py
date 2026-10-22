"""Structural permutation and candidate-enumeration audits."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wt.htw_phase9.const import IDLE_INDEX, MAX_BUOYS, MAX_UAVS
from wt.htw_phase9.features import build_graph, feature_dict_l3b, stack_feature_dicts
from wt.htw_phase942.decode_order import permute_uav_batch
from wt.l3b.env import L3BEnv

from .evaluation import decode_assignment
from .order_free_decoder import OrderFreeAssignmentModel
from .set_context import assignment_set_tuple, default_buoy_cluster


def _batch(env: L3BEnv, info) -> dict[str, np.ndarray]:
    feats = stack_feature_dicts([feature_dict_l3b(info, env)])
    edge, edge_uu = build_graph(info, env)
    feats["edge"] = edge[None, ...]
    feats["edge_uu"] = edge_uu[None, ...]
    return feats


def permute_buoy_batch(batch: dict[str, np.ndarray], perm: np.ndarray) -> dict[str, np.ndarray]:
    out = {key: np.array(value, copy=True) for key, value in batch.items()}
    perm = np.asarray(perm, dtype=np.int64)
    out["buoy"][0] = batch["buoy"][0][perm]
    out["buoy_mask"][0] = batch["buoy_mask"][0][perm]
    out["edge"][0] = batch["edge"][0][:, perm]
    return out


def map_assignment_uav(assignment: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return np.asarray(assignment, dtype=np.int64)[np.asarray(perm, dtype=np.int64)]


def map_assignment_buoy(assignment: np.ndarray, perm: np.ndarray) -> np.ndarray:
    inv = np.argsort(np.asarray(perm, dtype=np.int64))
    out = np.asarray(assignment, dtype=np.int64).copy()
    for uav, buoy in enumerate(out):
        if int(buoy) >= 0:
            out[uav] = int(inv[int(buoy)])
    return out


def sets_equivalent(left: np.ndarray, right: np.ndarray) -> bool:
    return assignment_set_tuple(left) == assignment_set_tuple(right)


def audit_model(model: OrderFreeAssignmentModel, seeds: tuple[int, ...], n_uav: int = 20, n_buoy: int = 20, n_joint: int = 10) -> dict[str, Any]:
    rng = np.random.default_rng(943)
    model.eval()
    uav_hits = buoy_hits = joint_hits = uav_tot = buoy_tot = joint_tot = 0
    with torch.no_grad():
        for seed in seeds:
            env = L3BEnv()
            info = env.reset(seed=int(seed))
            batch = _batch(env, info)
            base = decode_assignment(model, batch, env)
            for _ in range(int(n_uav)):
                perm = rng.permutation(MAX_UAVS)
                env_u = __import__("copy").deepcopy(env)
                env_u.uav_positions = np.asarray(env.uav_positions, dtype=np.float64)[perm]
                pred = decode_assignment(model, permute_uav_batch(batch, perm), env_u)
                expected = map_assignment_uav(base, perm)
                uav_hits += int(sets_equivalent(pred, expected))
                uav_tot += 1
            for _ in range(int(n_buoy)):
                perm = rng.permutation(MAX_BUOYS)
                env_b = __import__("copy").deepcopy(env)
                env_b.buoy_positions = np.asarray(env.buoy_positions, dtype=np.float64)[perm]
                env_b.buoy_cluster = default_buoy_cluster()[perm]
                pred = decode_assignment(model, permute_buoy_batch(batch, perm), env_b)
                expected = map_assignment_buoy(base, perm)
                buoy_hits += int(sets_equivalent(pred, expected))
                buoy_tot += 1
            for _ in range(int(n_joint)):
                pu = rng.permutation(MAX_UAVS)
                pb = rng.permutation(MAX_BUOYS)
                env_j = __import__("copy").deepcopy(env)
                env_j.uav_positions = np.asarray(env.uav_positions, dtype=np.float64)[pu]
                env_j.buoy_positions = np.asarray(env.buoy_positions, dtype=np.float64)[pb]
                env_j.buoy_cluster = default_buoy_cluster()[pb]
                pred = decode_assignment(model, permute_buoy_batch(permute_uav_batch(batch, pu), pb), env_j)
                expected = map_assignment_buoy(map_assignment_uav(base, pu), pb)
                joint_hits += int(sets_equivalent(pred, expected))
                joint_tot += 1
    return {
        "uav_rate": float(uav_hits / max(uav_tot, 1)),
        "buoy_rate": float(buoy_hits / max(buoy_tot, 1)),
        "joint_rate": float(joint_hits / max(joint_tot, 1)),
        "uav_hits": uav_hits,
        "buoy_hits": buoy_hits,
        "joint_hits": joint_hits,
        "uav_tot": uav_tot,
        "buoy_tot": buoy_tot,
        "joint_tot": joint_tot,
    }


def candidate_order_invariance(model: OrderFreeAssignmentModel, seeds: tuple[int, ...], n_shuffle: int = 100) -> dict[str, Any]:
    """Re-decode with shuffled tied-candidate scan order; physical keys decide ties."""
    hits = tot = 0
    model.eval()
    with torch.no_grad():
        for seed in seeds:
            env = L3BEnv()
            info = env.reset(seed=int(seed))
            batch = _batch(env, info)
            base = decode_assignment(model, batch, env)
            for shuffle in range(int(n_shuffle)):
                again = decode_assignment(model, batch, env, shuffle_seed=int(seed) * 1000 + shuffle)
                hits += int(sets_equivalent(base, again))
                tot += 1
    return {"candidate_order_invariance_rate": float(hits / max(tot, 1)), "hits": hits, "tot": tot}


__all__ = ["audit_model", "candidate_order_invariance", "map_assignment_buoy", "map_assignment_uav", "permute_buoy_batch", "sets_equivalent"]
