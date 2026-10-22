"""Permutation equivariance audit for DeepSets/HGNN AR models."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from wt.htw_phase9.const import MAX_BUOYS, MAX_UAVS
from wt.htw_phase9.features import build_graph, feature_dict_l3b, stack_feature_dicts
from wt.htw_phase94.autoregressive_decoder import AssignmentARModel
from wt.htw_phase94.evaluation import decode_sequence
from wt.l3b.env import L3BEnv


def _batch(env: L3BEnv, info) -> dict[str, np.ndarray]:
    feats = stack_feature_dicts([feature_dict_l3b(info, env)])
    edge, edge_uu = build_graph(info, env)
    feats["edge"] = edge[None, ...]
    feats["edge_uu"] = edge_uu[None, ...]
    return feats


def _permute_uav(batch: dict[str, np.ndarray], perm: np.ndarray) -> dict[str, np.ndarray]:
    out = {key: np.array(value, copy=True) for key, value in batch.items()}
    inv = np.argsort(perm)
    out["uav"][0] = batch["uav"][0][inv]
    out["uav_mask"][0] = batch["uav_mask"][0][inv]
    out["edge"][0] = batch["edge"][0][inv]
    out["edge_uu"][0] = batch["edge_uu"][0][np.ix_(inv, inv)]
    return out


def _permute_buoy(batch: dict[str, np.ndarray], perm: np.ndarray) -> dict[str, np.ndarray]:
    out = {key: np.array(value, copy=True) for key, value in batch.items()}
    inv = np.argsort(perm)
    out["buoy"][0] = batch["buoy"][0][inv]
    out["buoy_mask"][0] = batch["buoy_mask"][0][inv]
    out["edge"][0] = batch["edge"][0][:, inv]
    return out


def audit_model(model: AssignmentARModel, seed: int = 24, n_perm: int = 4) -> dict[str, Any]:
    env = L3BEnv()
    info = env.reset(seed=int(seed))
    batch = _batch(env, info)
    model.eval()
    rng = np.random.default_rng(941)
    with torch.no_grad():
        base = decode_sequence(model, batch, env, teacher_tokens=None)["assignment"]
        uav_hits = 0
        buoy_hits = 0
        for _ in range(int(n_perm)):
            u_perm = rng.permutation(MAX_UAVS)
            b_perm = rng.permutation(MAX_BUOYS)
            inv_u = np.argsort(u_perm)
            env_u = copy.deepcopy(env)
            env_u.uav_positions = np.asarray(env.uav_positions, dtype=np.float64)[inv_u]
            u_dec = decode_sequence(model, _permute_uav(batch, u_perm), env_u, teacher_tokens=None)["assignment"]
            expected_u = base[inv_u]
            uav_hits += int(np.array_equal(u_dec, expected_u))
            inv_b = np.argsort(b_perm)
            env_b = copy.deepcopy(env)
            env_b.buoy_positions = np.asarray(env.buoy_positions, dtype=np.float64)[inv_b]
            b_dec = decode_sequence(model, _permute_buoy(batch, b_perm), env_b, teacher_tokens=None)["assignment"]
            expected_b = np.array([-1 if int(v) < 0 else int(b_perm[int(v)]) for v in base], dtype=np.int64)
            buoy_hits += int(np.array_equal(b_dec, expected_b))
    return {
        "encoder": model.encoder_name,
        "uav_pass": uav_hits == int(n_perm),
        "buoy_pass": buoy_hits == int(n_perm),
        "uav_hits": int(uav_hits),
        "buoy_hits": int(buoy_hits),
        "n_perm": int(n_perm),
    }


__all__ = ["audit_model"]
