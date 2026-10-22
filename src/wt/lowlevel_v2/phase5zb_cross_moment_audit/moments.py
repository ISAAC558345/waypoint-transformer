"""Diagnostic second-order moments. No parameters are fit."""
from __future__ import annotations

import numpy as np

from wt.lowlevel_v2.phase5zb_cross_moment_audit.protocol import EPS


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / (norms + EPS)


def _moment(left: np.ndarray, right: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Mean of outer products over active UAVs. left/right are [N, M, D]."""
    mask = active.astype(np.float64)[..., None]
    count = np.maximum(active.sum(axis=1).astype(np.float64), 1.0)[:, None, None]
    return np.einsum("nmi,nmj->nij", left * mask, right, dtype=np.float64) / count


def cross_moment(h_u: np.ndarray, h_b: np.ndarray, assignments: np.ndarray) -> dict[str, np.ndarray]:
    """R_CM = vec(C_UB), idle UAV mean, idle fraction. No learned map."""
    chosen = np.asarray(assignments, dtype=np.int64)
    buoy = np.clip(chosen, 0, h_b.shape[0] - 1)
    active = chosen >= 0
    u_hat = _unit(np.asarray(h_u, dtype=np.float64))
    b_hat = _unit(np.asarray(h_b, dtype=np.float64))
    u_token = np.broadcast_to(u_hat, (chosen.shape[0],) + u_hat.shape).copy()
    b_token = b_hat[buoy]
    moment = _moment(u_token, b_token, active)
    idle = ~active
    m_idle = (idle.astype(np.float64)[..., None] * u_token).sum(axis=1) / float(chosen.shape[1])
    idle_fraction = idle.mean(axis=1).astype(np.float64)
    flat = np.concatenate((moment.reshape(chosen.shape[0], -1), m_idle, idle_fraction[:, None]), axis=1)
    return {"representation": flat, "moment": moment, "m_idle": m_idle, "idle_fraction": idle_fraction}


def edge_mean(h_edge: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    chosen = np.asarray(assignments, dtype=np.int64)
    buoy = np.clip(chosen, 0, h_edge.shape[1] - 1)
    active = chosen >= 0
    rows = np.arange(chosen.shape[1])[None, :]
    selected = h_edge[rows, buoy]
    mask = active.astype(np.float64)[..., None]
    count = np.maximum(active.sum(axis=1).astype(np.float64), 1.0)[:, None]
    return (selected * mask).sum(axis=1) / count


def pair_mean(h_u: np.ndarray, h_b: np.ndarray, h_edge: np.ndarray, assignments: np.ndarray, relational: np.ndarray) -> np.ndarray:
    """First-order control: mean of the already-bound token, including idle zeros."""
    chosen = np.asarray(assignments, dtype=np.int64)
    buoy = np.clip(chosen, 0, h_b.shape[0] - 1)
    active = chosen >= 0
    uav = np.broadcast_to(h_u, (chosen.shape[0],) + h_u.shape).copy()
    target = h_b[buoy] * active.astype(np.float64)[..., None]
    rows = np.arange(chosen.shape[1])[None, :]
    edge = h_edge[rows, buoy] * active.astype(np.float64)[..., None]
    context = np.broadcast_to(relational[:, None, :], (chosen.shape[0], chosen.shape[1], relational.shape[1])).copy()
    tokens = np.concatenate((uav, target, edge, context), axis=-1)
    return tokens.mean(axis=1)


def optional_terms(h_u: np.ndarray, h_b: np.ndarray, h_edge: np.ndarray, assignments: np.ndarray) -> dict[str, np.ndarray]:
    """Extra outer products. Used only after CM-UB misses the structural bar."""
    chosen = np.asarray(assignments, dtype=np.int64)
    buoy = np.where(chosen >= 0, chosen, 0)
    active = chosen >= 0
    u_hat = _unit(np.asarray(h_u, dtype=np.float64))
    b_hat = _unit(np.asarray(h_b, dtype=np.float64))
    rows = np.arange(chosen.shape[1])[None, :]
    edge = h_edge[rows, np.clip(buoy, 0, h_edge.shape[1] - 1)]
    e_hat = _unit(edge)
    u_token = np.broadcast_to(u_hat, (chosen.shape[0],) + u_hat.shape).copy()
    b_token = b_hat[np.clip(buoy, 0, h_b.shape[0] - 1)]
    base = cross_moment(h_u, h_b, assignments)
    ue = _moment(u_token, e_hat, active)
    be = _moment(b_token, e_hat, active)
    tail = np.concatenate((base["m_idle"], base["idle_fraction"][:, None]), axis=1)
    ub = base["moment"].reshape(chosen.shape[0], -1)
    ue_flat = ue.reshape(chosen.shape[0], -1)
    be_flat = be.reshape(chosen.shape[0], -1)
    return {
        "cm_ub": base["representation"],
        "cm_ub_ue": np.concatenate((ub, ue_flat, tail), axis=1),
        "cm_ub_be": np.concatenate((ub, be_flat, tail), axis=1),
        "cm_all": np.concatenate((ub, ue_flat, be_flat, tail), axis=1),
    }


def permute_uavs(h_u: np.ndarray, h_edge: np.ndarray, assignments: np.ndarray, order: np.ndarray):
    perm = np.asarray(order, dtype=np.int64)
    return h_u[perm], h_edge[perm], np.asarray(assignments, dtype=np.int64)[:, perm]


def permute_buoys(h_b: np.ndarray, h_edge: np.ndarray, assignments: np.ndarray, order: np.ndarray):
    perm = np.asarray(order, dtype=np.int64)
    inverse = np.empty(perm.shape[0], dtype=np.int64)
    inverse[perm] = np.arange(perm.shape[0])
    relabeled = np.asarray(assignments, dtype=np.int64).copy()
    active = relabeled >= 0
    relabeled[active] = inverse[relabeled[active]]
    return h_b[perm], h_edge[:, perm], relabeled


__all__ = [
    "cross_moment",
    "edge_mean",
    "optional_terms",
    "pair_mean",
    "permute_buoys",
    "permute_uavs",
]
