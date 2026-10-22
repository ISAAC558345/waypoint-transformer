"""Set-valued partial assignment and SINR context. Independent of selection order."""
from __future__ import annotations

import numpy as np

from wt.htw_phase9.const import IDLE_INDEX, MAX_BUOYS, MAX_UAVS
from wt.htw_phase94.assignment_context import PHYS_DIM, PositionSnap, make_snap as _make_snap_positions, physics_vector
from wt.l3b.env import cluster_id, kappa_for, path_gain, reference_rate, sinr_rates_for_links

UNDECIDED = -2


def default_buoy_cluster(num_buoys: int = MAX_BUOYS) -> np.ndarray:
    return np.asarray([cluster_id(index) for index in range(int(num_buoys))], dtype=np.int64)


def make_snap(uav_positions: np.ndarray, buoy_positions: np.ndarray, config=None, buoy_cluster: np.ndarray | None = None, env=None):
    """Position snapshot plus cluster ids that travel with buoy slots under permutation."""
    snap = _make_snap_positions(uav_positions, buoy_positions, config)
    if buoy_cluster is not None:
        snap.buoy_cluster = np.asarray(buoy_cluster, dtype=np.int64)
    elif env is not None and getattr(env, "buoy_cluster", None) is not None:
        snap.buoy_cluster = np.asarray(env.buoy_cluster, dtype=np.int64)
    else:
        snap.buoy_cluster = default_buoy_cluster(int(np.asarray(buoy_positions).shape[0]))
    return snap


def assignment_links(assignment: np.ndarray) -> list[tuple[int, int]]:
    links = []
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if 0 <= int(buoy) < IDLE_INDEX:
            links.append((int(uav), int(buoy)))
    links.sort()
    return links


def decided_mask(assignment: np.ndarray) -> np.ndarray:
    values = np.asarray(assignment, dtype=np.int64)
    return values != UNDECIDED


def occupied_buoys(assignment: np.ndarray, num_buoys: int = MAX_BUOYS) -> np.ndarray:
    occupied = np.zeros(int(num_buoys), dtype=bool)
    for buoy in np.asarray(assignment, dtype=np.int64):
        if 0 <= int(buoy) < int(num_buoys):
            occupied[int(buoy)] = True
    return occupied


def legal_candidate_mask(assignment: np.ndarray, buoy_available: np.ndarray | None = None) -> np.ndarray:
    """True where (m, k) is a legal next candidate. k=IDLE always legal for undecided UAVs."""
    mask = np.zeros((MAX_UAVS, MAX_BUOYS + 1), dtype=bool)
    undecided = ~decided_mask(assignment)
    occupied = occupied_buoys(assignment)
    avail = np.ones(MAX_BUOYS, dtype=bool) if buoy_available is None else np.asarray(buoy_available, dtype=bool)
    for uav in range(MAX_UAVS):
        if not bool(undecided[uav]):
            continue
        mask[uav, :MAX_BUOYS] = (~occupied) & avail
        mask[uav, IDLE_INDEX] = True
    return mask


def empty_assignment(num_uavs: int = MAX_UAVS) -> np.ndarray:
    return np.full(int(num_uavs), UNDECIDED, dtype=np.int64)


def apply_choice(assignment: np.ndarray, uav: int, token: int) -> np.ndarray:
    out = np.asarray(assignment, dtype=np.int64).copy()
    out[int(uav)] = -1 if int(token) >= IDLE_INDEX else int(token)
    return out


def _gains(snap: PositionSnap) -> np.ndarray:
    cfg = snap.config
    g = np.zeros((MAX_UAVS, MAX_BUOYS), dtype=np.float64)
    for uav in range(min(MAX_UAVS, snap.uav_positions.shape[0])):
        for buoy in range(min(MAX_BUOYS, snap.buoy_positions.shape[0])):
            g[uav, buoy] = path_gain(cfg, snap.uav_positions[uav], snap.buoy_positions[buoy])
    return g


def physics_from_gains(snap: PositionSnap, links: list[tuple[int, int]], uav: int, gains: np.ndarray | None = None) -> np.ndarray:
    cfg = snap.config
    if gains is None:
        return physics_vector(snap, links, uav)
    ref = max(float(reference_rate(cfg)), 1.0e-9)
    tx = float(cfg.tx_power)
    noise = float(cfg.noise_power)
    n_uav = int(cfg.num_uavs)
    sinr = np.zeros(n_uav, dtype=np.float64)
    rate = np.zeros(n_uav, dtype=np.float64)
    for i, bi in links:
        signal = gains[i, bi] * tx
        interf = noise
        for j, bj in links:
            if int(j) == int(i):
                continue
            interf += gains[j, bi] * tx * _kappa_pair(snap, bj, bi)
        sinr[i] = signal / max(interf, 1.0e-18)
        rate[i] = float(cfg.bandwidth_hz) * np.log2(1.0 + sinr[i])
    active = [index for index, _ in links]
    mean_rate = float(np.mean(rate[active])) if active else 0.0
    own_rate = float(rate[uav]) if any(index == int(uav) for index, _ in links) else 0.0
    own_sinr = float(sinr[uav]) if any(index == int(uav) for index, _ in links) else 0.0
    return np.asarray(
        (
            np.clip(own_rate / ref, 0.0, 2.0) / 2.0,
            np.clip(np.log1p(own_sinr) / 10.0, 0.0, 1.0),
            np.clip(mean_rate / ref, 0.0, 2.0) / 2.0,
            float(len(links)) / float(MAX_UAVS),
        ),
        dtype=np.float32,
    )


def candidate_physics_set(snap: PositionSnap, assignment: np.ndarray, uav: int, token: int, gains: np.ndarray | None = None) -> np.ndarray:
    links = assignment_links(assignment)
    if int(token) < IDLE_INDEX:
        links = list(links) + [(int(uav), int(token))]
        links.sort()
    return physics_from_gains(snap, links, uav, gains)


def _kappa_pair(snap: PositionSnap, buoy_a: int, buoy_b: int) -> float:
    clusters = getattr(snap, "buoy_cluster", None)
    if clusters is None:
        return kappa_for(snap.config, buoy_a, buoy_b)
    same = int(clusters[int(buoy_a)]) == int(clusters[int(buoy_b)])
    return float(snap.config.kappa_same_cluster if same else snap.config.kappa_cross_cluster)


def _kappa_matrix(snap: PositionSnap) -> np.ndarray:
    cfg = snap.config
    clusters = np.asarray(getattr(snap, "buoy_cluster", default_buoy_cluster()), dtype=np.int64)
    same = clusters[:, None] == clusters[None, :]
    mat = np.where(same, float(cfg.kappa_same_cluster), float(cfg.kappa_cross_cluster)).astype(np.float64)
    return mat


def all_candidate_physics_set(snap: PositionSnap, assignment: np.ndarray) -> np.ndarray:
    cfg = snap.config
    out = np.zeros((MAX_UAVS, MAX_BUOYS + 1, PHYS_DIM), dtype=np.float32)
    gains = getattr(snap, "_gains", None)
    if gains is None:
        gains = _gains(snap)
        snap._gains = gains
    kappa = getattr(snap, "_kappa", None)
    if kappa is None:
        kappa = _kappa_matrix(snap)
        snap._kappa = kappa
    legal = legal_candidate_mask(assignment)
    links = assignment_links(assignment)
    tx = float(cfg.tx_power)
    noise = float(cfg.noise_power)
    bandwidth = float(cfg.bandwidth_hz)
    ref = max(float(reference_rate(cfg)), 1.0e-9)
    n_link = len(links)
    if n_link:
        u_idx = np.asarray([u for u, _ in links], dtype=np.int64)
        b_idx = np.asarray([b for _, b in links], dtype=np.int64)
        signal = gains[u_idx, b_idx] * tx
        interf = np.full(n_link, noise, dtype=np.float64)
        for i in range(n_link):
            for j in range(n_link):
                if i == j:
                    continue
                interf[i] += gains[u_idx[j], b_idx[i]] * tx * kappa[b_idx[j], b_idx[i]]
        sinr = signal / np.maximum(interf, 1.0e-18)
        rate = bandwidth * np.log2(1.0 + sinr)
        mean_rate = float(np.mean(rate))
    else:
        u_idx = np.zeros(0, dtype=np.int64)
        b_idx = np.zeros(0, dtype=np.int64)
        signal = np.zeros(0, dtype=np.float64)
        interf = np.zeros(0, dtype=np.float64)
        rate = np.zeros(0, dtype=np.float64)
        mean_rate = 0.0
    idle_vec = np.asarray(
        (0.0, 0.0, np.clip(mean_rate / ref, 0.0, 2.0) / 2.0, float(n_link) / float(MAX_UAVS)),
        dtype=np.float32,
    )
    for uav in range(MAX_UAVS):
        if bool(legal[uav, IDLE_INDEX]):
            out[uav, IDLE_INDEX] = idle_vec
    extra = np.zeros((MAX_UAVS, MAX_BUOYS, max(n_link, 1)), dtype=np.float64)
    if n_link:
        # extra[m,k,i] interference from new UAV m@k onto existing link i
        extra = gains[:, b_idx][:, None, :] * tx * kappa[None, :, b_idx]
    interf_on_new = np.full(MAX_BUOYS, noise, dtype=np.float64)
    if n_link:
        interf_on_new = noise + np.sum(gains[u_idx][:, :] * tx * kappa[b_idx][:, :], axis=0)
    new_signal = gains * tx
    new_sinr = new_signal / np.maximum(interf_on_new[None, :], 1.0e-18)
    new_rate = bandwidth * np.log2(1.0 + new_sinr)
    if n_link:
        new_interf_old = interf[None, None, :] + extra
        old_sinr = signal[None, None, :] / np.maximum(new_interf_old, 1.0e-18)
        old_rate = bandwidth * np.log2(1.0 + old_sinr)
        mean_new = (old_rate.sum(axis=2) + new_rate) / float(n_link + 1)
    else:
        mean_new = new_rate
    own_rate = new_rate
    own_sinr = new_sinr
    phys = np.stack(
        (
            np.clip(own_rate / ref, 0.0, 2.0) / 2.0,
            np.clip(np.log1p(own_sinr) / 10.0, 0.0, 1.0),
            np.clip(mean_new / ref, 0.0, 2.0) / 2.0,
            np.full((MAX_UAVS, MAX_BUOYS), float(n_link + 1) / float(MAX_UAVS)),
        ),
        axis=-1,
    ).astype(np.float32)
    for uav in range(MAX_UAVS):
        for token in range(MAX_BUOYS):
            if bool(legal[uav, token]):
                out[uav, token] = phys[uav, token]
    return out


def canonical_keys(snap: PositionSnap, assignment: np.ndarray) -> np.ndarray:
    """Physical lexicographic keys (M, K+1, 6): UAV xy, buoy xy or +inf idle, idle flag, -rate."""
    keys = np.zeros((MAX_UAVS, MAX_BUOYS + 1, 6), dtype=np.float64)
    uav_pos = np.asarray(snap.uav_positions, dtype=np.float64)
    buoy_pos = np.asarray(snap.buoy_positions, dtype=np.float64)
    for uav in range(min(MAX_UAVS, uav_pos.shape[0])):
        keys[uav, :, 0] = uav_pos[uav, 0]
        keys[uav, :, 1] = uav_pos[uav, 1]
        for buoy in range(min(MAX_BUOYS, buoy_pos.shape[0])):
            keys[uav, buoy, 2] = buoy_pos[buoy, 0]
            keys[uav, buoy, 3] = buoy_pos[buoy, 1]
            keys[uav, buoy, 4] = 0.0
        keys[uav, IDLE_INDEX, 2] = 1.0e9
        keys[uav, IDLE_INDEX, 3] = 1.0e9
        keys[uav, IDLE_INDEX, 4] = 1.0
    phys = all_candidate_physics_set(snap, assignment)
    keys[:, :, 5] = -phys[:, :, 0]
    return keys


def assignment_set_tuple(assignment: np.ndarray) -> tuple[tuple[int, int], ...]:
    pairs = []
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        token = IDLE_INDEX if int(buoy) < 0 else int(buoy)
        pairs.append((int(uav), token))
    return tuple(sorted(pairs))


__all__ = [
    "UNDECIDED",
    "all_candidate_physics_set",
    "apply_choice",
    "assignment_links",
    "assignment_set_tuple",
    "candidate_physics_set",
    "canonical_keys",
    "default_buoy_cluster",
    "empty_assignment",
    "legal_candidate_mask",
    "make_snap",
    "occupied_buoys",
    "PositionSnap",
    "PHYS_DIM",
]
