"""Partial-assignment legality masks and SINR context. Does not edit L3-B."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from wt.htw_phase9.const import IDLE_INDEX, MAX_BUOYS, MAX_UAVS
from wt.l3b.config import L3BConfig
from wt.l3b.env import reference_rate, sinr_rates_for_links

PHYS_DIM = 4


class PositionSnap(SimpleNamespace):
    pass


def make_snap(uav_positions: np.ndarray, buoy_positions: np.ndarray, config: L3BConfig | None = None) -> PositionSnap:
    cfg = config or L3BConfig()
    return PositionSnap(uav_positions=np.asarray(uav_positions, dtype=np.float64), buoy_positions=np.asarray(buoy_positions, dtype=np.float64), config=cfg)


def occupied_buoys(partial: np.ndarray, upto: int, num_buoys: int) -> np.ndarray:
    occupied = np.zeros(int(num_buoys), dtype=bool)
    for uav in range(int(upto)):
        buoy = int(partial[uav])
        if 0 <= buoy < int(num_buoys):
            occupied[buoy] = True
    return occupied


def legality_mask(partial: np.ndarray, uav_index: int, num_buoys: int = MAX_BUOYS, buoy_available: np.ndarray | None = None) -> np.ndarray:
    """True = legal. Idle always legal. Capacity-1 buoys already taken are illegal."""
    mask = np.ones(int(num_buoys) + 1, dtype=bool)
    occupied = occupied_buoys(partial, uav_index, num_buoys)
    mask[: int(num_buoys)] = ~occupied
    if buoy_available is not None:
        avail = np.asarray(buoy_available, dtype=bool)
        mask[: int(num_buoys)] &= avail
    mask[IDLE_INDEX] = True
    return mask


def prefix_links(partial: np.ndarray, upto: int) -> list[tuple[int, int]]:
    links: list[tuple[int, int]] = []
    for uav in range(int(upto)):
        buoy = int(partial[uav])
        if buoy >= 0 and buoy < IDLE_INDEX:
            links.append((uav, buoy))
    return links


def physics_vector(snap: PositionSnap, links: list[tuple[int, int]], uav_index: int) -> np.ndarray:
    ref = max(float(reference_rate(snap.config)), 1.0e-9)
    sinr, rate = sinr_rates_for_links(snap, links)
    active = [index for index, _buoy in links]
    mean_rate = float(np.mean(rate[active])) if active else 0.0
    own_rate = float(rate[uav_index]) if any(index == int(uav_index) for index, _ in links) else 0.0
    own_sinr = float(sinr[uav_index]) if any(index == int(uav_index) for index, _ in links) else 0.0
    return np.asarray(
        (
            np.clip(own_rate / ref, 0.0, 2.0) / 2.0,
            np.clip(np.log1p(own_sinr) / 10.0, 0.0, 1.0),
            np.clip(mean_rate / ref, 0.0, 2.0) / 2.0,
            float(len(links)) / float(MAX_UAVS),
        ),
        dtype=np.float32,
    )


def candidate_physics(snap: PositionSnap, partial: np.ndarray, uav_index: int, token: int) -> np.ndarray:
    links = prefix_links(partial, uav_index)
    if int(token) < IDLE_INDEX:
        links = list(links) + [(int(uav_index), int(token))]
    return physics_vector(snap, links, uav_index)


def all_candidate_physics(snap: PositionSnap, partial: np.ndarray, uav_index: int, num_buoys: int = MAX_BUOYS) -> np.ndarray:
    out = np.zeros((int(num_buoys) + 1, PHYS_DIM), dtype=np.float32)
    for token in range(int(num_buoys) + 1):
        out[token] = candidate_physics(snap, partial, uav_index, token)
    return out


def partial_vector(partial: np.ndarray, uav_index: int, num_uavs: int = MAX_UAVS, num_buoys: int = MAX_BUOYS) -> np.ndarray:
    assigned = np.zeros(int(num_uavs), dtype=np.float32)
    occupied = occupied_buoys(partial, uav_index, num_buoys).astype(np.float32)
    assigned[:uav_index] = (np.asarray(partial[:uav_index], dtype=np.int64) >= 0).astype(np.float32)
    n_active = float(np.sum(assigned)) / float(num_uavs)
    mean_choice = 0.0
    if uav_index > 0:
        mean_choice = float(np.mean(np.clip(np.asarray(partial[:uav_index], dtype=np.float32), -1.0, float(num_buoys)))) / float(num_buoys)
    return np.concatenate((assigned, occupied, np.asarray((n_active, mean_choice), dtype=np.float32)))


PARTIAL_DIM = MAX_UAVS + MAX_BUOYS + 2


__all__ = [
    "PARTIAL_DIM",
    "PHYS_DIM",
    "PositionSnap",
    "all_candidate_physics",
    "candidate_physics",
    "legality_mask",
    "make_snap",
    "occupied_buoys",
    "partial_vector",
    "prefix_links",
]
