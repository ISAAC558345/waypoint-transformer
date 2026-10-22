"""Shared LL-V2 contracts. Low-level controls movement only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wt.htw_phase100.contract import TaskOption, buoy_or_idle

from .protocol import DECODER_IDLE, ENV_IDLE, HOLD_STEPS


def refuse_rematch(option: TaskOption) -> None:
    if not np.array_equal(np.asarray(option.association), np.asarray(option.targets)):
        raise RuntimeError("low-level refused rematching: association != targets")


def env_target(token: int, num_buoys: int) -> int:
    """Map association token to env buoy index or ENV_IDLE. Never uses buoy[-1]."""
    value = int(token)
    if value < 0 or value >= int(num_buoys) or value >= int(DECODER_IDLE):
        return int(ENV_IDLE)
    return int(buoy_or_idle(value))


def clip_unit_step(movement: np.ndarray) -> np.ndarray:
    move = np.asarray(movement, dtype=np.float64)
    if move.ndim != 2 or move.shape[-1] != 2:
        raise ValueError("movement must have shape (M, 2)")
    norms = np.linalg.norm(move, axis=-1, keepdims=True)
    return np.where(norms > 1.0, move / np.clip(norms, 1.0e-9, None), move)


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.float64)
    return np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)


def minimum_pairwise_distance(positions: np.ndarray) -> float:
    dist = pairwise_distances(positions)
    m_count = int(dist.shape[0])
    if m_count < 2:
        return float("inf")
    mask = np.triu(np.ones((m_count, m_count), dtype=bool), k=1)
    return float(np.min(dist[mask]))


@dataclass
class CollisionConfig:
    uav_physical_radius: float
    collision_margin: float
    placeholder: bool = True

    @property
    def d_safe(self) -> float:
        return float(2.0 * self.uav_physical_radius + self.collision_margin)


@dataclass
class ResidualConfig:
    alpha_residual: float
    placeholder: bool = True


__all__ = [
    "CollisionConfig",
    "HOLD_STEPS",
    "ResidualConfig",
    "TaskOption",
    "clip_unit_step",
    "env_target",
    "minimum_pairwise_distance",
    "pairwise_distances",
    "refuse_rematch",
]
