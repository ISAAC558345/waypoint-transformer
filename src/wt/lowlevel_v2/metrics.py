"""Diagnostic metrics. Evaluation uses these, not train reward."""
from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import CollisionConfig, minimum_pairwise_distance, pairwise_distances
from .protocol import D_SAFE_PLACEHOLDER_M, NEAR_COLLISION_FACTOR


def collision_diagnostics(positions: np.ndarray, collision: CollisionConfig) -> dict[str, Any]:
    dist = pairwise_distances(positions)
    m_count = int(dist.shape[0])
    d_safe = float(collision.d_safe)
    near_thr = float(NEAR_COLLISION_FACTOR) * d_safe
    collisions = 0
    nears = 0
    mins = []
    for i in range(m_count):
        for j in range(i + 1, m_count):
            value = float(dist[i, j])
            mins.append(value)
            if value < d_safe - 1.0e-9:
                collisions += 1
            elif value < near_thr:
                nears += 1
    return {
        "minimum_pairwise_distance": float(min(mins) if mins else float("inf")),
        "collision_count": int(collisions),
        "near_collision_count": int(nears),
        "d_safe": d_safe,
        "d_safe_is_placeholder": bool(collision.placeholder or abs(d_safe - D_SAFE_PLACEHOLDER_M) < 1.0e-12),
    }


def near_collision_cost(positions: np.ndarray, collision: CollisionConfig) -> float:
    d_safe = float(collision.d_safe)
    dist = pairwise_distances(positions)
    cost = 0.0
    m_count = int(dist.shape[0])
    for i in range(m_count):
        for j in range(i + 1, m_count):
            gap = d_safe - float(dist[i, j])
            if gap > 0.0:
                cost += float(gap * gap)
    return float(cost)


__all__ = ["collision_diagnostics", "minimum_pairwise_distance", "near_collision_cost"]
