"""Square Hungarian (minimize). No scipy dependency."""
from __future__ import annotations

import numpy as np


def _hungarian_square(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cost = np.asarray(cost, dtype=np.float64)
    n_count = int(cost.shape[0])
    u_dual = np.zeros(n_count + 1, dtype=np.float64)
    v_dual = np.zeros(n_count + 1, dtype=np.float64)
    p_match = np.zeros(n_count + 1, dtype=np.int64)
    way = np.zeros(n_count + 1, dtype=np.int64)
    for i in range(1, n_count + 1):
        p_match[0] = i
        j0 = 0
        minv = np.full(n_count + 1, np.inf)
        used = np.zeros(n_count + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p_match[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, n_count + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u_dual[i0] - v_dual[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = float(minv[j])
                    j1 = j
            for j in range(n_count + 1):
                if used[j]:
                    u_dual[int(p_match[j])] += delta
                    v_dual[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p_match[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p_match[j0] = p_match[j1]
            j0 = j1
            if j0 == 0:
                break
    col_for_row = np.full(n_count, -1, dtype=np.int64)
    for j in range(1, n_count + 1):
        if p_match[j] != 0:
            col_for_row[int(p_match[j]) - 1] = j - 1
    rows = np.arange(n_count, dtype=np.int64)
    return rows, col_for_row


def hungarian_maximize(scores: np.ndarray) -> np.ndarray:
    """Maximize row-to-col assignment. Returns column index per row; -1 if dummy idle col."""
    scores = np.asarray(scores, dtype=np.float64)
    n_row, n_col = int(scores.shape[0]), int(scores.shape[1])
    n_count = max(n_row, n_col)
    padded = np.zeros((n_count, n_count), dtype=np.float64)
    padded[:n_row, :n_col] = scores
    cost = -padded
    _rows, cols = _hungarian_square(cost)
    return cols[:n_row]


def match_uav_buoy(buoy_scores: np.ndarray, idle_scores: np.ndarray, buoy_mask: np.ndarray, uav_mask: np.ndarray) -> np.ndarray:
    """Exclusive matching. Buoy capacity 1. Idle allowed via private dummy columns."""
    m_count = int(buoy_scores.shape[0])
    k_count = int(buoy_scores.shape[1])
    pad = np.full((m_count, k_count + m_count), -1.0e9, dtype=np.float64)
    valid = np.asarray(buoy_mask, dtype=bool)
    pad[:, :k_count] = np.asarray(buoy_scores, dtype=np.float64)
    pad[:, :k_count][:, ~valid] = -1.0e9
    idle = np.asarray(idle_scores, dtype=np.float64)
    for uav in range(m_count):
        if not bool(uav_mask[uav]):
            pad[uav, :] = -1.0e9
            pad[uav, k_count + uav] = 0.0
            continue
        pad[uav, k_count + uav] = float(idle[uav])
    cols = hungarian_maximize(pad)
    assignment = np.full(m_count, -1, dtype=np.int64)
    for uav, col in enumerate(cols):
        if int(col) < k_count and bool(uav_mask[uav]) and bool(valid[int(col)]):
            assignment[uav] = int(col)
    return assignment


__all__ = ["hungarian_maximize", "match_uav_buoy"]
