"""Rank-mean consensus. Selection does not read J15, scenario ids, or a preferred seed."""
from __future__ import annotations

import numpy as np


def competition_ranks(values: np.ndarray) -> np.ndarray:
    """Rank 1 is the best value. Equal values share the minimum rank."""
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    better = np.sum(scores[None, :] > scores[:, None], axis=1)
    return better.astype(np.int64) + 1


def canonical_key(assignment: np.ndarray) -> tuple[int, ...]:
    return tuple(int(item) for item in np.asarray(assignment, dtype=np.int64).reshape(-1))


def rank_mean_select(values: np.ndarray, assignments: np.ndarray) -> dict:
    """Choose the assignment with the lowest mean competition rank.

    Ties use the larger Top-1 majority, then the lexicographic assignment.
    """
    scores = np.asarray(values, dtype=np.float64)
    chosen_rows = np.asarray(assignments, dtype=np.int64)
    if scores.ndim != 2 or chosen_rows.shape[0] != scores.shape[1]:
        raise ValueError("values must be [seeds, candidates] and align with assignments")
    if not np.all(np.isfinite(scores)):
        raise ValueError("non-finite values")
    ranks = np.stack([competition_ranks(scores[seed]) for seed in range(scores.shape[0])], axis=0)
    rank_sum = np.sum(ranks, axis=0).astype(np.int64)
    majority = np.sum(ranks == 1, axis=0).astype(np.int64)
    keys = [canonical_key(row) for row in chosen_rows]
    best_sum = int(np.min(rank_sum))
    pool = [index for index in range(scores.shape[1]) if int(rank_sum[index]) == best_sum]
    stage = "mean_rank"
    if len(pool) > 1:
        best_majority = max(int(majority[index]) for index in pool)
        pool = [index for index in pool if int(majority[index]) == best_majority]
        stage = "majority_top1"
    if len(pool) > 1:
        stage = "canonical_order"
    selected = min(pool, key=lambda index: keys[index])
    return {
        "index": int(selected),
        "assignment": np.asarray(chosen_rows[selected], dtype=np.int64).copy(),
        "key": keys[selected],
        "ranks": np.asarray(ranks[:, selected], dtype=np.int64),
        "rank_matrix": ranks,
        "rank_sum": rank_sum,
        "majority": majority,
        "mean_rank": float(rank_sum[selected] / scores.shape[0]),
        "majority_top1": int(majority[selected]),
        "stage": stage,
        "strength": float(np.max(majority) / scores.shape[0]),
    }


__all__ = ["canonical_key", "competition_ranks", "rank_mean_select"]
