"""Validation metrics for one decision state.

Training still uses exact J15 order. Top-k uses the stored 1-bit utility_rank
groups. Those groups are not model inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr

from wt.benchmark_v2.phase6id_joint_training.protocol import COLLAPSE_EPS


def is_collapsed(scores: np.ndarray, eps: float = COLLAPSE_EPS) -> bool:
    array = np.asarray(scores, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return True
    return float(np.std(array)) < float(eps) or float(np.max(array) - np.min(array)) < float(eps)


def rank_correlation(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    left = np.asarray(scores, dtype=np.float64)
    right = np.asarray(labels, dtype=np.float64)
    if left.shape != right.shape or left.size < 3 or is_collapsed(left) or float(np.std(right)) <= 1.0e-12:
        return float("nan"), float("nan")
    spearman = spearmanr(left, right)
    kendall = kendalltau(left, right)
    spearman_value = getattr(spearman, "statistic", spearman.correlation)
    kendall_value = getattr(kendall, "statistic", kendall.correlation)
    return float(spearman_value), float(kendall_value)


def state_metrics(scores: np.ndarray, labels: np.ndarray, utility_rank: np.ndarray) -> dict[str, float]:
    """Score row i belongs to assignment row i and j15_bits row i."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    ranks = np.asarray(utility_rank, dtype=np.int64).reshape(-1)
    if values.shape != truth.shape or values.shape != ranks.shape:
        raise ValueError("scores, j15_bits, and utility_rank are not aligned")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(truth)):
        raise ValueError("non-finite score or J15 label")
    order = np.argsort(-values, kind="stable")
    chosen = int(order[0])
    best_payload = float(np.max(truth))
    chosen_payload = float(truth[chosen])
    regret = float(best_payload - chosen_payload)
    optimal = ranks == 1
    if not np.any(optimal):
        raise ValueError("utility_rank has no best group")
    spearman, kendall = rank_correlation(values, truth)
    return {
        "spearman": spearman,
        "kendall": kendall,
        "top1": float(bool(optimal[chosen])),
        "top3": float(np.any(optimal[order[: min(3, order.size)]])),
        "top5": float(np.any(optimal[order[: min(5, order.size)]])),
        "payload": chosen_payload,
        "oracle_payload": best_payload,
        "regret": regret,
        "normalized_regret": float(regret / (best_payload + 1.0)),
        "collapsed": float(is_collapsed(values)),
        "score_std": float(np.std(values)),
        "chosen": float(chosen),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("no validation rows")
    packed = {"n_states": float(len(rows))}
    for key in ("spearman", "kendall", "top1", "top3", "top5", "payload", "oracle_payload", "regret", "normalized_regret", "collapsed", "score_std"):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        packed[key] = float(np.mean(finite)) if finite.size else float("nan")
        packed[f"{key}_n"] = float(finite.size)
    return packed


__all__ = ["is_collapsed", "mean_metrics", "rank_correlation", "state_metrics"]
