"""Distance, correlation, and ranking. No probe is fit."""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr

from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import assignment_metrics, quantize

from .protocol import BLOCK_GAPS, COSINE_LIMIT, PAYLOAD_BLOCK_BITS, REGRET_BAR, RESIDUAL_BAR, SPEARMAN_BAR, SPEARMAN_GAIN, TOP3_BAR


def _finite(value: float) -> float:
    number = float(value)
    return number if np.isfinite(number) else float("nan")


def correlation(left, right) -> tuple[float, float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3 or float(np.std(x[mask])) < 1.0e-12 or float(np.std(y[mask])) < 1.0e-12:
        return float("nan"), float("nan")
    spearman = spearmanr(x[mask], y[mask])
    kendall = kendalltau(x[mask], y[mask])
    spearman_value = getattr(spearman, "statistic", spearman.correlation)
    kendall_value = getattr(kendall, "statistic", kendall.correlation)
    return _finite(spearman_value), _finite(kendall_value)


def sign_accuracy(scores, labels) -> float:
    score = np.asarray(scores, dtype=np.float64)
    label = np.asarray(labels, dtype=np.float64)
    gap_score = score[:, None] - score[None, :]
    gap_label = label[:, None] - label[None, :]
    mask = np.triu(np.ones(score.shape[0], dtype=bool), k=1)
    mask &= np.abs(gap_label) >= float(PAYLOAD_BLOCK_BITS) - 1.0
    mask &= np.abs(gap_score) > 1.0e-12
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(gap_score[mask]) == np.sign(gap_label[mask])))


def ranking(scores, labels) -> dict[str, float]:
    measured = assignment_metrics(scores, labels)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    levels = quantize(labels)
    best = float(np.max(levels))
    width = min(5, int(order.size))
    measured["top5"] = float(np.any(levels[order[:width]] == best))
    measured["sign_accuracy"] = sign_accuracy(scores, labels)
    return measured


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, 1.0e-8)


def nearest_neighbor(vectors: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Fraction of assignments whose nearest other assignment is close and far in J15."""
    array = np.nan_to_num(np.asarray(vectors, dtype=np.float64), nan=0.0)
    if array.ndim == 1:
        array = array[:, None]
    if int(array.shape[1]) == 1:
        values = array[:, 0]
        distance = np.abs(values[:, None] - values[None, :])
        scale = 0.5 * (np.abs(values[:, None]) + np.abs(values[None, :])) + 1.0e-8
        normalized = distance / scale
        np.fill_diagonal(normalized, np.inf)
        neighbor = np.argmin(normalized, axis=1)
        cosine = 1.0 - normalized[np.arange(normalized.shape[0]), neighbor]
        close = normalized[np.arange(normalized.shape[0]), neighbor] <= (1.0 - float(COSINE_LIMIT))
    else:
        units = unit_rows(array)
        gram = units @ units.T
        np.fill_diagonal(gram, -np.inf)
        neighbor = np.argmax(gram, axis=1)
        cosine = gram[np.arange(gram.shape[0]), neighbor]
        close = cosine >= float(COSINE_LIMIT)
    gap = np.abs(np.asarray(labels, dtype=np.float64) - np.asarray(labels, dtype=np.float64)[neighbor])
    result = {
        "neighbor_cosine_median": float(np.median(cosine[np.isfinite(cosine)])) if np.any(np.isfinite(cosine)) else float("nan"),
    }
    for blocks in BLOCK_GAPS:
        result[f"contradiction_{blocks}"] = float(np.mean(close & (gap >= blocks * float(PAYLOAD_BLOCK_BITS) - 1.0)))
    return result


def pairwise_spearman(distance: np.ndarray, labels: np.ndarray) -> float:
    index = np.triu_indices(int(labels.shape[0]), k=1)
    gaps = np.abs(np.asarray(labels, dtype=np.float64)[index[0]] - np.asarray(labels, dtype=np.float64)[index[1]])
    values = np.asarray(distance, dtype=np.float64)[index]
    spearman, _kendall = correlation(values, gaps)
    return spearman


def rank_bar(spearman: float, top3: float, regret: float) -> bool:
    return bool(
        np.isfinite(spearman)
        and np.isfinite(top3)
        and np.isfinite(regret)
        and float(spearman) >= float(SPEARMAN_BAR)
        and float(top3) >= float(TOP3_BAR)
        and float(regret) <= float(REGRET_BAR)
    )


def execution_informative(spearman: float, top3: float, regret: float, residual: float) -> bool:
    return bool(rank_bar(spearman, top3, regret) and np.isfinite(residual) and float(residual) >= float(RESIDUAL_BAR))


def choose_gate(
    *,
    l0_rank: bool,
    phy_rank: bool,
    act0_informative: bool,
    act0_gain: float,
    h_dyn: str,
) -> str:
    """DYN-A, then DYN-B, then DYN-C, otherwise DYN-D. Thresholds stay in protocol.py."""
    if bool(l0_rank) or bool(phy_rank):
        return "DYN-A"
    recovered = np.isfinite(act0_gain) and float(act0_gain) >= float(SPEARMAN_GAIN)
    if bool(act0_informative) and recovered:
        return "DYN-B"
    if str(h_dyn) in {"1", "3", "5", "8"}:
        return "DYN-C"
    return "DYN-D"


__all__ = [
    "choose_gate",
    "correlation",
    "execution_informative",
    "nearest_neighbor",
    "pairwise_spearman",
    "rank_bar",
    "ranking",
    "sign_accuracy",
    "unit_rows",
]
