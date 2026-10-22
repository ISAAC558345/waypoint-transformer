"""State-conditional response ranking. No model update."""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .metrics import pairwise_accuracy
from .protocol import PAIR_EPS


def correlations(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    left = np.asarray(truth, dtype=np.float64)
    right = np.asarray(prediction, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3 or float(np.std(left[mask])) <= 1.0e-12 or float(np.std(right[mask])) <= 1.0e-12:
        return float("nan"), float("nan")
    return float(spearmanr(left[mask], right[mask]).statistic), float(kendalltau(left[mask], right[mask]).statistic)


def overlap(truth: np.ndarray, prediction: np.ndarray, width: int) -> float:
    if truth.size < width or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        return float("nan")
    true_top = set(np.argsort(-truth, kind="stable")[:width].tolist())
    pred_top = set(np.argsort(-prediction, kind="stable")[:width].tolist())
    return float(len(true_top & pred_top) / float(width))


def rank_block(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    spearman, kendall = correlations(truth, prediction)
    top1 = float("nan")
    finite = np.isfinite(truth) & np.isfinite(prediction)
    if int(finite.sum()) >= 1 and float(np.std(truth[finite])) > 1.0e-12 and float(np.std(prediction[finite])) > 1.0e-12:
        top1 = float(int(np.nanargmax(truth)) == int(np.nanargmax(prediction)))
    return {
        "spearman": spearman,
        "kendall": kendall,
        "pairwise": pairwise_accuracy(truth, prediction, PAIR_EPS),
        "top1": top1,
        "top3": overlap(truth, prediction, 3),
        "top5": overlap(truth, prediction, 5),
    }


def diagnose(spearman: float, pairwise: float, bias: float, scale: float, std_true: float) -> str:
    if not np.isfinite(spearman) or not np.isfinite(pairwise):
        return "UNDEFINED"
    if float(spearman) < 0.50 or float(pairwise) < 0.70:
        return "ORDERING_ERROR"
    scale_bad = float(scale) < 0.75 or float(scale) > 1.25
    bias_bad = abs(float(bias)) > 0.25 * max(float(std_true), 1.0e-8)
    if (scale_bad or bias_bad) and float(spearman) >= 0.80 and float(pairwise) >= 0.85:
        return "CALIBRATION_ERROR"
    if scale_bad or bias_bad:
        return "MIXED"
    return "ORDERING_OK"


__all__ = ["correlations", "diagnose", "rank_block"]
