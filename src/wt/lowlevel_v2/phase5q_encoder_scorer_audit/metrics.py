"""Gate rules and the offline additivity audit. These are not deployment models."""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .protocol import (
    CLEAR_PAYLOAD_BITS,
    COLLAPSE_EPS,
    LOSS_DROP,
    PAYLOAD_BLOCK_BITS,
    R2_MARGIN,
    SPEARMAN_MARGIN,
)


def quantize(values, block: float = PAYLOAD_BLOCK_BITS) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.round(array / float(block)) * float(block)


def is_collapsed(scores, eps: float = COLLAPSE_EPS) -> bool:
    array = np.asarray(scores, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return True
    return float(np.std(array)) < float(eps) or float(np.max(array) - np.min(array)) < float(eps)


def rank_correlation(scores, labels) -> tuple[float, float]:
    left = np.asarray(scores, dtype=np.float64)
    right = np.asarray(labels, dtype=np.float64)
    if left.size < 3 or np.unique(quantize(right)).size < 2 or is_collapsed(left):
        return float("nan"), float("nan")
    spearman = spearmanr(left, right)
    kendall = kendalltau(left, right)
    spearman_value = getattr(spearman, "statistic", spearman.correlation)
    kendall_value = getattr(kendall, "statistic", kendall.correlation)
    return float(spearman_value), float(kendall_value)


def metrics_at(scores, labels, index: int) -> dict[str, float]:
    """Same state metrics, with the decoded assignment supplied by the caller."""
    measured = assignment_metrics(scores, labels)
    levels = quantize(labels)
    chosen = int(index)
    best = float(measured["oracle_j_mappo"])
    regret = float(best - float(levels[chosen]))
    measured["top1"] = float(float(levels[chosen]) == best)
    measured["j_mappo"] = float(np.asarray(labels, dtype=np.float64)[chosen])
    measured["regret"] = regret
    measured["normalized_regret"] = float(regret / (best + 1.0))
    measured["chosen"] = float(chosen)
    return measured


def assignment_metrics(scores, labels) -> dict[str, float]:
    """State-local Top-1 and regret. The chosen assignment is the stable argmax."""
    values = np.asarray(scores, dtype=np.float64)
    levels = quantize(labels)
    order = np.argsort(-values, kind="stable")
    chosen = int(order[0])
    best = float(np.max(levels))
    regret = float(best - float(levels[chosen]))
    spearman, kendall = rank_correlation(values, labels)
    width = min(3, int(order.size))
    return {
        "spearman": spearman,
        "kendall": kendall,
        "top1": float(float(levels[chosen]) == best),
        "top3": float(np.any(levels[order[:width]] == best)),
        "j_mappo": float(np.asarray(labels, dtype=np.float64)[chosen]),
        "oracle_j_mappo": best,
        "regret": regret,
        "normalized_regret": float(regret / (best + 1.0)),
        "score_std": float(np.std(values)) if np.all(np.isfinite(values)) else float("nan"),
        "score_span": float(np.max(values) - np.min(values)) if np.all(np.isfinite(values)) else float("nan"),
        "collapsed": float(is_collapsed(values)),
        "chosen": float(chosen),
    }


def train_fit(*, first_loss: float, final_loss: float, collapsed_fraction: float) -> bool:
    """A train fit needs a lower ranking loss and a lifted score collapse."""
    if not np.isfinite(first_loss) or not np.isfinite(final_loss) or not np.isfinite(collapsed_fraction):
        return False
    loss_down = float(final_loss) < float(first_loss) - float(LOSS_DROP)
    return bool(loss_down and float(collapsed_fraction) < 0.5)


def stable_improvement(delta_j, delta_regret) -> bool:
    """Mean payload rises by one block, mean regret falls, and a majority of seeds do both."""
    payload = np.asarray(list(delta_j), dtype=np.float64)
    regret = np.asarray(list(delta_regret), dtype=np.float64)
    if payload.size == 0 or payload.shape != regret.shape:
        return False
    if not np.all(np.isfinite(payload)) or not np.all(np.isfinite(regret)):
        return False
    improved = int(np.sum((payload > 0.0) & (regret < 0.0)))
    majority = improved * 2 > int(payload.size)
    return bool(float(np.mean(payload)) >= float(CLEAR_PAYLOAD_BITS) and float(np.mean(regret)) < 0.0 and majority)


def pairwise_beats_additive(additive: dict[str, float], pairwise: dict[str, float]) -> bool:
    """Dev-set audit. At least two margins must move, and regret must not get worse."""
    if not np.isfinite(additive["normalized_regret"]) or not np.isfinite(pairwise["normalized_regret"]):
        return False
    if float(pairwise["normalized_regret"]) > float(additive["normalized_regret"]):
        return False
    wins = 0
    if np.isfinite(additive["r2"]) and np.isfinite(pairwise["r2"]) and float(pairwise["r2"]) >= float(additive["r2"]) + float(R2_MARGIN):
        wins += 1
    if np.isfinite(additive["spearman"]) and np.isfinite(pairwise["spearman"]) and float(pairwise["spearman"]) >= float(additive["spearman"]) + float(SPEARMAN_MARGIN):
        wins += 1
    if np.isfinite(additive["top1"]) and np.isfinite(pairwise["top1"]) and float(pairwise["top1"]) > float(additive["top1"]):
        wins += 1
    if float(pairwise["normalized_regret"]) < float(additive["normalized_regret"]):
        wins += 1
    return wins >= 2


def choose_gate(
    *,
    b_dev_improved: bool,
    c_dev_improved: bool,
    b_train_fit: bool,
    c_train_fit: bool,
    pairwise_better: bool,
) -> str:
    """One bottleneck. Encoder training and the new scorer are never combined here."""
    if bool(b_dev_improved) and bool(c_dev_improved):
        return "Q-C"
    if bool(b_dev_improved) and not bool(c_dev_improved):
        return "Q-A"
    if bool(c_dev_improved) and not bool(b_dev_improved) and bool(pairwise_better):
        return "Q-B"
    if not bool(b_dev_improved) and not bool(c_dev_improved):
        if bool(b_train_fit) or bool(c_train_fit):
            return "Q-D"
        return "Q-E"
    return "Q-E"


def _targets(assignments: np.ndarray) -> list[int]:
    values = {int(item) for item in np.asarray(assignments).reshape(-1) if int(item) >= 0}
    if not values:
        raise ValueError("assignments contain no selected buoys")
    return sorted(values)


def design_matrix(assignments: np.ndarray, targets: list[int], *, pairwise: bool) -> np.ndarray:
    rows = np.asarray(assignments, dtype=np.int64)
    count, uavs = int(rows.shape[0]), int(rows.shape[1])
    columns = []
    for uav in range(uavs):
        for buoy in targets:
            columns.append((rows[:, uav] == int(buoy)).astype(np.float64))
    if pairwise:
        for left in range(uavs):
            for right in range(left + 1, uavs):
                for buoy_left in targets:
                    for buoy_right in targets:
                        if int(buoy_left) == int(buoy_right):
                            continue
                        active = (rows[:, left] == int(buoy_left)) & (rows[:, right] == int(buoy_right))
                        columns.append(active.astype(np.float64))
    if not columns:
        return np.zeros((count, 0), dtype=np.float64)
    return np.stack(columns, axis=1)


def fit_linear(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Unpenalized least squares. The column mask is chosen on the training rows only."""
    keep = np.std(np.asarray(x_train, dtype=np.float64), axis=0) > 0.0
    train = np.asarray(x_train, dtype=np.float64)[:, keep]
    evalu = np.asarray(x_eval, dtype=np.float64)[:, keep]
    train_design = np.concatenate((np.ones((train.shape[0], 1)), train), axis=1)
    eval_design = np.concatenate((np.ones((evalu.shape[0], 1)), evalu), axis=1)
    coefficient, *_ = np.linalg.lstsq(train_design, np.asarray(y_train, dtype=np.float64), rcond=None)
    return eval_design @ coefficient, train_design @ coefficient, int(keep.sum()) + 1


def regression_metrics(predictions, labels) -> dict[str, float]:
    pred = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    residual = truth - pred
    total = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = float("nan") if total <= 1.0e-12 else float(1.0 - np.sum(residual ** 2) / total)
    spearman, kendall = rank_correlation(pred, truth)
    ranked = assignment_metrics(pred, truth)
    return {
        "r2": r2,
        "mae": float(np.mean(np.abs(residual))),
        "spearman": spearman,
        "kendall": kendall,
        "top1": ranked["top1"],
        "normalized_regret": ranked["normalized_regret"],
    }


__all__ = [
    "assignment_metrics",
    "choose_gate",
    "design_matrix",
    "fit_linear",
    "is_collapsed",
    "metrics_at",
    "pairwise_beats_additive",
    "quantize",
    "rank_correlation",
    "regression_metrics",
    "stable_improvement",
    "train_fit",
    "_targets",
]
