"""Label discrimination, alignment, and the single next-step conclusion."""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .protocol import (
    COVERAGE_STATES,
    MIN_DISCRIMINATING_STATES,
    PAYLOAD_BLOCK_BITS,
    POOR_REGRET,
    POOR_SPEARMAN,
    POOR_TOP1,
)


def quantize_block(values, block: float = PAYLOAD_BLOCK_BITS) -> np.ndarray:
    """Round to the delivered payload block. Sub-block gaps are one label level."""
    array = np.asarray(values, dtype=np.float64)
    return np.round(array / float(block)) * float(block)


def tie_fraction(values, block: float = PAYLOAD_BLOCK_BITS) -> float:
    levels = quantize_block(values, block)
    count = int(levels.size)
    if count < 2:
        return float("nan")
    tied = 0
    total = 0
    for index in range(count):
        same = int(np.sum(levels[index + 1 :] == levels[index]))
        tied += same
        total += count - index - 1
    return float(tied / total) if total else float("nan")


def discrimination(values, block: float = PAYLOAD_BLOCK_BITS) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    levels = quantize_block(array, block)
    unique = np.unique(levels)
    span = float(np.max(array) - np.min(array)) if array.size else float("nan")
    return {
        "min": float(np.min(array)) if array.size else float("nan"),
        "max": float(np.max(array)) if array.size else float("nan"),
        "std": float(np.std(array)) if array.size else float("nan"),
        "n_unique": float(unique.size),
        "tie_fraction": tie_fraction(array, block),
        "range_bits": span,
        "discriminates": float(unique.size >= 2 and span + 1.0e-6 >= float(block)),
    }


def rank_correlation(left, right) -> tuple[float, float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size < 3 or np.unique(quantize_block(a)).size < 2 or np.unique(quantize_block(b)).size < 2:
        return float("nan"), float("nan")
    spearman = spearmanr(a, b)
    kendall = kendalltau(a, b)
    spearman_value = getattr(spearman, "statistic", spearman.correlation)
    kendall_value = getattr(kendall, "statistic", kendall.correlation)
    return float(spearman_value), float(kendall_value)


def top1_overlap(j_fly, j_mappo, block: float = PAYLOAD_BLOCK_BITS) -> float:
    """1 when the stable J_FLY maximizer is also a J_MAPPO maximizer."""
    fly = np.asarray(j_fly, dtype=np.float64)
    executed = np.asarray(j_mappo, dtype=np.float64)
    fly_levels = quantize_block(fly, block)
    executed_levels = quantize_block(executed, block)
    chosen = int(np.argmax(fly_levels))
    return float(executed_levels[chosen] == float(np.max(executed_levels)))


def top3_recovery(j_fly, j_mappo, block: float = PAYLOAD_BLOCK_BITS) -> float:
    """1 when a stable top-3 J_FLY index is a J_MAPPO maximizer.

    Fully tied J_FLY makes those three indices an arbitrary prefix. Callers
    should read n_unique before treating the bit as alignment.
    """
    fly = np.asarray(j_fly, dtype=np.float64)
    executed = np.asarray(j_mappo, dtype=np.float64)
    fly_levels = quantize_block(fly, block)
    executed_levels = quantize_block(executed, block)
    order = np.argsort(-fly_levels, kind="stable")
    width = min(3, int(order.size))
    best = float(np.max(executed_levels))
    return float(np.any(executed_levels[order[:width]] == best))


def regret(j_fly, j_mappo, block: float = PAYLOAD_BLOCK_BITS) -> tuple[float, float]:
    fly = np.asarray(j_fly, dtype=np.float64)
    executed = np.asarray(j_mappo, dtype=np.float64)
    chosen = int(np.argmax(quantize_block(fly, block)))
    best = float(np.max(executed))
    gap = float(best - float(executed[chosen]))
    return gap, float(gap / (best + 1.0))


def choose_label(
    *,
    n_states: int,
    n_discriminating: int,
    pooled_spearman: float,
    mean_normalized_regret: float,
    mean_top1: float,
    n_outside_beats_pool: int,
) -> str:
    """One of LABEL-A, LABEL-B, LABEL-C, LABEL-D. Thresholds live in the protocol."""
    if int(n_states) <= 0:
        raise ValueError("label choice needs the audited states")
    if int(n_discriminating) < int(MIN_DISCRIMINATING_STATES):
        return "LABEL-B"
    # Coverage is an outside assignment beating the whole top-3 pool on most states.
    # A pool whose own maximum is below one block already fails the discrimination count.
    if int(n_outside_beats_pool) >= int(COVERAGE_STATES):
        return "LABEL-D"
    spearman_poor = (not np.isfinite(pooled_spearman)) or float(pooled_spearman) < float(POOR_SPEARMAN)
    regret_poor = (not np.isfinite(mean_normalized_regret)) or float(mean_normalized_regret) >= float(POOR_REGRET)
    top1_poor = (not np.isfinite(mean_top1)) or float(mean_top1) <= float(POOR_TOP1)
    if spearman_poor or regret_poor or top1_poor:
        return "LABEL-A"
    return "LABEL-C"


__all__ = [
    "choose_label",
    "discrimination",
    "quantize_block",
    "rank_correlation",
    "regret",
    "tie_fraction",
    "top1_overlap",
    "top3_recovery",
]
