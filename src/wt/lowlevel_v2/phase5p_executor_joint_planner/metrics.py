"""Pair counts and the label-retraining gate. Thresholds are fixed here."""
from __future__ import annotations

import numpy as np

from .protocol import PAYLOAD_BLOCK_BITS


def quantize(values, block: float = PAYLOAD_BLOCK_BITS) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.round(array / float(block)) * float(block)


def signal_counts(values, block: float = PAYLOAD_BLOCK_BITS) -> dict[str, float]:
    """Ordered pairs. A pair is usable only when the block gap is at least one block."""
    levels = quantize(values, block)
    count = int(levels.size)
    positive = 0
    ties = 0
    for left in range(count):
        gap = levels[left] - levels
        positive += int(np.sum(gap >= float(block)))
        ties += int(np.sum(gap == 0.0)) - 1
    ordered = count * max(count - 1, 0)
    return {
        "n_assignments": float(count),
        "n_unique": float(np.unique(levels).size),
        "n_positive_pairs": float(positive),
        "n_ties": float(ties),
        "usable_pair_fraction": float(positive / ordered) if ordered else float("nan"),
    }


def choose_gate(
    *,
    train_rank_up: bool,
    train_payload_up: bool,
    dev_rank_up: bool,
    dev_payload_up: bool,
    dev_regret_down: bool,
    n_seeds_improved: int,
    n_seeds: int,
) -> str:
    """One conclusion. Development metrics are macro-averaged inside each state first."""
    if int(n_seeds) <= 0:
        raise ValueError("gate needs the trained seeds")
    majority = int(n_seeds_improved) * 2 > int(n_seeds)
    if bool(train_rank_up) and bool(train_payload_up) and not bool(dev_rank_up) and not bool(dev_payload_up):
        return "LABELTRAIN-D"
    if bool(dev_payload_up) and bool(dev_regret_down) and majority:
        return "LABELTRAIN-A"
    if bool(dev_rank_up) and not bool(dev_payload_up):
        return "LABELTRAIN-B"
    return "LABELTRAIN-C"


__all__ = ["choose_gate", "quantize", "signal_counts"]
