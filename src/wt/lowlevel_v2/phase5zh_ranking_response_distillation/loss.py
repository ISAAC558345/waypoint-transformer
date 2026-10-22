"""State-conditional value ranking pairs. Pairs never cross states."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .protocol import RANDOM_PAIRS, TOP_INTERNAL, TOP_OUTSIDE, VALUE_TEMPERATURE, VALUE_TIE


def ranking_pairs(teacher: np.ndarray, rng: np.random.Generator, eps: float = VALUE_TIE, n_random: int = RANDOM_PAIRS) -> tuple[np.ndarray, np.ndarray]:
    """Higher-teacher index first. Ties within eps are dropped."""
    values = np.asarray(teacher, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    top_out = order[: int(TOP_OUTSIDE)]
    top_in = order[: int(TOP_INTERNAL)]
    rest = order[int(TOP_OUTSIDE) :]
    left: list[int] = []
    right: list[int] = []

    def add(first: int, second: int) -> None:
        gap = float(values[first] - values[second])
        if abs(gap) <= float(eps):
            return
        if gap > 0.0:
            left.append(int(first))
            right.append(int(second))
        else:
            left.append(int(second))
            right.append(int(first))

    for winner in top_out:
        for other in rest:
            add(int(winner), int(other))
    for index, winner in enumerate(top_in):
        for other in top_in[index + 1 :]:
            add(int(winner), int(other))
    count = int(values.shape[0])
    seen: set[tuple[int, int]] = set()
    tries = 0
    while len(seen) < int(n_random) and tries < int(n_random) * 30 and count >= 2:
        first = int(rng.integers(0, count))
        second = int(rng.integers(0, count))
        tries += 1
        if first == second:
            continue
        seen.add((first, second) if first < second else (second, first))
    for first, second in seen:
        add(first, second)
    if not left:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def pairwise_rank_loss(student: torch.Tensor, left: torch.Tensor, right: torch.Tensor, temperature: float = VALUE_TEMPERATURE) -> torch.Tensor:
    """softplus(-(V_i - V_j) / T) for teacher-ordered pairs. Empty pairs add nothing."""
    if int(left.numel()) == 0:
        return student.sum() * 0.0
    margin = (student.index_select(0, left) - student.index_select(0, right)) / float(temperature)
    return F.softplus(-margin).mean()


__all__ = ["pairwise_rank_loss", "ranking_pairs"]
