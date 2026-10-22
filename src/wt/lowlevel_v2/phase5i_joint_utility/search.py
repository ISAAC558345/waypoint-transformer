"""Exclusive assignments inside each UAV's top marginal targets."""
from __future__ import annotations

import numpy as np


def top_options(candidate_ids: list[int], values: list[float], width: int) -> list[int]:
    order = sorted(range(len(candidate_ids)), key=lambda index: (-float(values[index]), int(candidate_ids[index])))
    chosen = []
    for index in order:
        buoy = int(candidate_ids[index])
        if buoy not in chosen:
            chosen.append(buoy)
        if len(chosen) >= int(width):
            break
    return chosen


def enumerate_exclusive(options: list[list[int]]) -> list[np.ndarray]:
    """Every exclusive assignment that gives each UAV one of its own options when one remains."""
    count = len(options)
    assignment = np.full(count, -1, dtype=np.int64)
    found: list[np.ndarray] = []

    def walk(index: int, taken: set[int]) -> None:
        if index == count:
            found.append(assignment.copy())
            return
        placed = False
        for buoy in options[index]:
            if int(buoy) in taken:
                continue
            assignment[index] = int(buoy)
            taken.add(int(buoy))
            walk(index + 1, taken)
            taken.remove(int(buoy))
            placed = True
        if not placed:
            assignment[index] = -1
            walk(index + 1, taken)

    walk(0, set())
    return found


def largest_complete_pool(options_by_width: dict[int, list[list[int]]], limit: int) -> tuple[int, list[np.ndarray]]:
    """Use the widest top-k pool whose exclusive set still fits the evaluation budget."""
    selected_width = 1
    selected: list[np.ndarray] = []
    for width in sorted(options_by_width):
        assignments = enumerate_exclusive(options_by_width[width])
        if len(assignments) > int(limit) and selected:
            break
        selected_width = int(width)
        selected = assignments
        if len(assignments) > int(limit):
            break
    return selected_width, selected


__all__ = ["enumerate_exclusive", "largest_complete_pool", "top_options"]
