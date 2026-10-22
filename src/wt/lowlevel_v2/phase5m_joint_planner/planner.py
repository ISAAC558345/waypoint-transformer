"""Candidate generation and joint refinement. J(A) is not a low-level reward."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wt.lowlevel_v2.phase5i_joint_utility.search import enumerate_exclusive, largest_complete_pool, top_options


def candidate_lists(logits: np.ndarray, width: int, num_uavs: int, num_buoys: int) -> list[list[int]]:
    """Top-width buoys for each UAV. This list is not a final assignment."""
    matrix = np.asarray(logits, dtype=np.float64)
    lists = []
    for uav in range(int(num_uavs)):
        values = [float(matrix[uav, buoy]) for buoy in range(int(num_buoys))]
        lists.append(top_options(list(range(int(num_buoys))), values, int(width)))
    return lists


def greedy_refine(options: list[list[int]], evaluate: Callable[[np.ndarray], float]) -> tuple[np.ndarray, int]:
    """Add the idle UAV-buoy pair whose partial assignment has the highest J(A)."""
    count = len(options)
    assignment = np.full(count, -1, dtype=np.int64)
    remaining = set(range(count))
    taken: set[int] = set()
    calls = 0
    for _ in range(count):
        best_pair = None
        best_value = None
        for uav in sorted(remaining):
            for buoy in options[uav]:
                if int(buoy) in taken:
                    continue
                trial = assignment.copy()
                trial[int(uav)] = int(buoy)
                value = float(evaluate(trial))
                calls += 1
                if best_value is None or value > best_value:
                    best_value = value
                    best_pair = (int(uav), int(buoy))
        if best_pair is None:
            break
        assignment[best_pair[0]] = best_pair[1]
        remaining.remove(best_pair[0])
        taken.add(best_pair[1])
    return assignment, calls


def exhaustive_refine(options: list[list[int]], evaluate: Callable[[np.ndarray], float]) -> tuple[np.ndarray, int]:
    """Score every exclusive candidate assignment and keep the highest J(A)."""
    assignments = enumerate_exclusive(options)
    if not assignments:
        raise RuntimeError("candidate set produced no assignment")
    calls = 0
    best = assignments[0]
    best_value = None
    for assignment in assignments:
        value = float(evaluate(assignment))
        calls += 1
        if best_value is None or value > best_value:
            best_value = value
            best = np.asarray(assignment, dtype=np.int64)
    return np.asarray(best, dtype=np.int64), calls


def candidate_pool(logits: np.ndarray, num_uavs: int, num_buoys: int, limit: int) -> tuple[int, list[list[int]], list[np.ndarray]]:
    """Widest top-k exclusive pool that stays inside the search budget."""
    by_width = {
        width: candidate_lists(logits, width, num_uavs, num_buoys)
        for width in (1, 2, 3)
    }
    width, assignments = largest_complete_pool({width: options for width, options in by_width.items()}, int(limit))
    if not assignments:
        raise RuntimeError("hierarchical search found no exclusive assignment")
    return int(width), by_width[int(width)], assignments


__all__ = ["candidate_lists", "candidate_pool", "exhaustive_refine", "greedy_refine"]
