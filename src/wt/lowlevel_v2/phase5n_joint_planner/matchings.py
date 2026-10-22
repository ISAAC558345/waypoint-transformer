"""Maximal one-to-one assignments inside a per-UAV candidate mask."""
from __future__ import annotations

import numpy as np


def maximal_assignments(options: list[list[int]]) -> list[np.ndarray]:
    """Every exclusive assignment that cannot take another candidate buoy.

    A UAV may stay idle only when every buoy in its own list is already taken.
    Filling the earliest UAVs and then idling the rest is only one subset of
    this collection. Hungarian can give the same buoys to later UAVs, so those
    matchings have to be in the supervised set.
    """
    count = len(options)
    allowed = [tuple(dict.fromkeys(int(buoy) for buoy in row)) for row in options]
    assignment = np.full(count, -1, dtype=np.int64)
    found: list[np.ndarray] = []

    def maximal(taken: set[int]) -> bool:
        for uav, buoy in enumerate(assignment):
            if int(buoy) >= 0:
                continue
            if any(int(option) not in taken for option in allowed[uav]):
                return False
        return True

    def walk(index: int, taken: set[int]) -> None:
        if index == count:
            if maximal(taken):
                found.append(assignment.copy())
            return
        for buoy in allowed[index]:
            if buoy in taken:
                continue
            assignment[index] = buoy
            taken.add(buoy)
            walk(index + 1, taken)
            taken.remove(buoy)
        assignment[index] = -1
        walk(index + 1, taken)

    walk(0, set())
    if not found:
        raise RuntimeError("candidate mask produced no maximal assignment")
    return found


__all__ = ["maximal_assignments"]
