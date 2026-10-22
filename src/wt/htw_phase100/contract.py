"""Minimal high-level / low-level Task Option contract. No rematching."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wt.htw_phase9.const import IDLE_INDEX


@dataclass
class TaskOption:
    """High-level output consumed by a movement executor.

    ``urgency`` is optional. ``None`` keeps the previous target-only contract.
    When set, it is the shared buoy urgency frozen at the option decision.
    """

    targets: np.ndarray
    decision_time: int
    remaining_hold: int
    uav_mask: np.ndarray
    buoy_mask: np.ndarray
    association: np.ndarray
    urgency: np.ndarray | None = None

    def copy(self) -> "TaskOption":
        urgency = None if self.urgency is None else np.asarray(self.urgency, dtype=np.float64).copy()
        return TaskOption(
            targets=np.asarray(self.targets, dtype=np.int64).copy(),
            decision_time=int(self.decision_time),
            remaining_hold=int(self.remaining_hold),
            uav_mask=np.asarray(self.uav_mask, dtype=bool).copy(),
            buoy_mask=np.asarray(self.buoy_mask, dtype=bool).copy(),
            association=np.asarray(self.association, dtype=np.int64).copy(),
            urgency=urgency,
        )


def idle_code() -> int:
    return int(IDLE_INDEX)


def buoy_or_idle(token: int) -> int:
    value = int(token)
    if value < 0 or value >= IDLE_INDEX:
        return -1
    return value


__all__ = ["TaskOption", "buoy_or_idle", "idle_code"]
