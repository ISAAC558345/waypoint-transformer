"""Shared task urgency. Current queue only. No future arrivals."""
from __future__ import annotations

import numpy as np

from .protocol import URGENCY_EPS


def shared_task_urgency(queue: np.ndarray) -> np.ndarray:
    """u_k(t) = q_k(t) / (max_j q_j(t) + eps), clipped to [0, 1]."""
    values = np.asarray(queue, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("urgency expects one queue vector")
    scale = float(np.max(values)) + float(URGENCY_EPS)
    urgency = np.clip(values / scale, 0.0, 1.0)
    if not np.all(np.isfinite(urgency)):
        raise RuntimeError("non-finite shared urgency")
    return urgency


__all__ = ["shared_task_urgency"]
