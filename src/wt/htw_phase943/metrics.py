"""Phase-9.4.3 metrics helpers."""
from __future__ import annotations

import numpy as np


def ppv(payloads: np.ndarray) -> float:
    values = np.asarray(payloads, dtype=np.float64)
    return float(np.std(values) / (abs(float(np.mean(values))) + 1.0e-9))


__all__ = ["ppv"]
