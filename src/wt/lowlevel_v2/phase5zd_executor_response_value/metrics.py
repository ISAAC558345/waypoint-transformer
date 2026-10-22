"""Gate for the matched H=1 response value. Thresholds stay in protocol.py."""
from __future__ import annotations

import numpy as np


def regression_block(prediction, labels) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    residual = pred - truth
    center = float(np.mean(truth))
    square = float(np.sum((truth - center) ** 2))
    huber = np.where(np.abs(residual) <= 1.0, 0.5 * residual**2, np.abs(residual) - 0.5)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / square) if square > 1.0e-12 else float("nan"),
        "bias": float(np.mean(residual)),
        "huber": float(np.mean(huber)),
    }


def choose_gate(*, integrity_ok: bool, ref1_reproduced: bool, confirmed: bool, mean_improved: bool) -> str:
    """RESP-STATE-FAIL, then RESP-D, then RESP-A, then RESP-B, otherwise RESP-C."""
    if not bool(integrity_ok):
        return "RESP-STATE-FAIL"
    if not bool(ref1_reproduced):
        return "RESP-D"
    if bool(confirmed):
        return "RESP-A"
    if bool(mean_improved):
        return "RESP-B"
    return "RESP-C"


__all__ = ["choose_gate", "regression_block"]
