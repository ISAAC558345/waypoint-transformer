"""Joint label J(A). Lambda is fixed. This is not a low-level reward."""
from __future__ import annotations

import numpy as np

from wt.l3a.env import conflict_mask

from .protocol import LAMBDA_CONFLICT


def duplicate_count(assignment: np.ndarray) -> int:
    chosen = [int(item) for item in assignment if int(item) >= 0]
    return int(len(chosen) - len(set(chosen)))


def conflict_rate(assignment: np.ndarray, num_buoys: int) -> float:
    blocked = conflict_mask(np.asarray(assignment, dtype=np.int64), int(num_buoys))
    if blocked.size == 0:
        return 0.0
    return float(np.mean(blocked))


def conflict_loss(communication_drop: float, duplicate: int, conflicts: float, interference: float) -> dict[str, float]:
    """Diagnostics plus the bit-valued loss used by J(A).

    Duplicate count, conflict pairs, and interference stay in their own units.
    The scalar subtracted in J(A) is the communication drop in bits, so a
    service wipe is not subtracted a second time.
    """
    drop = float(max(0.0, communication_drop))
    return {
        "duplicate_targets": float(duplicate),
        "competition_pairs": float(conflicts),
        "interference": float(interference),
        "communication_drop": drop,
        "conflict_loss": drop,
    }


def joint_utility(sum_individual: float, joint_payload: float, *, lam: float = LAMBDA_CONFLICT) -> float:
    drop = max(0.0, float(sum_individual) - float(joint_payload))
    return float(float(sum_individual) - float(lam) * drop)


__all__ = ["conflict_loss", "conflict_rate", "duplicate_count", "joint_utility"]
