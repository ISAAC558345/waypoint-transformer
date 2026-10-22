"""Fixed residual-scale action candidates. Single offset frozen in protocol. No grid."""
from __future__ import annotations

import numpy as np

from wt.lowlevel_v2.contracts import clip_unit_step
from wt.lowlevel_v2.phase4b.protocol import CANDIDATE_OFFSET


def _unit(vec: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(vec))
    if nrm < 1.0e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / nrm


def candidate_actions(a_base_u: np.ndarray, a_rl_u: np.ndarray | None, offset: float = CANDIDATE_OFFSET) -> dict[str, np.ndarray]:
    base = np.asarray(a_base_u, dtype=np.float64).reshape(2)
    fwd = _unit(base)
    left = np.array([-fwd[1], fwd[0]], dtype=np.float64)
    out = {
        "C0_base": clip_unit_step(base.reshape(1, 2))[0],
        "C2_left": clip_unit_step((base + float(offset) * left).reshape(1, 2))[0],
        "C3_right": clip_unit_step((base - float(offset) * left).reshape(1, 2))[0],
        "C4_less_fwd": clip_unit_step((base - float(offset) * fwd).reshape(1, 2))[0],
        "C5_more_fwd": clip_unit_step((base + float(offset) * fwd).reshape(1, 2))[0],
    }
    if a_rl_u is not None:
        out["C1_rl_desired"] = clip_unit_step(np.asarray(a_rl_u, dtype=np.float64).reshape(1, 2))[0]
    return out


__all__ = ["candidate_actions"]
