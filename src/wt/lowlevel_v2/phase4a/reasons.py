"""Per-UAV projector reasons. Does not change SafetyProjector rules."""
from __future__ import annotations

import numpy as np

from wt.lowlevel_v2.phase1a.diagnose import predicted_next
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M, SERVICE_SAFETY_MARGIN_M
from wt.lowlevel_v2.phase4a.protocol import INTERVENE_EPS
from wt.lowlevel_v2.service_safe_envelope import next_in_envelope


def per_uav_constraint_bits(env, option, a_des: np.ndarray, a_projected: np.ndarray) -> dict[str, np.ndarray]:
    des = np.asarray(a_des, dtype=np.float64)
    proj = np.asarray(a_projected, dtype=np.float64)
    nxt = predicted_next(env, des)
    m_count = int(des.shape[0])
    pairwise = np.zeros(m_count, dtype=bool)
    for i in range(m_count):
        for j in range(m_count):
            if i == j:
                continue
            if float(np.linalg.norm(nxt[i] - nxt[j])) + 1.0e-9 < D_SAFE_M:
                pairwise[i] = True
                break
    envelope = ~np.asarray(next_in_envelope(env, option, nxt, SERVICE_SAFETY_MARGIN_M), dtype=bool)
    step = np.linalg.norm(des, axis=-1) > 1.0 + 1.0e-9
    intervened = np.linalg.norm(proj - des, axis=-1) > INTERVENE_EPS
    n_reasons = pairwise.astype(np.int32) + envelope.astype(np.int32) + step.astype(np.int32)
    multi = intervened & (n_reasons >= 2)
    pairwise_only = intervened & pairwise & (n_reasons == 1)
    service_only = intervened & envelope & ~pairwise & ~step
    step_only = intervened & step & ~pairwise & ~envelope
    collision_distill = intervened & pairwise
    return {
        "pairwise": pairwise,
        "envelope": envelope,
        "step": step,
        "intervened": intervened,
        "multi": multi,
        "pairwise_only": pairwise_only,
        "service_only": service_only,
        "step_only": step_only,
        "collision_distill": collision_distill,
        "n_reasons": n_reasons,
    }


__all__ = ["per_uav_constraint_bits"]
