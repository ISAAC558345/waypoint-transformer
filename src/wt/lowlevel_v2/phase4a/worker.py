"""Phase-4A worker: frozen ResidualWorker plus pre-step collision-distill masks."""
from __future__ import annotations

import numpy as np

from wt.lowlevel_v2.phase1b.controller import ResidualWorker
from wt.lowlevel_v2.phase4a.protocol import refuse_phase4a
from wt.lowlevel_v2.phase4a.reasons import per_uav_constraint_bits


def classify_prestep(worker: ResidualWorker, rec, pos0: np.ndarray) -> None:
    saved = worker.env.uav_positions.copy()
    worker.env.uav_positions = pos0
    bits = per_uav_constraint_bits(worker.env, worker.option, rec.a_des, rec.a_projected)
    worker.env.uav_positions = saved
    rec.collision_distill_mask = bits["collision_distill"].astype(np.float32)
    rec.pairwise_mask = bits["pairwise"].astype(np.float32)
    rec.envelope_mask = bits["envelope"].astype(np.float32)
    rec.step_mask = bits["step"].astype(np.float32)
    rec.multi_mask = bits["multi"].astype(np.float32)
    rec.intervened_mask = bits["intervened"].astype(np.float32)
    rec.service_only_mask = bits["service_only"].astype(np.float32)
    rec.step_only_mask = bits["step_only"].astype(np.float32)
    rec.pairwise_only_mask = bits["pairwise_only"].astype(np.float32)
    rec.r_teacher = np.asarray(rec.a_projected - rec.a_base, dtype=np.float32)


class Phase4AWorker(ResidualWorker):
    def reset(self, seed: int):
        refuse_phase4a(int(seed))
        return super().reset(seed)

    def step(self, info: dict, *, deterministic: bool, zero_residual: bool = False):
        pos0 = np.asarray(self.env.uav_positions, dtype=np.float64)
        after, rec = ResidualWorker.step(self, info, deterministic=deterministic, zero_residual=zero_residual)
        if rec.preexisting_safety_violation:
            z = np.zeros(6, dtype=np.float32)
            rec.collision_distill_mask = z
            rec.pairwise_mask = z
            rec.envelope_mask = z
            rec.step_mask = z
            rec.multi_mask = z
            rec.intervened_mask = z
            rec.service_only_mask = z
            rec.step_only_mask = z
            rec.pairwise_only_mask = z
            rec.r_teacher = np.zeros((6, 2), dtype=np.float32)
            return after, rec
        classify_prestep(self, rec, pos0)
        return after, rec


def make_worker(high, actor, critic) -> Phase4AWorker:
    return Phase4AWorker(high, actor, critic, service_lock=True, trust_region=True)


__all__ = ["Phase4AWorker", "classify_prestep", "make_worker"]
