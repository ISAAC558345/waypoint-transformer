"""H=15 frozen-assignment shadow utility on the real L3-B environment."""
from __future__ import annotations

import numpy as np

from wt.lowlevel_v2.phase4b.snapshot import restore_env

from .protocol import HOLD_STEPS, refuse_compat


def restore_worker(worker, state: dict) -> dict:
    refuse_compat(int(state["scenario"]))
    info = worker.reset(int(state["scenario"]))
    restore_env(worker.env, state["env"])
    worker.last_arrival = np.asarray(state["last_arrival"], dtype=np.float32).copy()
    worker.last_positions = np.asarray(state["last_positions"], dtype=np.float32).copy()
    worker.last_movement = np.asarray(state["last_movement"], dtype=np.float64).copy()
    worker.pending = 0
    worker.frozen_assignment = None
    info = worker.env.info()
    info["terminated"] = False
    info["arrivals_blocks"] = np.asarray(state["last_arrival"], dtype=np.float32).copy()
    return info


def shadow_h15(worker, state: dict, assignment: np.ndarray, *, zero_residual: bool) -> dict:
    info = restore_worker(worker, state)
    worker.bind_assignment(np.asarray(assignment, dtype=np.int64), info)
    start_payload = float(worker.env.payload_bits)
    start_progress = float(np.sum(worker.env.progress))
    team_rate = 0.0
    n_step = 0
    min_sep = float("inf")
    fail = 0
    coll = 0
    for _ in range(int(HOLD_STEPS)):
        if worker.env.t >= int(worker.env.config.episode_horizon):
            break
        info, rec = worker.step(info, deterministic=True, zero_residual=zero_residual)
        n_step += 1
        team_rate += float(np.sum(worker.env.last_rate))
        min_sep = min(min_sep, float(rec.min_sep))
        fail += int(rec.projector_failure)
        if rec.done:
            break
    payload = float(worker.env.payload_bits - start_payload)
    return {
        "payload_increment": payload,
        "completed_progress": float(np.sum(worker.env.progress) - start_progress),
        "team_rate_integral": team_rate,
        "n_step": n_step,
        "min_sep": min_sep if n_step else float("nan"),
        "projector_failure": fail,
        "idle_count": float(np.sum(np.asarray(assignment, dtype=np.int64) < 0)),
    }


__all__ = ["restore_worker", "shadow_h15"]
