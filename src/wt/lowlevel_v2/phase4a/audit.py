"""Frozen teacher-signal audit on LLV2_train. No optimizer.step."""
from __future__ import annotations

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import load_frozen_high, make_networks
from wt.lowlevel_v2.phase4a.protocol import TEACHER_MIN_ACTIVE_FRACTION, refuse_phase4a
from wt.lowlevel_v2.phase4a.worker import make_worker
from wt.round30.ppo import seed_everything


def teacher_signal_audit(train_seeds: list[int], *, model_seed: int = 0) -> dict:
    for s in train_seeds:
        refuse_phase4a(int(s))
    seed_everything(int(model_seed))
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    actor.eval()
    critic.eval()
    worker = make_worker(high, actor, critic)
    n_agent = 0
    n_active = 0
    n_eligible = 0
    n_pair_event = 0
    n_intervened = 0
    near = mid = far = 0
    elig_near = elig_mid = elig_far = 0
    d_all = []
    rows = []
    for seed in train_seeds:
        refuse_phase4a(int(seed))
        info = worker.reset(int(seed))
        while True:
            pos0 = np.asarray(worker.env.uav_positions, dtype=np.float64)
            info, rec = worker.step(info, deterministic=True, zero_residual=False)
            d_near = np.asarray(rec.obs["d_nearest"], dtype=np.float64)
            d_all.extend(d_near.tolist())
            active = np.asarray(rec.policy_active_mask) > 0.5
            distill = np.asarray(rec.collision_distill_mask) > 0.5
            pair = np.asarray(rec.pairwise_mask) > 0.5
            inter = np.asarray(rec.intervened_mask) > 0.5
            n_agent += 6
            n_active += int(np.sum(active))
            n_eligible += int(np.sum(active & distill))
            n_pair_event += int(np.sum(inter & pair))
            n_intervened += int(np.sum(inter))
            if rec.done or worker.env.t >= worker.env.config.episode_horizon:
                break
            _ = pos0
    d_all = np.asarray(d_all, dtype=np.float64)
    q1, q2 = np.quantile(d_all, [1 / 3, 2 / 3]) if d_all.size else (0.0, 0.0)
    for seed in train_seeds:
        info = worker.reset(int(seed))
        while True:
            info, rec = worker.step(info, deterministic=True, zero_residual=False)
            d_near = np.asarray(rec.obs["d_nearest"], dtype=np.float64)
            active = np.asarray(rec.policy_active_mask) > 0.5
            distill = np.asarray(rec.collision_distill_mask) > 0.5
            for u in range(6):
                band = "near" if d_near[u] <= q1 else ("middle" if d_near[u] <= q2 else "far")
                if band == "near":
                    near += 1
                elif band == "middle":
                    mid += 1
                else:
                    far += 1
                if active[u] and distill[u]:
                    if band == "near":
                        elig_near += 1
                    elif band == "middle":
                        elig_mid += 1
                    else:
                        elig_far += 1
            if rec.done or worker.env.t >= worker.env.config.episode_horizon:
                break
    frac = float(n_eligible / max(n_active, 1))
    sparse = bool(frac < TEACHER_MIN_ACTIVE_FRACTION)
    summary = {
        "n_agent_steps": n_agent,
        "n_active": n_active,
        "n_collision_distill_eligible_active": n_eligible,
        "eligible_frac_of_active": frac,
        "pairwise_intervened_agent_steps": n_pair_event,
        "intervened_agent_steps": n_intervened,
        "pairwise_event_rate": float(n_pair_event / max(n_agent, 1)),
        "near_steps": near,
        "middle_steps": mid,
        "far_steps": far,
        "eligible_near": elig_near,
        "eligible_middle": elig_mid,
        "eligible_far": elig_far,
        "d_near_q1": float(q1),
        "d_near_q2": float(q2),
        "collision_teacher_signal_too_sparse": sparse,
        "threshold": TEACHER_MIN_ACTIVE_FRACTION,
        "model_seed_for_init": int(model_seed),
        "trained": False,
    }
    rows.append(summary)
    return {"summary": summary, "rows": rows}


__all__ = ["teacher_signal_audit"]
