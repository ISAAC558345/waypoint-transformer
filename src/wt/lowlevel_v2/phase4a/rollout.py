"""Rollout storage for collision distillation. PPO log_prob still uses sampled residual."""
from __future__ import annotations

import numpy as np
import torch

from wt.lowlevel_v2.critic_state import expected_critic_dim
from wt.lowlevel_v2.phase1b.controller import ResidualWorker
from wt.lowlevel_v2.phase4a.protocol import refuse_phase4a


def collect_rollout(workers: list[ResidualWorker], agent, horizon: int) -> dict[str, np.ndarray]:
    n_env = len(workers)
    keys = {
        "self": (horizon, n_env, 6, 5),
        "task": (horizon, n_env, 6, 8),
        "communication": (horizon, n_env, 6, 4),
        "neighbor_set": (horizon, n_env, 6, 5, 11),
        "d_nearest": (horizon, n_env, 6),
        "global_state": (horizon, n_env, expected_critic_dim(6, 12)),
        "pre_tanh": (horizon, n_env, 6, 2),
        "log_prob_uav": (horizon, n_env, 6),
        "policy_active_mask": (horizon, n_env, 6),
        "collision_distill_mask": (horizon, n_env, 6),
        "a_base": (horizon, n_env, 6, 2),
        "r_teacher": (horizon, n_env, 6, 2),
        "service_lock": (horizon, n_env, 6),
        "base_zero_lock": (horizon, n_env, 6),
        "values": (horizon, n_env),
        "rewards": (horizon, n_env),
        "dones": (horizon, n_env),
    }
    storage = {key: np.zeros(shape, dtype=np.float32) for key, shape in keys.items()}
    infos = [w.env.info() for w in workers]
    for t in range(horizon):
        for e, worker in enumerate(workers):
            infos[e], rec = worker.step(infos[e], deterministic=False)
            storage["self"][t, e] = rec.obs["self"]
            storage["task"][t, e] = rec.obs["task"]
            storage["communication"][t, e] = rec.obs["communication"]
            storage["neighbor_set"][t, e] = rec.obs["neighbor_set"]
            storage["d_nearest"][t, e] = rec.obs["d_nearest"]
            storage["global_state"][t, e] = rec.global_state
            storage["pre_tanh"][t, e] = rec.pre_tanh
            storage["log_prob_uav"][t, e] = rec.log_prob_uav
            storage["policy_active_mask"][t, e] = rec.policy_active_mask
            storage["collision_distill_mask"][t, e] = rec.collision_distill_mask
            storage["a_base"][t, e] = rec.a_base
            storage["r_teacher"][t, e] = rec.r_teacher
            storage["service_lock"][t, e] = rec.service_lock.astype(np.float32)
            storage["base_zero_lock"][t, e] = rec.base_zero_lock.astype(np.float32)
            storage["values"][t, e] = rec.value
            storage["rewards"][t, e] = rec.reward
            storage["dones"][t, e] = float(rec.done)
            if rec.done:
                seed = int(worker._next_seed())
                refuse_phase4a(seed)
                infos[e] = worker.reset(seed)
    last_values = np.zeros(n_env, dtype=np.float32)
    for e, worker in enumerate(workers):
        if storage["dones"][-1, e] > 0.5:
            last_values[e] = 0.0
            continue
        from wt.lowlevel_v2.critic_state import build_critic_state

        glob = build_critic_state(worker.env, worker.option, worker.last_movement)["vector"]
        with torch.no_grad():
            last_values[e] = float(agent.critic(torch.as_tensor(glob, dtype=torch.float32)).cpu())
    storage["last_values"] = last_values
    return storage


__all__ = ["collect_rollout"]
