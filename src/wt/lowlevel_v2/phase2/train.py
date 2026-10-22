"""Train one low-level seed with frozen Phase-1B method. Does not load other seeds' weights."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import ResidualWorker, load_frozen_high, make_networks
from wt.lowlevel_v2.phase1b.mappo import ICARMAPPOLocked
from wt.lowlevel_v2.phase1b.run_phase1b import SeedCursor, attach_cursor, collect_rollout
from wt.round30.ppo import seed_everything

from .eval import evaluate_validation
from .protocol import (
    EVAL_EVERY,
    EXPECTED_HIGH_SHA,
    OUTPUT,
    PILOT_UPDATES_N,
    PPO,
    SMOKE_UPDATES_N,
    high_sha,
    vector_seed_pilot,
    vector_seed_smoke,
)


def train_seed(model_seed: int, train_seeds: list[int], curve_seeds: list[int]) -> dict:
    started = time.time()
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("high SHA mismatch")
    seed_everything(int(model_seed))
    high, digest = load_frozen_high()
    actor, critic = make_networks()
    agent = ICARMAPPOLocked(actor, critic, PPO)
    init_state = copy.deepcopy(agent.state_dict())
    workers = [ResidualWorker(high, actor, critic, service_lock=True) for _ in range(int(PPO.num_envs))]
    smoke_cursor = SeedCursor(train_seeds, np.random.default_rng(vector_seed_smoke(model_seed)))
    attach_cursor(workers, smoke_cursor)
    for worker in workers:
        worker.reset(smoke_cursor.next())
    smoke_hist = []
    env_smoke = 0
    for _ in range(int(SMOKE_UPDATES_N)):
        storage = collect_rollout(workers, agent, int(PPO.rollout_length))
        stats = agent.update(storage)
        env_smoke += int(PPO.rollout_length) * len(workers)
        smoke_hist.append(stats)
        if not np.isfinite(stats["policy_loss"]):
            raise RuntimeError("smoke NaN")
    agent.load_state_dict(init_state)
    seed_everything(int(model_seed))
    actor.load_state_dict(init_state["actor"])
    critic.load_state_dict(init_state["critic"])
    agent = ICARMAPPOLocked(actor, critic, PPO)
    agent.load_state_dict(init_state)
    workers = [ResidualWorker(high, actor, critic, service_lock=True) for _ in range(int(PPO.num_envs))]
    cursor = SeedCursor(train_seeds, np.random.default_rng(vector_seed_pilot(model_seed)))
    attach_cursor(workers, cursor)
    for worker in workers:
        worker.reset(cursor.next())
    eval_worker = ResidualWorker(high, actor, critic, service_lock=True)
    history = []
    curve = []
    env_steps = 0
    ckpts = []
    for update in range(0, PILOT_UPDATES_N + 1):
        if update in {0, 20, 40, 60, 80, 100} or update % EVAL_EVERY == 0:
            metrics = evaluate_validation(eval_worker, curve_seeds, zero_residual=False, diagnose=False)
            row = {
                "model_seed": int(model_seed),
                "update": update,
                "payload": metrics["payload"],
                "completed_blocks": metrics["completed_blocks"],
                "mean_residual_norm": metrics["mean_residual_norm"],
                "projector_intervention_rate": metrics["projector_intervention_rate"],
                "source": "llv2_train_curve16",
            }
            curve.append(row)
            path = OUTPUT / f"ckpt_seed{model_seed}_update{update}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(agent.state_dict(), path)
            ckpts.append({"model_seed": int(model_seed), "update": update, "path": str(path), "primary": update == 100})
        if update == PILOT_UPDATES_N:
            break
        storage = collect_rollout(workers, agent, int(PPO.rollout_length))
        stats = agent.update(storage)
        env_steps += int(PPO.rollout_length) * len(workers)
        history.append({"model_seed": int(model_seed), "update": update + 1, **stats, "env_steps": env_steps})
    if high_sha() != digest:
        raise RuntimeError("high SHA changed")
    final = OUTPUT / f"ckpt_seed{model_seed}_update100.pt"
    return {
        "model_seed": int(model_seed),
        "env_steps": env_steps,
        "smoke_env_steps": env_smoke,
        "wall_s": time.time() - started,
        "history": history,
        "curve": curve,
        "ckpts": ckpts,
        "final": str(final),
        "high_sha": digest,
        "retrained": True,
    }
