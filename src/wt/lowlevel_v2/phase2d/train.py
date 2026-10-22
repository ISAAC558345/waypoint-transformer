"""Train Phase-2D seed0 and seed1 from scratch. Active-aware actor only."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import load_frozen_high, make_networks
from wt.lowlevel_v2.phase1b.run_phase1b import SeedCursor, attach_cursor, collect_rollout
from wt.lowlevel_v2.phase2b.eval import evaluate_design
from wt.lowlevel_v2.phase2b.train import make_worker
from wt.lowlevel_v2.phase2d.actor_opt import ICARMAPPOActiveAware
from wt.lowlevel_v2.phase2d.protocol import (
    EVAL_EVERY,
    EXPECTED_HIGH_SHA,
    OUTPUT,
    PILOT_UPDATES_N,
    PPO,
    SMOKE_UPDATES_N,
    high_sha,
    refuse_phase2_validation,
    vector_seed_pilot,
    vector_seed_smoke,
)
from wt.round30.ppo import seed_everything


def train_seed(model_seed: int, train_seeds: list[int], design_seeds: list[int]) -> dict:
    started = time.time()
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("high SHA mismatch")
    for s in train_seeds + design_seeds:
        refuse_phase2_validation(int(s))
    seed_everything(int(model_seed))
    high, digest = load_frozen_high()
    actor, critic = make_networks()
    agent = ICARMAPPOActiveAware(actor, critic, PPO)
    init_state = copy.deepcopy(agent.state_dict())
    workers = [make_worker(high, actor, critic) for _ in range(int(PPO.num_envs))]
    smoke_cursor = SeedCursor(train_seeds, np.random.default_rng(vector_seed_smoke(model_seed)))
    attach_cursor(workers, smoke_cursor)
    for worker in workers:
        worker.reset(smoke_cursor.next())
    for _ in range(int(SMOKE_UPDATES_N)):
        storage = collect_rollout(workers, agent, int(PPO.rollout_length))
        stats = agent.update(storage)
        if not np.isfinite(stats["policy_loss"]):
            raise RuntimeError("smoke NaN")
    agent.load_state_dict(init_state)
    seed_everything(int(model_seed))
    actor.load_state_dict(init_state["actor"])
    critic.load_state_dict(init_state["critic"])
    agent = ICARMAPPOActiveAware(actor, critic, PPO)
    agent.load_state_dict(init_state)
    workers = [make_worker(high, actor, critic) for _ in range(int(PPO.num_envs))]
    cursor = SeedCursor(train_seeds, np.random.default_rng(vector_seed_pilot(model_seed)))
    attach_cursor(workers, cursor)
    for worker in workers:
        worker.reset(cursor.next())
    eval_worker = make_worker(high, actor, critic)
    history = []
    curve = []
    env_steps = 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for update in range(0, PILOT_UPDATES_N + 1):
        if update in {0, 20, 40, 60, 80, 100} or update % EVAL_EVERY == 0:
            metrics = evaluate_design(eval_worker, design_seeds, zero_residual=False)
            curve.append(
                {
                    "model_seed": int(model_seed),
                    "update": update,
                    "payload": metrics["payload"],
                    "completed_blocks": metrics["completed_blocks"],
                    "residual_queue": metrics["residual_queue"],
                    "drop": metrics["drop"],
                    "flight_distance": metrics["flight_distance"],
                    "projector_intervention_rate": metrics["projector_intervention_rate"],
                    "raw_residual_norm": metrics["mean_raw_residual"],
                    "used_residual_norm": metrics["mean_used_residual"],
                    "cap_activation_rate": metrics["cap_activation_rate"],
                    "base_zero_freeze_rate": metrics["base_zero_freeze_rate"],
                    "arrival_steps": metrics["mean_arrival_steps"],
                    "never_reach_count": metrics["never_reach_count"],
                    "never_reach_fraction": metrics["never_reach_fraction"],
                    "time_in_range": metrics["time_in_range"],
                    "idle_fraction": metrics["idle_fraction"],
                    "active_fraction_eval": 1.0 - metrics["base_zero_freeze_rate"],
                }
            )
            path = OUTPUT / f"ckpt_seed{model_seed}_update{update}.pt"
            torch.save(agent.state_dict(), path)
        if update == PILOT_UPDATES_N:
            break
        storage = collect_rollout(workers, agent, int(PPO.rollout_length))
        stats = agent.update(storage)
        env_steps += int(PPO.rollout_length) * len(workers)
        history.append({"model_seed": int(model_seed), "update": update + 1, **stats, "env_steps": env_steps})
        if stats.get("min_actor_minibatch_size", 1) <= 0 and stats.get("active_samples", 0) > 0:
            raise RuntimeError("zero-active actor minibatch during training")
    if high_sha() != digest:
        raise RuntimeError("high SHA changed")
    if env_steps != int(PPO.rollout_length) * int(PPO.num_envs) * int(PILOT_UPDATES_N):
        raise RuntimeError("budget mismatch")
    return {
        "model_seed": int(model_seed),
        "env_steps": env_steps,
        "agent_steps": env_steps * 6,
        "wall_s": time.time() - started,
        "history": history,
        "curve": curve,
        "final": str(OUTPUT / f"ckpt_seed{model_seed}_update100.pt"),
        "high_sha": digest,
        "loaded_old_weights": False,
        "loaded_phase2b_weights": False,
    }


__all__ = ["train_seed"]
