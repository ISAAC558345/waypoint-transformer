"""Train Phase-4A seed0/1 from scratch. Collision distillation only new term."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import load_frozen_high, make_networks
from wt.lowlevel_v2.phase1b.run_phase1b import SeedCursor, attach_cursor
from wt.lowlevel_v2.phase2.protocol import vector_seed_pilot, vector_seed_smoke
from wt.lowlevel_v2.phase4a.actor_opt import ICARMAPPOCollisionDistill
from wt.lowlevel_v2.phase4a.eval import evaluate_design_mechanics
from wt.lowlevel_v2.phase4a.protocol import (
    EVAL_EVERY,
    EXPECTED_HIGH_SHA,
    LAMBDA_COLLISION_DISTILL,
    OUTPUT,
    PILOT_UPDATES_N,
    PPO,
    SMOKE_UPDATES_N,
    high_sha,
    refuse_phase4a,
)
from wt.lowlevel_v2.phase4a.rollout import collect_rollout
from wt.lowlevel_v2.phase4a.worker import make_worker
from wt.round30.ppo import seed_everything


def train_seed(model_seed: int, train_seeds: list[int], design_seeds: list[int], *, output_dir: Path | None = None) -> dict:
    out = Path(output_dir) if output_dir is not None else OUTPUT
    started = time.time()
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("high SHA mismatch")
    for s in list(train_seeds) + list(design_seeds):
        refuse_phase4a(int(s))
    seed_everything(int(model_seed))
    high, digest = load_frozen_high()
    actor, critic = make_networks()
    agent = ICARMAPPOCollisionDistill(actor, critic, PPO, lambda_distill=LAMBDA_COLLISION_DISTILL)
    init_state = copy.deepcopy(agent.state_dict())
    if init_state.get("loaded_phase3_weights"):
        raise RuntimeError("must not load Phase-3 weights")
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
    agent = ICARMAPPOCollisionDistill(actor, critic, PPO, lambda_distill=LAMBDA_COLLISION_DISTILL)
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
    out.mkdir(parents=True, exist_ok=True)
    for update in range(0, PILOT_UPDATES_N + 1):
        if update in {0, 20, 40, 60, 80, 100} or update % EVAL_EVERY == 0:
            metrics = evaluate_design_mechanics(eval_worker, design_seeds, zero_residual=False)
            curve.append(
                {
                    "model_seed": int(model_seed),
                    "update": update,
                    "payload": metrics["payload"],
                    "completed_blocks": metrics["completed_blocks"],
                    "time_in_range": metrics["time_in_range"],
                    "projector_intervention_rate": metrics["projector_intervention_rate"],
                    "pairwise_intervention_rate": metrics["pairwise_intervention_rate"],
                    "service_only_intervention_rate": metrics["service_only_intervention_rate"],
                    "multi_intervention_rate": metrics["multi_intervention_rate"],
                    "teacher_mse": float("nan"),
                    "teacher_cosine": float("nan"),
                    "sep_mean": metrics["separation"]["mean"],
                    "sep_positive_fraction": metrics["separation"]["positive_fraction"],
                }
            )
            torch.save(agent.state_dict(), out / f"ckpt_c1_seed{model_seed}_update{update}.pt")
        if update == PILOT_UPDATES_N:
            break
        storage = collect_rollout(workers, agent, int(PPO.rollout_length))
        stats = agent.update(storage)
        env_steps += int(PPO.rollout_length) * len(workers)
        history.append({"model_seed": int(model_seed), "update": update + 1, **stats, "env_steps": env_steps})
        if update + 1 in {20, 40, 60, 80, 100}:
            for row in curve:
                if int(row["update"]) == update + 1 and int(row["model_seed"]) == int(model_seed):
                    row["teacher_mse"] = stats.get("teacher_mse")
                    row["teacher_cosine"] = stats.get("teacher_cosine")
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
        "final": str(out / f"ckpt_c1_seed{model_seed}_update100.pt"),
        "high_sha": digest,
        "loaded_phase3_weights": False,
        "lambda_collision_distill": LAMBDA_COLLISION_DISTILL,
    }


__all__ = ["train_seed"]
