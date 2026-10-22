"""Train Phase-3 new seeds 2/3/4. Frozen Phase-2D algorithm. Checkpoints under phase3/outputs."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import ResidualWorker, load_frozen_high, make_networks
from wt.lowlevel_v2.phase1b.run_phase1b import SeedCursor, attach_cursor, collect_rollout
from wt.lowlevel_v2.phase2b.eval import evaluate_design
from wt.lowlevel_v2.phase2b.train import make_worker as make_p2b_worker
from wt.lowlevel_v2.phase2d.actor_opt import ICARMAPPOActiveAware
from wt.lowlevel_v2.phase3.protocol import (
    EVAL_EVERY,
    EXPECTED_HIGH_SHA,
    OUTPUT,
    PILOT_UPDATES_N,
    PPO,
    SMOKE_UPDATES_N,
    high_sha,
    refuse_historical,
    vector_seed_pilot,
    vector_seed_smoke,
)
from wt.round30.ppo import seed_everything


class Phase3Worker(ResidualWorker):
    def reset(self, seed: int):
        refuse_historical(int(seed), allow_phase3_val=False)
        return super().reset(seed)


def make_worker(high, actor, critic) -> Phase3Worker:
    return Phase3Worker(high, actor, critic, service_lock=True, trust_region=True)


def train_seed(model_seed: int, train_seeds: list[int], design_seeds: list[int], *, output_dir: Path | None = None) -> dict:
    out = Path(output_dir) if output_dir is not None else OUTPUT
    started = time.time()
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("high SHA mismatch")
    for s in list(train_seeds) + list(design_seeds):
        refuse_historical(int(s), allow_phase3_val=False)
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
    eval_worker = make_p2b_worker(high, actor, critic)
    history = []
    curve = []
    env_steps = 0
    out.mkdir(parents=True, exist_ok=True)
    for update in range(0, PILOT_UPDATES_N + 1):
        if update in {0, 20, 40, 60, 80, 100} or update % EVAL_EVERY == 0:
            metrics = evaluate_design(eval_worker, design_seeds, zero_residual=False)
            curve.append(
                {
                    "model_seed": int(model_seed),
                    "update": update,
                    "payload": metrics["payload"],
                    "completed_blocks": metrics["completed_blocks"],
                    "flight_distance": metrics["flight_distance"],
                    "arrival_steps": metrics["mean_arrival_steps"],
                    "projector_intervention_rate": metrics["projector_intervention_rate"],
                    "raw_residual_norm": metrics["mean_raw_residual"],
                    "used_residual_norm": metrics["mean_used_residual"],
                    "cap_activation_rate": metrics["cap_activation_rate"],
                    "base_zero_freeze_rate": metrics["base_zero_freeze_rate"],
                    "active_fraction_eval": 1.0 - metrics["base_zero_freeze_rate"],
                }
            )
            torch.save(agent.state_dict(), out / f"ckpt_seed{model_seed}_update{update}.pt")
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
        "final": str(out / f"ckpt_seed{model_seed}_update100.pt"),
        "high_sha": digest,
        "reused_phase2d": False,
        "loaded_phase2b_weights": False,
    }


__all__ = ["Phase3Worker", "make_worker", "train_seed"]
