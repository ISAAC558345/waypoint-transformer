"""Independent MAPPO loop. Shapes are 10 and 200; the formal trainer is not called."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT, PPO
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6i_urgency_stage_a.runner import ScoreTap, limit_threads
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.lowlevel_v2.phase1b.mappo import ICARMAPPOLocked
from wt.lowlevel_v2.residual_policy import CentralizedCritic
from wt.round30.ppo import PPOConfig, seed_everything

from .models import UrgencyActor, save_checkpoint, warm_start_from_frozen
from .protocol import ACTOR_IN_DIM, CHECKPOINTS, CRITIC_IN_DIM, LAMBDA_URGENCY, MODEL_SEEDS_STAGE_C, UPDATES
from .worker import SchemeCWorker


def train_variant(
    variant: str,
    model_seed: int,
    *,
    updates: int | None = None,
    ppo: PPOConfig | None = None,
    checkpoint_path: Path | None = None,
    history_path: Path | None = None,
) -> dict:
    if variant not in ("L1", "L2"):
        raise RuntimeError(f"L0 is the frozen baseline and is not trained here: {variant}")
    if int(model_seed) not in MODEL_SEEDS_STAGE_C:
        raise RuntimeError(f"seed {model_seed} is outside the pre-registered stage C seeds")
    budget = PPO if ppo is None else ppo
    total = int(UPDATES if updates is None else updates)
    if ppo is None and total != int(UPDATES):
        raise RuntimeError("formal training budget is 300 updates")
    limit_threads()
    started = time.perf_counter()
    seed_everything(int(model_seed))
    actor = UrgencyActor(log_std_init=float(LOG_STD_INIT))
    critic = CentralizedCritic(int(CRITIC_IN_DIM))
    layer_report = warm_start_from_frozen(actor, critic, primary_checkpoint())
    agent = ICARMAPPOLocked(actor, critic, budget)
    planner = FormalPlanner("scheme_c")
    tap = ScoreTap(planner)
    workers = []
    for index in range(int(budget.num_envs)):
        worker = SchemeCWorker(int(model_seed), index, actor, critic, planner, tap, variant)
        worker.reset()
        workers.append(worker)
    destination = checkpoint_path or (CHECKPOINTS / f"{variant}_seed{int(model_seed)}.pt")
    history_file = history_path or destination.with_suffix(".jsonl")
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history_file.write_text("", encoding="utf-8")
    history = []
    for update in range(total):
        storage, metrics = _collect(workers, int(budget.rollout_length))
        if not np.isfinite(storage["rewards"]).all():
            raise RuntimeError("NONFINITE")
        stats = agent.update(storage)
        row = {
            "variant": variant,
            "model_seed": int(model_seed),
            "update": update + 1,
            "lambda_urgency": 0.0 if variant == "L1" else float(LAMBDA_URGENCY),
            "actor_loss": stats["policy_loss"],
            "critic_loss": stats["value_loss"],
            "entropy": stats["entropy"],
            "kl": stats["approx_kl"],
            "clip_fraction": stats["clip_fraction"],
            "explained_variance": stats["explained_variance"],
            "gradient_norm_preclip": stats["gradient_norm_preclip"],
            **metrics,
        }
        history.append(row)
        with history_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if update == 0 or (update + 1) % 10 == 0:
            print(
                f"{variant} seed {model_seed} update {update + 1}/{total} "
                f"reward={float(np.mean(storage['rewards'])):.4f} idle={metrics['mean_idle_slots']:.3f}",
                flush=True,
            )
    actor.eval()
    critic.eval()
    elapsed = time.perf_counter() - started
    save_checkpoint(
        destination,
        actor,
        critic,
        {
            "variant": variant,
            "model_seed": int(model_seed),
            "updates": int(total),
            "lambda_urgency": 0.0 if variant == "L1" else float(LAMBDA_URGENCY),
            "initialization": "phase6g_seed0_zero_new_columns",
            "layer_load_report": layer_report,
            "ppo": budget.to_dict(),
            "formal_updates": int(UPDATES),
        },
    )
    print(f"{variant} seed {model_seed} finished {total} updates in {elapsed:.1f}s", flush=True)
    return {
        "variant": variant,
        "model_seed": int(model_seed),
        "updates": int(total),
        "seconds": float(elapsed),
        "history": history,
        "checkpoint": str(destination),
    }


def _collect(workers: list[SchemeCWorker], horizon: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    n_env = len(workers)
    storage = {
        "self": np.zeros((horizon, n_env, 6, 5), dtype=np.float32),
        "task": np.zeros((horizon, n_env, 6, 10), dtype=np.float32),
        "communication": np.zeros((horizon, n_env, 6, 4), dtype=np.float32),
        "neighbor_set": np.zeros((horizon, n_env, 6, 5, 11), dtype=np.float32),
        "d_nearest": np.zeros((horizon, n_env, 6), dtype=np.float32),
        "global_state": np.zeros((horizon, n_env, int(CRITIC_IN_DIM)), dtype=np.float32),
        "pre_tanh": np.zeros((horizon, n_env, 6, 2), dtype=np.float32),
        "log_prob_uav": np.zeros((horizon, n_env, 6), dtype=np.float32),
        "policy_active_mask": np.zeros((horizon, n_env, 6), dtype=np.float32),
        "values": np.zeros((horizon, n_env), dtype=np.float32),
        "rewards": np.zeros((horizon, n_env), dtype=np.float32),
        "dones": np.zeros((horizon, n_env), dtype=np.float32),
    }
    if int(ACTOR_IN_DIM) != 36:
        raise RuntimeError("actor width is not 36")
    communication = []
    urgency = []
    served = []
    projected = []
    safety = []
    idle = []
    calls = []
    counts = []
    payload = []
    for t in range(horizon):
        for index, worker in enumerate(workers):
            rec = worker.step()
            storage["self"][t, index] = rec["obs"]["self"]
            storage["task"][t, index] = rec["obs"]["task"]
            storage["communication"][t, index] = rec["obs"]["communication"]
            storage["neighbor_set"][t, index] = rec["obs"]["neighbor_set"]
            storage["d_nearest"][t, index] = rec["obs"]["d_nearest"]
            storage["global_state"][t, index] = rec["global_state"]
            storage["pre_tanh"][t, index] = rec["pre_tanh"]
            storage["log_prob_uav"][t, index] = rec["log_prob_uav"]
            storage["policy_active_mask"][t, index] = rec["policy_active_mask"]
            storage["values"][t, index] = rec["value"]
            storage["rewards"][t, index] = rec["reward"]
            storage["dones"][t, index] = float(rec["done"])
            communication.append(rec["communication_reward"])
            urgency.append(rec["urgency_reward"])
            served.append(rec["served_bits"])
            projected.append(rec["projected"])
            safety.append(rec["safety_violation"])
            payload.append(rec["payload"])
            if rec["decision"] is not None:
                idle.append(float(rec["idle_slots"]))
                calls.append(1.0)
                counts.append(float(rec["decision"]["candidate_count"]))
    last = np.zeros(n_env, dtype=np.float32)
    for index, worker in enumerate(workers):
        if storage["dones"][-1, index] > 0.5 or worker.option is None:
            continue
        with torch.no_grad():
            view_state = build_bootstrap(worker)
            last[index] = float(worker.critic(torch.as_tensor(view_state, dtype=torch.float32)).cpu())
    storage["last_values"] = last
    metrics = {
        "mean_reward": float(np.mean(storage["rewards"])),
        "communication_reward": float(np.mean(communication)),
        "urgency_reward": float(np.mean(urgency)),
        "mean_served_bits": float(np.mean(served)),
        "projection_rate": float(np.mean(projected)),
        "safety_violation_rate": float(np.mean(safety)),
        "mean_idle_slots": float(np.mean(idle)) if idle else 0.0,
        "scheme_c_calls": int(sum(calls)),
        "mean_candidate_count": float(np.mean(counts)) if counts else 0.0,
        "mean_step_payload_bits": float(np.mean(payload)),
    }
    return storage, metrics


def build_bootstrap(worker: SchemeCWorker) -> np.ndarray:
    from .observation import build_urgency_critic_state

    queue = np.asarray(worker.env.q_bits, dtype=np.float64)
    view = worker.ledger.view(int(worker.env.t), queue, float(worker.env.tol_bits))
    glob = build_urgency_critic_state(worker.env, worker.option, view, worker.last_movement)
    return np.asarray(glob["vector"], dtype=np.float32)
