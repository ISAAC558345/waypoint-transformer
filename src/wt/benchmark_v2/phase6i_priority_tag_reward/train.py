"""Train L1 and L2 with the locked MAPPO budget. L0 is not retrained."""
from __future__ import annotations

import json
import subprocess
import time

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT, PPO
from wt.lowlevel_v2.residual_policy import CentralizedCritic
from wt.round30.ppo import seed_everything

from .actor import PriorityTagActor, load_formal_actor, transplant_legacy_actor
from .mappo import ExperimentMAPPO
from .protocol import (
    FEATURE_NAMES,
    L1_LAMBDA_U,
    LAMBDA_U,
    LEGACY_ACTOR,
    OBSERVATION_SCHEMA,
    OUTPUT,
    REWARD_SCHEMA,
    RHO,
    ROOT,
    SCHEMA_VERSION,
    TAG_RATE,
    TAG_SEED,
    TRAIN_SEEDS,
    UPDATES,
)
from .rollout import ExperimentWorker, critic_dim, variant_lambda
from .smoke import file_sha256


def source_commit() -> str:
    result = subprocess.run(
        ["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def _collect(workers: list[ExperimentWorker], horizon: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    n_env = len(workers)
    n_uav = int(workers[0].env.config.num_uavs)
    keys = {
        "self": (horizon, n_env, n_uav, 5),
        "task": (horizon, n_env, n_uav, 8),
        "communication": (horizon, n_env, n_uav, 4),
        "neighbor_set": (horizon, n_env, n_uav, 5, 11),
        "d_nearest": (horizon, n_env, n_uav),
        "priority": (horizon, n_env, n_uav, len(FEATURE_NAMES)),
        "global_state": (horizon, n_env, critic_dim(n_uav, int(workers[0].env.config.num_buoys))),
        "pre_tanh": (horizon, n_env, n_uav, 2),
        "log_prob_uav": (horizon, n_env, n_uav),
        "policy_active_mask": (horizon, n_env, n_uav),
        "values": (horizon, n_env),
        "rewards": (horizon, n_env),
        "dones": (horizon, n_env),
    }
    storage = {key: np.zeros(shape, dtype=np.float32) for key, shape in keys.items()}
    served, bonus, projected, safety, failures = [], [], [], [], []
    for t in range(horizon):
        for index, worker in enumerate(workers):
            record = worker.step()
            storage["self"][t, index] = record["obs"]["self"]
            storage["task"][t, index] = record["obs"]["task"]
            storage["communication"][t, index] = record["obs"]["communication"]
            storage["neighbor_set"][t, index] = record["obs"]["neighbor_set"]
            storage["d_nearest"][t, index] = record["obs"]["d_nearest"]
            storage["priority"][t, index] = record["priority"]
            storage["global_state"][t, index] = record["global_state"]
            storage["pre_tanh"][t, index] = record["pre_tanh"]
            storage["log_prob_uav"][t, index] = record["log_prob_uav"]
            storage["policy_active_mask"][t, index] = record["policy_active_mask"]
            storage["values"][t, index] = record["value"]
            storage["rewards"][t, index] = record["reward"]
            storage["dones"][t, index] = float(record["done"])
            served.append(record["served_bits"])
            bonus.append(record["bonus"])
            projected.append(record["projected"])
            safety.append(record["safety_violation"])
            failures.append(record["constraint_failure"])
    last = np.zeros(n_env, dtype=np.float32)
    for index, worker in enumerate(workers):
        if storage["dones"][-1, index] > 0.5 or worker.option is None or worker.critic is None:
            continue
        from .rollout import critic_vector

        with torch.no_grad():
            vector = critic_vector(worker.env, worker.option, worker.last_movement, worker._priority())
            last[index] = float(worker.critic(torch.as_tensor(vector, dtype=torch.float32)).cpu())
    storage["last_values"] = last
    if not np.isfinite(storage["rewards"]).all():
        raise RuntimeError("NONFINITE")
    metrics = {
        "mean_served_bits": float(np.mean(served)),
        "mean_bonus": float(np.mean(bonus)),
        "projection_rate": float(np.mean(projected)),
        "safety_violation_rate": float(np.mean(safety)),
        "constraint_failure_rate": float(np.mean(failures)),
    }
    return storage, metrics


def _checkpoint_payload(actor, critic, model_seed: int, variant: str, lambda_u: float) -> dict:
    return {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "model_seed": int(model_seed),
        "updates": int(UPDATES),
        "variant": variant,
        "initialization": "LEGACY_WARM_START_EXPLICIT_ZERO_NEW_COLUMNS",
        "observation_schema": OBSERVATION_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "reward_schema": REWARD_SCHEMA,
        "reward_config": {
            "lambda_u": float(lambda_u),
            "rho": float(RHO),
            "formula": "r_base + min(lambda_u * urgency_served_bits/1e6, rho * served_bits/1e6)",
        },
        "synthetic_tag_protocol": SCHEMA_VERSION,
        "synthetic_tag": True,
        "tag_seed": int(TAG_SEED),
        "tag_rate": float(TAG_RATE),
        "source_commit": source_commit(),
        "legacy_actor_sha256": file_sha256(LEGACY_ACTOR),
        "ppo": {
            "updates": int(UPDATES),
            "rollout_length": int(PPO.rollout_length),
            "num_envs": int(PPO.num_envs),
            "num_minibatches": int(PPO.num_minibatches),
            "update_epochs": int(PPO.update_epochs),
            "learning_rate": float(PPO.learning_rate),
            "gamma": float(PPO.gamma),
            "gae_lambda": float(PPO.gae_lambda),
            "clip_eps": float(PPO.clip_eps),
            "value_coef": float(PPO.value_coef),
            "entropy_coef": float(PPO.entropy_coef),
            "max_grad_norm": float(PPO.max_grad_norm),
        },
    }


def train_variant(variant: str, model_seed: int) -> dict:
    if int(model_seed) not in TRAIN_SEEDS:
        raise RuntimeError(f"unexpected training seed {model_seed}")
    if variant not in {"L1", "L2"}:
        raise RuntimeError(f"L0 is frozen and is not trained here: {variant}")
    lambda_u = variant_lambda(variant)
    if variant == "L1" and float(lambda_u) != float(L1_LAMBDA_U):
        raise RuntimeError("L1 must keep the formal reward")
    if variant == "L2" and float(lambda_u) != float(LAMBDA_U):
        raise RuntimeError("L2 lambda drifted from the preregistered candidate")
    torch.set_num_threads(1)
    seed_everything(int(model_seed))
    actor = PriorityTagActor(log_std_init=float(LOG_STD_INIT))
    transplant_legacy_actor(load_formal_actor(LEGACY_ACTOR, LOG_STD_INIT), actor)
    critic = CentralizedCritic(critic_dim())
    agent = ExperimentMAPPO(actor, critic, PPO)
    workers = []
    for index in range(int(PPO.num_envs)):
        worker = ExperimentWorker(
            int(model_seed),
            index,
            actor,
            critic,
            lambda_u=float(lambda_u),
            rho=float(RHO),
            use_ledger=True,
            deterministic=False,
        )
        worker.reset()
        workers.append(worker)
    history = []
    started = time.perf_counter()
    for update in range(int(UPDATES)):
        storage, metrics = _collect(workers, int(PPO.rollout_length))
        stats = agent.update(storage)
        row = {"model_seed": int(model_seed), "variant": variant, "update": update + 1, **stats, **metrics}
        history.append(row)
        if (update + 1) % 10 == 0 or update == 0:
            print(
                f"{variant} seed {model_seed} update {update + 1}/{UPDATES} "
                f"reward={float(np.mean(storage['rewards'])):.4f} served={metrics['mean_served_bits']:.1f}",
                flush=True,
            )
    destination = OUTPUT / "checkpoints" / f"{variant.lower()}_seed{int(model_seed)}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(actor, critic, int(model_seed), variant, float(lambda_u))
    torch.save(payload, destination)
    digest = file_sha256(destination)
    manifest = {
        "path": str(destination.relative_to(ROOT)).replace("\\", "/"),
        "sha256": digest,
        "variant": variant,
        "model_seed": int(model_seed),
        "updates": int(UPDATES),
        "elapsed_seconds": time.perf_counter() - started,
        "final_policy_loss": history[-1]["policy_loss"],
        "final_value_loss": history[-1]["value_loss"],
        "final_gradient_norm": history[-1]["gradient_norm_preclip"],
        "mean_safety_violation_rate": float(np.mean([row["safety_violation_rate"] for row in history])),
        "mean_constraint_failure_rate": float(np.mean([row["constraint_failure_rate"] for row in history])),
    }
    history_path = OUTPUT / "logs" / f"{variant.lower()}_seed{int(model_seed)}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history), encoding="utf-8")
    return manifest
