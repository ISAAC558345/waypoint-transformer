"""Warm-start the frozen actor and train on mixed live task sources."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT, PPO
from wt.lowlevel_v2.critic_state import expected_critic_dim
from wt.lowlevel_v2.phase1b.mappo import ICARMAPPOLocked
from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor
from wt.round30.ppo import seed_everything

from .protocol import SOURCE_ACTOR, assert_phase_output
from .worker import RobustTrainWorker


def load_source() -> dict:
    payload = torch.load(SOURCE_ACTOR, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "actor" not in payload:
        raise RuntimeError("phase6g checkpoint has no actor state")
    return payload


def load_warm_actor(payload: dict) -> ResidualActor:
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    actor.load_state_dict(payload["actor"], strict=True)
    for parameter in actor.parameters():
        parameter.requires_grad_(True)
    return actor


def load_critic(payload: dict) -> tuple[CentralizedCritic, str]:
    width = int(expected_critic_dim(6, 12))
    critic = CentralizedCritic(width)
    state = payload.get("critic")
    weight = None if not isinstance(state, dict) else state.get("net.0.weight")
    if weight is None or int(weight.shape[1]) != width:
        return critic, "reinitialized"
    critic.load_state_dict(state, strict=True)
    return critic, "reused_phase6g_critic"


def _collect(workers: list[RobustTrainWorker], horizon: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    n_env = len(workers)
    glob_dim = int(expected_critic_dim(6, 12))
    keys = {
        "self": (horizon, n_env, 6, 5),
        "task": (horizon, n_env, 6, 8),
        "communication": (horizon, n_env, 6, 4),
        "neighbor_set": (horizon, n_env, 6, 5, 11),
        "d_nearest": (horizon, n_env, 6),
        "global_state": (horizon, n_env, glob_dim),
        "pre_tanh": (horizon, n_env, 6, 2),
        "log_prob_uav": (horizon, n_env, 6),
        "policy_active_mask": (horizon, n_env, 6),
        "values": (horizon, n_env),
        "rewards": (horizon, n_env),
        "dones": (horizon, n_env),
    }
    storage = {key: np.zeros(shape, dtype=np.float32) for key, shape in keys.items()}
    served, reward, comm, projected, safety, boundary_open = [], [], [], [], [], []
    for _t in range(horizon):
        for index, worker in enumerate(workers):
            rec = worker.step()
            for key in ("self", "task", "communication", "neighbor_set", "d_nearest"):
                storage[key][_t, index] = rec["obs"][key]
            storage["global_state"][_t, index] = rec["global_state"]
            storage["pre_tanh"][_t, index] = rec["pre_tanh"]
            storage["log_prob_uav"][_t, index] = rec["log_prob_uav"]
            storage["policy_active_mask"][_t, index] = rec["policy_active_mask"]
            storage["values"][_t, index] = rec["value"]
            storage["rewards"][_t, index] = rec["reward"]
            storage["dones"][_t, index] = float(rec["done"])
            served.append(rec["served_bits"])
            reward.append(rec["reward"])
            comm.append(rec["r_comm"])
            projected.append(rec["projected"])
            safety.append(rec["safety_violation"])
            if rec["task_boundary"] and not rec["done"]:
                boundary_open.append(1.0)
    last = np.zeros(n_env, dtype=np.float32)
    for index, worker in enumerate(workers):
        if storage["dones"][-1, index] <= 0.5:
            last[index] = float(worker.bootstrap_value())
    storage["last_values"] = last
    metrics = {
        "mean_reward": float(np.mean(reward)),
        "mean_r_comm": float(np.mean(comm)),
        "mean_served_bits": float(np.mean(served)),
        "projection_rate": float(np.mean(projected)),
        "safety_violation_rate": float(np.mean(safety)),
        "open_task_boundaries": int(len(boundary_open)),
        "nan_inf": int(not np.isfinite(storage["rewards"]).all()),
    }
    return storage, metrics


def train_seed(model_seed: int, updates: int, checkpoint_path: Path, payload: dict, selector) -> dict:
    destination = assert_phase_output(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    seed_everything(int(model_seed))
    actor = load_warm_actor(payload)
    actor.train()
    critic, critic_mode = load_critic(payload)
    agent = ICARMAPPOLocked(actor, critic, PPO)
    workers = []
    for index in range(int(PPO.num_envs)):
        worker = RobustTrainWorker(int(model_seed), index, actor, critic, selector, int(PPO.num_envs))
        worker.reset()
        workers.append(worker)
    history = []
    for update in range(int(updates)):
        storage, metrics = _collect(workers, int(PPO.rollout_length))
        if metrics["nan_inf"]:
            raise RuntimeError("NONFINITE")
        if int(metrics["open_task_boundaries"]) < 1:
            raise RuntimeError("rollout contained no open task boundary")
        stats = agent.update(storage)
        row = {
            "model_seed": int(model_seed),
            "update": update + 1,
            "actor_loss": stats["policy_loss"],
            "critic_loss": stats["value_loss"],
            "entropy": stats["entropy"],
            "gradient_norm_preclip": stats["gradient_norm_preclip"],
            "explained_variance": stats["explained_variance"],
            "critic_mode": critic_mode,
            **metrics,
        }
        history.append(row)
        if (update + 1) % 10 == 0 or update == 0:
            print(
                f"robust seed {model_seed} update {update + 1}/{updates} reward={row['mean_reward']:.4f} served={row['mean_served_bits']:.1f}",
                flush=True,
            )
    actor.eval()
    decisions = [item for worker in workers for item in worker.decisions]
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "model_seed": int(model_seed),
            "updates": int(updates),
            "critic_mode": critic_mode,
            "action": "residual_alpha_0.25",
            "reward": "phase6g low_level_reward",
            "task_source": list(selector.config["train_sources"]),
            "held_out_source": selector.config["held_out_source"],
            "shaping": False,
            "observation_changed": False,
            "official_config": False,
        },
        destination,
    )
    destination.with_suffix(".history.json").write_text(json.dumps(history), encoding="utf-8")
    return {"model_seed": int(model_seed), "checkpoint": str(destination), "history": history, "critic_mode": critic_mode, "decisions": decisions}


__all__ = ["load_critic", "load_source", "load_warm_actor", "train_seed"]
