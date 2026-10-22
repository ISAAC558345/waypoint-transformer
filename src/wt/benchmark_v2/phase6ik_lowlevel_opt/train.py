"""Warm-start Phase-6G seed 0 and train on scaled observations. The source file is only read."""
from __future__ import annotations

import hashlib
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
from .scale import fold_scales_into_actor
from .worker import ScaledTrainWorker


def source_sha256() -> str:
    return hashlib.sha256(Path(SOURCE_ACTOR).read_bytes()).hexdigest()


def load_warm_actor() -> ResidualActor:
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    payload = torch.load(SOURCE_ACTOR, map_location="cpu", weights_only=False)
    state = payload["actor"] if isinstance(payload, dict) and "actor" in payload else payload
    actor.load_state_dict(state, strict=True)
    if tuple(actor.mu.weight.shape) != (2, 64):
        raise RuntimeError("warm-start actor is not the 2-D residual head")
    return actor


def _collect(workers: list[ScaledTrainWorker], horizon: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
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
    served, sinr, interference, path, residual, projected, safety, r_comm, payload = [], [], [], [], [], [], [], [], []
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
            served.append(rec["served_bits"])
            sinr.append(rec["sinr"])
            interference.append(rec["interference"])
            path.append(rec["path_step"])
            residual.append(rec["residual_norm"])
            projected.append(rec["projected"])
            safety.append(rec["safety_violation"])
            r_comm.append(rec["r_comm"])
            payload.append(rec["payload"])
    last = np.zeros(n_env, dtype=np.float32)
    for index, worker in enumerate(workers):
        if storage["dones"][-1, index] > 0.5:
            continue
        last[index] = float(worker.bootstrap_value())
    storage["last_values"] = last
    pre = storage["pre_tanh"]
    metrics = {
        "mean_reward": float(np.mean(storage["rewards"])),
        "r_comm": float(np.mean(r_comm)),
        "mean_served_bits": float(np.mean(served)),
        "mean_episode_payload_bits": float(np.mean(payload)),
        "mean_sinr": float(np.mean(sinr)),
        "mean_interference": float(np.mean(interference)),
        "path_length": float(np.sum(path) / n_env),
        "residual_action_norm": float(np.mean(residual)),
        "projection_rate": float(np.mean(projected)),
        "safety_violation_rate": float(np.mean(safety)),
        "pre_tanh_abs": float(np.mean(np.abs(pre))),
        "tanh_saturation": float(np.mean(np.abs(np.tanh(pre)) > 0.9)),
        "nan_inf": int(not np.isfinite(storage["rewards"]).all()),
    }
    return storage, metrics


def train_seed(model_seed: int, updates: int, checkpoint_path: Path) -> dict:
    destination = assert_phase_output(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source_sha256()
    torch.set_num_threads(1)
    seed_everything(int(model_seed))
    actor = load_warm_actor()
    critic = CentralizedCritic(expected_critic_dim(6, 12))
    agent = ICARMAPPOLocked(actor, critic, PPO)
    workers = []
    for index in range(int(PPO.num_envs)):
        worker = ScaledTrainWorker(int(model_seed), index, actor, critic)
        worker.reset()
        workers.append(worker)
    history = []
    for update in range(int(updates)):
        storage, metrics = _collect(workers, int(PPO.rollout_length))
        if metrics["nan_inf"]:
            raise RuntimeError("NONFINITE")
        stats = agent.update(storage)
        row = {
            "model_seed": int(model_seed),
            "update": update + 1,
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
        if (update + 1) % 5 == 0 or update == 0:
            print(
                f"seed {model_seed} update {update + 1}/{updates} reward={row['mean_reward']:.4f} served={row['mean_served_bits']:.1f} proj={row['projection_rate']:.3f} grad={row['gradient_norm_preclip']:.3f}",
                flush=True,
            )
    if source_sha256() != before:
        raise RuntimeError("source MAPPO checkpoint changed during training")
    actor.eval()
    folded = fold_scales_into_actor(actor)
    folded.eval()
    torch.save(
        {
            "actor": folded.state_dict(),
            "critic_normalized_input": critic.state_dict(),
            "model_seed": int(model_seed),
            "updates": int(updates),
            "initialization": "phase6g_seed0_scaled_inputs",
            "source_sha256": before,
            "observation_rank": "raw_environment_after_column_fold",
            "reward": "phase6g low_level_reward",
            "network": "ResidualActor 2-D residual",
        },
        destination,
    )
    history_path = destination.with_suffix(".history.json")
    history_path.write_text(json.dumps(history), encoding="utf-8")
    return {"model_seed": int(model_seed), "checkpoint": str(destination), "history": history, "source_sha256": before}


__all__ = ["load_warm_actor", "source_sha256", "train_seed"]
