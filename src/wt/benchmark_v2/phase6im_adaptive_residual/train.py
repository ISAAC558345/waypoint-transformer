"""Warm-start the frozen actor and train the adaptive residual map."""
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
from .worker import AdaptiveTrainWorker


def source_sha256() -> str:
    return hashlib.sha256(Path(SOURCE_ACTOR).read_bytes()).hexdigest()


def load_source() -> tuple[dict, str]:
    payload = torch.load(SOURCE_ACTOR, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "actor" not in payload:
        raise RuntimeError("phase6g checkpoint has no actor state")
    return payload, source_sha256()


def load_warm_actor(payload: dict) -> ResidualActor:
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    actor.load_state_dict(payload["actor"], strict=True)
    if tuple(actor.mu.weight.shape) != (2, 64):
        raise RuntimeError("adaptive actor is not the 2-D residual head")
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


def _collect(workers: list[AdaptiveTrainWorker], horizon: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
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
    served, reward, projected, safety = [], [], [], []
    turn, executed_turn, correction = [], [], []
    excessive_heading, excessive_speed, collision, envelope = [], [], [], []
    for t in range(horizon):
        for index, worker in enumerate(workers):
            rec = worker.step()
            for key in ("self", "task", "communication", "neighbor_set", "d_nearest"):
                storage[key][t, index] = rec["obs"][key]
            storage["global_state"][t, index] = rec["global_state"]
            storage["pre_tanh"][t, index] = rec["pre_tanh"]
            storage["log_prob_uav"][t, index] = rec["log_prob_uav"]
            storage["policy_active_mask"][t, index] = rec["policy_active_mask"]
            storage["values"][t, index] = rec["value"]
            storage["rewards"][t, index] = rec["reward"]
            storage["dones"][t, index] = float(rec["done"])
            served.append(rec["served_bits"])
            reward.append(rec["reward"])
            projected.append(rec["projected"])
            safety.append(rec["safety_violation"])
            turn.append(rec["command_turn_deg"])
            executed_turn.append(rec["executed_turn_deg"])
            correction.append(rec["correction_norm"])
            excessive_heading.append(rec["excessive_heading"])
            excessive_speed.append(rec["excessive_speed"])
            collision.append(rec["collision_risk"])
            envelope.append(rec["envelope_risk"])
    last = np.zeros(n_env, dtype=np.float32)
    for index, worker in enumerate(workers):
        if storage["dones"][-1, index] <= 0.5:
            last[index] = float(worker.bootstrap_value())
    storage["last_values"] = last
    metrics = {
        "mean_reward": float(np.mean(reward)),
        "mean_served_bits": float(np.mean(served)),
        "projection_rate": float(np.mean(projected)),
        "safety_violation_rate": float(np.mean(safety)),
        "command_turn_deg": float(np.mean(turn)),
        "executed_turn_deg": float(np.mean(executed_turn)),
        "correction_norm": float(np.mean(correction)),
        "excessive_heading_rate": float(np.mean(excessive_heading)),
        "excessive_speed_rate": float(np.mean(excessive_speed)),
        "collision_risk_rate": float(np.mean(collision)),
        "envelope_risk_rate": float(np.mean(envelope)),
        "nan_inf": int(not np.isfinite(storage["rewards"]).all()),
    }
    return storage, metrics


def train_seed(model_seed: int, updates: int, checkpoint_path: Path, calibration: dict, payload: dict) -> dict:
    destination = assert_phase_output(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source_sha256()
    torch.set_num_threads(1)
    seed_everything(int(model_seed))
    actor = load_warm_actor(payload)
    critic, critic_mode = load_critic(payload)
    agent = ICARMAPPOLocked(actor, critic, PPO)
    workers = []
    for index in range(int(PPO.num_envs)):
        worker = AdaptiveTrainWorker(int(model_seed), index, actor, critic, calibration)
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
            "gradient_norm_preclip": stats["gradient_norm_preclip"],
            "explained_variance": stats["explained_variance"],
            "critic_mode": critic_mode,
            **metrics,
        }
        history.append(row)
        if (update + 1) % 10 == 0 or update == 0:
            print(
                f"adaptive seed {model_seed} update {update + 1}/{updates} reward={row['mean_reward']:.4f} served={row['mean_served_bits']:.1f} turn={row['command_turn_deg']:.2f} proj={row['projection_rate']:.3f}",
                flush=True,
            )
    if source_sha256() != before:
        raise RuntimeError("frozen MAPPO checkpoint changed")
    actor.eval()
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "calibration": calibration,
            "model_seed": int(model_seed),
            "updates": int(updates),
            "critic_mode": critic_mode,
            "action": "adaptive_residual_turn_and_speed",
            "reward": "phase6g low_level_reward",
            "loaded_phase6il_actor": False,
            "loaded_phase6ik_actor": False,
            "source_sha256": before,
            "official_config": False,
        },
        destination,
    )
    destination.with_suffix(".history.json").write_text(json.dumps(history), encoding="utf-8")
    return {"model_seed": int(model_seed), "checkpoint": str(destination), "history": history, "critic_mode": critic_mode}


__all__ = ["load_critic", "load_source", "load_warm_actor", "source_sha256", "train_seed"]
