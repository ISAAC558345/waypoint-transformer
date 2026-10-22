"""One MAPPO seed. Dev scenarios are not reset here."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LEGACY_ACTOR, LOG_STD_INIT, MODEL_SEEDS, PPO, UPDATES
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import TrainWorker
from wt.lowlevel_v2.critic_state import expected_critic_dim
from wt.lowlevel_v2.phase1b.mappo import ICARMAPPOLocked
from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor
from wt.round30.ppo import seed_everything


def load_actor(initialization: str) -> ResidualActor:
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    if initialization == "LEGACY_WARM_START":
        state = torch.load(LEGACY_ACTOR, map_location="cpu", weights_only=False)
        actor.load_state_dict(state, strict=True)
    return actor


def probe_initialization() -> str:
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    state = torch.load(LEGACY_ACTOR, map_location="cpu", weights_only=False)
    actor.load_state_dict(state, strict=True)
    if tuple(actor.mu.weight.shape) != (2, 64):
        raise RuntimeError("actor action interface is not the residual convention")
    _ = expected_critic_dim(6, 12)
    return "LEGACY_WARM_START"


def _collect(workers: list[TrainWorker], horizon: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
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
    served = []
    sinr = []
    interference = []
    path = []
    residual = []
    projected = []
    safety = []
    r_comm = []
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
        if storage["dones"][-1, index] > 0.5 or worker.option is None:
            continue
        glob = build_last_value(worker)
        last[index] = glob
    storage["last_values"] = last
    metrics = {
        "mean_episode_payload_bits": float(np.mean(payload)),
        "r_comm": float(np.mean(r_comm)),
        "mean_served_bits": float(np.mean(served)),
        "mean_sinr": float(np.mean(sinr)),
        "mean_interference": float(np.mean(interference)),
        "path_length": float(np.sum(path) / n_env),
        "residual_action_norm": float(np.mean(residual)),
        "projection_rate": float(np.mean(projected)),
        "safety_violation_rate": float(np.mean(safety)),
        "nan_inf": int(not np.isfinite(storage["rewards"]).all()),
    }
    return storage, metrics


def build_last_value(worker: TrainWorker) -> float:
    from wt.lowlevel_v2.critic_state import build_critic_state

    glob = build_critic_state(worker.env, worker.option, worker.last_movement)
    with torch.no_grad():
        return float(worker.critic(torch.as_tensor(glob["vector"], dtype=torch.float32)).cpu())


def train_seed(model_seed: int, initialization: str, checkpoint_path: str) -> dict:
    if int(model_seed) not in MODEL_SEEDS:
        raise RuntimeError(f"unexpected model seed {model_seed}")
    torch.set_num_threads(1)
    seed_everything(int(model_seed))
    actor = load_actor(initialization)
    critic = CentralizedCritic(expected_critic_dim(6, 12))
    agent = ICARMAPPOLocked(actor, critic, PPO)
    workers = []
    for index in range(int(PPO.num_envs)):
        worker = TrainWorker(int(model_seed), index, actor, critic)
        worker.reset()
        workers.append(worker)
    history = []
    for update in range(int(UPDATES)):
        storage, metrics = _collect(workers, int(PPO.rollout_length))
        if metrics["nan_inf"]:
            raise RuntimeError("NONFINITE")
        stats = agent.update(storage)
        row = {
            "model_seed": int(model_seed),
            "update": update + 1,
            "initialization": initialization,
            "actor_loss": stats["policy_loss"],
            "critic_loss": stats["value_loss"],
            "entropy": stats["entropy"],
            "kl": stats["approx_kl"],
            "clip_fraction": stats["clip_fraction"],
            "explained_variance": stats["explained_variance"],
            **metrics,
        }
        history.append(row)
        if (update + 1) % 10 == 0 or update == 0:
            print(
                f"seed {model_seed} update {update + 1}/{UPDATES} reward={float(np.mean(storage['rewards'])):.4f} served={metrics['mean_served_bits']:.1f}",
                flush=True,
            )
    actor.eval()
    critic.eval()
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "model_seed": int(model_seed),
            "updates": int(UPDATES),
            "initialization": initialization,
        },
        checkpoint_path,
    )
    return {
        "model_seed": int(model_seed),
        "history": history,
        "projection_rate": float(np.mean([row["projection_rate"] for row in history])),
        "safety_violation_rate": float(np.mean([row["safety_violation_rate"] for row in history])),
    }
