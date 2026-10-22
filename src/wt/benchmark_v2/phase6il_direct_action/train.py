"""Fresh direct-velocity actor. Body is fit to fly-toward, then PPO uses the bit reward."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT, PPO, assert_allowed
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option, sample_assignment
from wt.benchmark_v2.phase6ik_lowlevel_opt.scale import fold_scales_into_actor, scale_actor_observation
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import clip_unit_step
from wt.lowlevel_v2.critic_state import build_critic_state, expected_critic_dim
from wt.lowlevel_v2.observation import build_actor_observation
from wt.lowlevel_v2.phase1b.mappo import ICARMAPPOLocked
from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor
from wt.round30.ppo import seed_everything

from .protocol import BC_EPOCHS, BC_SCENARIOS, SOURCE_ACTOR, assert_phase_output
from .worker import DirectTrainWorker, execute_command


def source_sha256() -> str:
    return hashlib.sha256(Path(SOURCE_ACTOR).read_bytes()).hexdigest()


def new_actor() -> ResidualActor:
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    if tuple(actor.mu.weight.shape) != (2, 64):
        raise RuntimeError("direct actor is not a 2-D command head")
    return actor


def _teacher_batch(model_seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(50_000 + int(model_seed))
    self_rows, task_rows, comm_rows, neigh_rows, near_rows, targets = [], [], [], [], [], []
    for scenario in BC_SCENARIOS:
        assert_allowed(int(scenario), split="train")
        env = L3BBitQueueEnvV2()
        install_light_info(env)
        env.reset(seed=int(scenario))
        option = None
        pending = 0
        last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
        while int(env.t) < int(env.config.episode_horizon):
            if pending <= 0:
                option = make_option(env, sample_assignment(env, rng))
                pending = 15
            association = np.asarray(option.association, dtype=np.int64).copy()
            obs = scale_actor_observation(build_actor_observation(env, option, last))
            target = 0.9 * fly_toward(env, association)
            self_rows.append(obs["self"])
            task_rows.append(obs["task"])
            comm_rows.append(obs["communication"])
            neigh_rows.append(obs["neighbor_set"])
            near_rows.append(obs["d_nearest"])
            targets.append(target)
            executed, _safety = execute_command(env, option, fly_toward(env, association))
            env.step(clip_unit_step(executed), association)
            pending -= 1
            option.remaining_hold = int(pending)
            last = executed.copy()
    obs_batch = {
        "self": np.asarray(self_rows, dtype=np.float32),
        "task": np.asarray(task_rows, dtype=np.float32),
        "communication": np.asarray(comm_rows, dtype=np.float32),
        "neighbor_set": np.asarray(neigh_rows, dtype=np.float32),
        "d_nearest": np.asarray(near_rows, dtype=np.float32),
    }
    return obs_batch, np.asarray(targets, dtype=np.float32)


def behavior_clone(actor: ResidualActor, model_seed: int) -> dict[str, float]:
    obs, target = _teacher_batch(int(model_seed))
    actor.train()
    actor.log_std.requires_grad_(False)
    optimizer = torch.optim.Adam((parameter for parameter in actor.parameters() if parameter.requires_grad), lr=1.0e-3)
    goal = torch.as_tensor(target, dtype=torch.float32)
    tensors = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}
    last = 0.0
    cosine = 0.0
    for epoch in range(int(BC_EPOCHS)):
        optimizer.zero_grad(set_to_none=True)
        mu, _log_std = actor(tensors["self"], tensors["task"], tensors["communication"], tensors["neighbor_set"], tensors["d_nearest"])
        pred = torch.tanh(mu)
        loss = torch.mean((pred - goal) ** 2)
        loss.backward()
        optimizer.step()
        last = float(loss.detach().cpu())
        if epoch == 0 or (epoch + 1) % 20 == 0:
            with torch.no_grad():
                cosine = _active_cosine(pred.detach().cpu().numpy(), target)
            print(f"bc epoch {epoch + 1} loss={last:.4f} cosine={cosine:.3f}", flush=True)
    actor.log_std.requires_grad_(True)
    actor.eval()
    with torch.no_grad():
        mu, _log_std = actor(tensors["self"], tensors["task"], tensors["communication"], tensors["neighbor_set"], tensors["d_nearest"])
        pred = torch.tanh(mu).cpu().numpy()
    cosine = _active_cosine(pred, target)
    return {"bc_loss": last, "bc_cosine": float(cosine), "bc_rows": int(target.shape[0])}


def _active_cosine(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
    goal = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    goal_norm = np.linalg.norm(goal, axis=-1)
    pred_norm = np.linalg.norm(pred, axis=-1)
    active = goal_norm > 0.5
    if not np.any(active):
        return 1.0
    dots = np.sum(pred[active] * goal[active], axis=-1) / np.clip(pred_norm[active] * goal_norm[active], 1.0e-8, None)
    return float(np.mean(dots))


def _collect(workers: list[DirectTrainWorker], horizon: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
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
    served, projected, safety, reward, correction = [], [], [], [], []
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
            projected.append(rec["projected"])
            safety.append(rec["safety_violation"])
            reward.append(rec["reward"])
            correction.append(rec["correction_norm"])
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
        "correction_norm": float(np.mean(correction)),
        "pre_tanh_abs": float(np.mean(np.abs(storage["pre_tanh"]))),
        "nan_inf": int(not np.isfinite(storage["rewards"]).all()),
    }
    return storage, metrics


def train_seed(model_seed: int, updates: int, checkpoint_path: Path) -> dict:
    destination = assert_phase_output(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source_sha256()
    torch.set_num_threads(1)
    seed_everything(int(model_seed))
    actor = new_actor()
    clone = behavior_clone(actor, int(model_seed))
    critic = CentralizedCritic(expected_critic_dim(6, 12))
    agent = ICARMAPPOLocked(actor, critic, PPO)
    workers = []
    for index in range(int(PPO.num_envs)):
        worker = DirectTrainWorker(int(model_seed), index, actor, critic)
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
            **metrics,
            **clone,
        }
        history.append(row)
        if (update + 1) % 10 == 0 or update == 0:
            print(
                f"direct seed {model_seed} update {update + 1}/{updates} reward={row['mean_reward']:.4f} served={row['mean_served_bits']:.1f} proj={row['projection_rate']:.3f}",
                flush=True,
            )
    if source_sha256() != before:
        raise RuntimeError("frozen MAPPO checkpoint changed during direct-action training")
    actor.eval()
    folded = fold_scales_into_actor(actor)
    folded.eval()
    torch.save(
        {
            "actor": folded.state_dict(),
            "model_seed": int(model_seed),
            "updates": int(updates),
            "initialization": "independent_plus_fly_toward_behavior_clone",
            "action": "direct_unit_step_velocity",
            "reward": "phase6g low_level_reward on commanded velocity",
            "loaded_phase6g_actor": False,
            "loaded_phase6ik_actor": False,
            "source_sha256": before,
            **clone,
        },
        destination,
    )
    destination.with_suffix(".history.json").write_text(json.dumps(history), encoding="utf-8")
    return {"model_seed": int(model_seed), "checkpoint": str(destination), "history": history, **clone}


__all__ = ["behavior_clone", "new_actor", "source_sha256", "train_seed"]
