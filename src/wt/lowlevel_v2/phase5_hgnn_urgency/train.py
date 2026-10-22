"""Sequential Phase-5 training. High-level and low-level are not updated together."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.htw_phase943.set_context import make_snap
from wt.htw_phase943.set_likelihood_dp import set_nll
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.phase1b.mappo import ICARMAPPOLocked
from wt.lowlevel_v2.phase1b.protocol import PPO
from wt.round30.ppo import seed_everything

from .features import current_urgency, graph_batch, observable_exclusive_assignment
from .mappo_urgency import UrgencyMAPPO
from .planner import Phase5Planner
from .protocol import HL_EPOCHS, HL_LR, MAPPO_UPDATES, assert_phase5_seed
from .worker import Phase5Worker, critic_dim, critic_vector, make_actor, make_critic


def collect_observable_samples(seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    template_positions = None
    for seed in seeds:
        assert_phase5_seed(int(seed))
        env = L3BEnv()
        info = env.reset(seed=int(seed))
        last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        template_positions = np.asarray(env.buoy_positions, dtype=np.float64).copy()
        while not info.get("terminated", False):
            if int(env.t) % 15 == 0:
                label = observable_exclusive_assignment(env)
                batch = graph_batch(info, env, last_arrival, last_positions, urgency=None)
                rows.append(
                    {
                        "batch": {key: np.array(value, copy=True) for key, value in batch.items()},
                        "label": np.asarray(label, dtype=np.int64).copy(),
                        "urgency": current_urgency(env).astype(np.float32),
                        "uav_positions": np.asarray(env.uav_positions, dtype=np.float64).copy(),
                        "buoy_positions": template_positions.copy(),
                    }
                )
            label = observable_exclusive_assignment(env)
            after = env.step(fly_toward(env, label), label)
            last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
            last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
            info = after
    if not rows:
        raise RuntimeError("phase-5 collected no high-level samples")
    return rows


def _tensors(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    out = {}
    for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
        dtype = torch.bool if key.endswith("mask") else torch.float32
        out[key] = torch.as_tensor(batch[key], dtype=dtype)
    return out


def _with_urgency(batch: dict[str, np.ndarray], urgency: np.ndarray) -> dict[str, np.ndarray]:
    from wt.htw_phase9.const import MAX_BUOYS

    copied = {key: np.array(value, copy=True) for key, value in batch.items()}
    extra = np.zeros((1, MAX_BUOYS, 1), dtype=np.float32)
    extra[0, : urgency.shape[0], 0] = np.asarray(urgency, dtype=np.float32)
    copied["buoy"] = np.concatenate((copied["buoy"], extra), axis=-1)
    return copied


def train_hgnn(model, rows: list[dict], *, high_urgency: bool, epochs: int = HL_EPOCHS) -> dict:
    seed_everything(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(HL_LR))
    template = L3BEnv()
    last_loss = float("nan")
    model.train()
    for epoch in range(int(epochs)):
        order = np.random.permutation(len(rows))
        losses = []
        for index in order:
            row = rows[int(index)]
            batch = _with_urgency(row["batch"], row["urgency"]) if high_urgency else row["batch"]
            tensors = _tensors(batch)
            encoded = model.encode(
                tensors["uav"],
                tensors["buoy"],
                tensors["uav_mask"],
                tensors["buoy_mask"],
                tensors["global"],
                tensors["edge"],
                tensors["edge_uu"],
            )
            snap = make_snap(row["uav_positions"], row["buoy_positions"], template.config, env=template)
            loss = set_nll(model, encoded, snap, row["label"], np.asarray(batch["buoy_mask"][0], dtype=bool))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses))
        print(f"hgnn urgency={high_urgency} epoch {epoch + 1}/{epochs} loss={last_loss:.4f}", flush=True)
    model.eval()
    return {"epochs": int(epochs), "final_loss": last_loss, "samples": len(rows), "high_urgency": bool(high_urgency), "joint_with_mappo": False}


def _collect_rollout(workers: list[Phase5Worker], horizon: int) -> dict[str, np.ndarray]:
    n_env = len(workers)
    use_urgency = bool(workers[0].use_urgency)
    glob_dim = critic_dim(use_urgency=use_urgency)
    infos = [worker.env.info() for worker in workers]
    keys = {
        "self": (horizon, n_env, 6, 5),
        "task": (horizon, n_env, 6, 8),
        "communication": (horizon, n_env, 6, 4),
        "neighbor_set": (horizon, n_env, 6, 5, 11),
        "d_nearest": (horizon, n_env, 6),
        "urgency": (horizon, n_env, 6, 1),
        "global_state": (horizon, n_env, int(glob_dim)),
        "pre_tanh": (horizon, n_env, 6, 2),
        "log_prob_uav": (horizon, n_env, 6),
        "policy_active_mask": (horizon, n_env, 6),
        "values": (horizon, n_env),
        "rewards": (horizon, n_env),
        "dones": (horizon, n_env),
    }
    storage = {key: np.zeros(shape, dtype=np.float32) for key, shape in keys.items()}
    for t in range(horizon):
        for env_index, worker in enumerate(workers):
            infos[env_index], rec = worker.step(infos[env_index], deterministic=False)
            storage["self"][t, env_index] = rec["obs"]["self"]
            storage["task"][t, env_index] = rec["obs"]["task"]
            storage["communication"][t, env_index] = rec["obs"]["communication"]
            storage["neighbor_set"][t, env_index] = rec["obs"]["neighbor_set"]
            storage["d_nearest"][t, env_index] = rec["obs"]["d_nearest"]
            storage["urgency"][t, env_index] = rec["urgency"]
            storage["global_state"][t, env_index] = rec["global_state"]
            storage["pre_tanh"][t, env_index] = rec["pre_tanh"]
            storage["log_prob_uav"][t, env_index] = rec["log_prob_uav"]
            storage["policy_active_mask"][t, env_index] = rec["policy_active_mask"]
            storage["values"][t, env_index] = rec["value"]
            storage["rewards"][t, env_index] = rec["reward"]
            storage["dones"][t, env_index] = float(rec["done"])
            if rec["done"]:
                infos[env_index] = worker.reset(worker._next_seed())
    last_values = np.zeros(n_env, dtype=np.float32)
    for env_index, worker in enumerate(workers):
        if storage["dones"][-1, env_index] > 0.5 or worker.option is None:
            last_values[env_index] = 0.0
            continue
        glob = critic_vector(worker.env, worker.option, worker.last_movement, use_urgency=use_urgency)
        with torch.no_grad():
            last_values[env_index] = float(worker.critic(torch.as_tensor(glob, dtype=torch.float32)).cpu())
    storage["last_values"] = last_values
    return storage


def train_mappo(planner: Phase5Planner, *, use_urgency: bool, seeds: tuple[int, ...], updates: int = MAPPO_UPDATES):
    seed_everything(0)
    actor = make_actor(use_urgency)
    critic = make_critic(use_urgency)
    agent = UrgencyMAPPO(actor, critic, PPO) if use_urgency else ICARMAPPOLocked(actor, critic, PPO)
    workers = []
    for index in range(int(PPO.num_envs)):
        worker = Phase5Worker(planner, actor, critic, use_urgency=use_urgency, seeds=seeds)
        worker.reset(int(seeds[index % len(seeds)]))
        worker.cursor = index
        workers.append(worker)
    history = []
    for update in range(int(updates)):
        storage = _collect_rollout(workers, int(PPO.rollout_length))
        stats = agent.update(storage)
        history.append({"update": update + 1, "reward_mean": float(np.mean(storage["rewards"])), **stats})
        print(
            f"mappo urgency={use_urgency} update {update + 1}/{updates} reward={history[-1]['reward_mean']:.4f} policy={stats['policy_loss']:.4f}",
            flush=True,
        )
    actor.eval()
    critic.eval()
    return actor, critic, {"updates": int(updates), "use_urgency": bool(use_urgency), "joint_with_hgnn": False, "final": history[-1]}


__all__ = ["collect_observable_samples", "train_hgnn", "train_mappo"]
