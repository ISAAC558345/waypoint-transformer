"""Active-sample-aware Actor MAPPO. Critic GAE/value path unchanged in formula."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.lowlevel_v2.phase1b.mappo import tanh_log_prob_uav
from wt.lowlevel_v2.phase2d.protocol import ADV_EPS, MAX_ACTOR_MINIBATCHES
from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor
from wt.round30.ppo import PPOConfig, compute_gae


def actor_minibatch_sizes(n_active: int, max_mb: int = MAX_ACTOR_MINIBATCHES) -> list[int]:
    if int(n_active) <= 0:
        return []
    k = min(int(max_mb), int(n_active))
    base, rem = divmod(int(n_active), k)
    sizes = [base + (1 if i < rem else 0) for i in range(k)]
    if any(s <= 0 for s in sizes):
        raise RuntimeError("zero-active actor minibatch constructed")
    return sizes


def split_active_indices(order: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    batches = []
    start = 0
    for size in sizes:
        if size <= 0:
            raise RuntimeError("zero-active actor minibatch constructed")
        batches.append(np.asarray(order[start : start + size]))
        start += int(size)
    if start != len(order):
        raise RuntimeError("active samples not partitioned exactly once")
    return batches


def active_actor_advantage(advantages: np.ndarray, mask: np.ndarray, eps: float = ADV_EPS) -> tuple[np.ndarray, dict]:
    """Broadcast env-step GAE to UAVs; normalize using active UAV-steps only."""
    if advantages.ndim != 1:
        raise ValueError("advantages must be flattened env-steps")
    if mask.shape[0] != advantages.shape[0] or mask.ndim != 2:
        raise ValueError("mask must be (env_steps, n_uav)")
    adv_uav = np.repeat(advantages.reshape(-1, 1), mask.shape[1], axis=1).astype(np.float64)
    active = np.asarray(mask, dtype=np.float64) > 0.5
    locked = ~active
    raw_active = adv_uav[active]
    stats = {
        "n_active": int(np.sum(active)),
        "n_locked": int(np.sum(locked)),
        "active_advantage_mean_before": float(np.mean(raw_active)) if raw_active.size else float("nan"),
        "active_advantage_std_before": float(np.std(raw_active)) if raw_active.size else float("nan"),
        "locked_used_in_actor_norm": False,
    }
    if raw_active.size == 0:
        stats["active_advantage_mean_after"] = float("nan")
        stats["active_advantage_std_after"] = float("nan")
        return np.zeros_like(adv_uav, dtype=np.float32), stats
    mu = float(np.mean(raw_active))
    std = float(np.std(raw_active))
    normed = (adv_uav - mu) / (std + float(eps))
    after = normed[active]
    stats["active_advantage_mean_after"] = float(np.mean(after))
    stats["active_advantage_std_after"] = float(np.std(after))
    if locked.any():
        # locked values exist in the array but must not have entered mu/std
        if abs(float(np.mean(raw_active)) - mu) > 1.0e-12:
            raise RuntimeError("locked samples leaked into actor advantage mean")
    return normed.astype(np.float32), stats


class ICARMAPPOActiveAware:
    def __init__(self, actor: ResidualActor, critic: CentralizedCritic, config: PPOConfig) -> None:
        self.actor = actor
        self.critic = critic
        self.config = config
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)

    def update(self, storage: dict[str, np.ndarray]) -> dict[str, float]:
        steps, envs = storage["rewards"].shape
        advantages, returns = compute_gae(
            storage["rewards"], storage["values"], storage["dones"], storage["last_values"], self.config.gamma, self.config.gae_lambda
        )
        count = steps * envs
        critic_mb = count // self.config.num_minibatches
        if critic_mb < 1 or count % self.config.num_minibatches:
            raise ValueError("MAPPO critic batch cannot fill registered minibatches")

        def flat(key):
            value = storage[key]
            return value.reshape((count,) + value.shape[2:])

        self_f = flat("self")
        task_f = flat("task")
        comm_f = flat("communication")
        neigh = flat("neighbor_set")
        near = flat("d_nearest")
        glob = flat("global_state")
        pre = flat("pre_tanh")
        old_lp = flat("log_prob_uav")
        mask_np = flat("policy_active_mask")
        adv_env = advantages.reshape(-1).astype(np.float32)
        ret = returns.reshape(-1)
        a_norm, adv_stats = active_actor_advantage(adv_env, mask_np)
        env_i, uav_i = np.nonzero(mask_np > 0.5)
        n_active = int(env_i.shape[0])
        sizes = actor_minibatch_sizes(n_active, self.config.num_minibatches)
        actor_rows: list[dict[str, float]] = []
        critic_rows: list[dict[str, float]] = []
        self.actor.train()
        self.critic.train()
        actor_steps = 0
        for _ in range(self.config.update_epochs):
            order = np.random.permutation(count)
            for start in range(0, count, critic_mb):
                index = order[start : start + critic_mb]
                glob_t = torch.as_tensor(glob[index], dtype=torch.float32)
                target = torch.as_tensor(ret[index], dtype=torch.float32)
                value = self.critic(glob_t)
                value_loss = 0.5 * (target - value).square().mean()
                critic_loss = self.config.value_coef * value_loss
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_gn = float(clip_grad_norm_(list(self.critic.parameters()), self.config.max_grad_norm).detach().cpu())
                self.critic_optimizer.step()
                critic_rows.append({"value_loss": float(value_loss.detach().cpu()), "critic_gradient_norm": critic_gn})
            if n_active == 0:
                continue
            perm = np.random.permutation(n_active)
            batches = split_active_indices(perm, sizes)
            for batch in batches:
                ei = env_i[batch]
                ui = uav_i[batch]
                self_t = torch.as_tensor(self_f[ei, ui], dtype=torch.float32)
                task_t = torch.as_tensor(task_f[ei, ui], dtype=torch.float32)
                comm_t = torch.as_tensor(comm_f[ei, ui], dtype=torch.float32)
                neigh_t = torch.as_tensor(neigh[ei, ui], dtype=torch.float32)
                near_t = torch.as_tensor(near[ei, ui], dtype=torch.float32)
                pre_t = torch.as_tensor(pre[ei, ui], dtype=torch.float32)
                old = torch.as_tensor(old_lp[ei, ui], dtype=torch.float32)
                advantage = torch.as_tensor(a_norm[ei, ui], dtype=torch.float32)
                mu, log_std = self.actor(self_t, task_t, comm_t, neigh_t, near_t)
                log_prob, entropy = tanh_log_prob_uav(mu, log_std, pre_t)
                ratio = (log_prob - old).exp()
                clipped = ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
                per = -torch.minimum(ratio * advantage, clipped * advantage)
                policy_loss = per.mean()
                entropy_term = entropy.mean()
                actor_loss = policy_loss - self.config.entropy_coef * entropy_term
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_gn = float(clip_grad_norm_(list(self.actor.parameters()), self.config.max_grad_norm).detach().cpu())
                self.actor_optimizer.step()
                actor_steps += 1
                kl = (old - log_prob).mean()
                clip_frac = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean()
                actor_rows.append(
                    {
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "entropy": float(entropy_term.detach().cpu()),
                        "approx_kl": float(kl.detach().cpu()),
                        "clip_fraction": float(clip_frac.detach().cpu()),
                        "actor_gradient_norm": actor_gn,
                        "action_std": float(log_std.exp().mean().detach().cpu()),
                        "actor_minibatch_size": float(len(batch)),
                    }
                )
        explained = 1.0 - float(np.var(ret - storage["values"].reshape(-1)) / max(np.var(ret), 1.0e-8))
        if critic_rows and not all(np.isfinite(v) for row in critic_rows for v in row.values()):
            raise RuntimeError("Critic update produced NaN/Inf")
        if actor_rows and not all(np.isfinite(v) for row in actor_rows for v in row.values()):
            raise RuntimeError("Actor update produced NaN/Inf")
        if not np.isfinite(explained):
            raise RuntimeError("explained variance NaN/Inf")
        total_uav = int(mask_np.size)
        out = {
            "policy_loss": float(np.mean([r["policy_loss"] for r in actor_rows])) if actor_rows else 0.0,
            "entropy": float(np.mean([r["entropy"] for r in actor_rows])) if actor_rows else 0.0,
            "approx_kl": float(np.mean([r["approx_kl"] for r in actor_rows])) if actor_rows else 0.0,
            "clip_fraction": float(np.mean([r["clip_fraction"] for r in actor_rows])) if actor_rows else 0.0,
            "actor_gradient_norm": float(np.mean([r["actor_gradient_norm"] for r in actor_rows])) if actor_rows else 0.0,
            "action_std": float(np.mean([r["action_std"] for r in actor_rows])) if actor_rows else 0.0,
            "value_loss": float(np.mean([r["value_loss"] for r in critic_rows])),
            "critic_gradient_norm": float(np.mean([r["critic_gradient_norm"] for r in critic_rows])),
            "explained_variance": explained,
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
            "total_agent_samples": float(total_uav),
            "active_samples": float(n_active),
            "active_fraction": float(n_active / max(total_uav, 1)),
            "actor_minibatch_count": float(len(sizes)),
            "min_actor_minibatch_size": float(min(sizes) if sizes else 0),
            "max_actor_minibatch_size": float(max(sizes) if sizes else 0),
            "mean_actor_minibatch_size": float(np.mean(sizes) if sizes else 0.0),
            "actor_optimizer_steps": float(actor_steps),
            "critic_optimizer_steps": float(len(critic_rows)),
            "policy_active_fraction": float(n_active / max(total_uav, 1)),
            **adv_stats,
        }
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "loaded_phase101_ppo_weights": False,
            "loaded_phase1_actor": False,
            "loaded_phase2b_weights": False,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if "critic_optimizer" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])


__all__ = [
    "ICARMAPPOActiveAware",
    "active_actor_advantage",
    "actor_minibatch_sizes",
    "split_active_indices",
]
