"""MAPPO update over shared residual actor + centralized critic. Does not load old PPO weights."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.round30.ppo import PPOConfig, compute_gae, tanh_log_prob

from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor


class ICARMAPPO:
    def __init__(self, actor: ResidualActor, critic: CentralizedCritic, config: PPOConfig) -> None:
        self.actor = actor
        self.critic = critic
        self.config = config
        self.optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=config.learning_rate)

    def update(self, storage: dict[str, np.ndarray]) -> dict[str, float]:
        steps, envs = storage["rewards"].shape
        advantages, returns = compute_gae(
            storage["rewards"], storage["values"], storage["dones"], storage["last_values"], self.config.gamma, self.config.gae_lambda
        )
        count = steps * envs
        minibatch = count // self.config.num_minibatches
        if minibatch < 1 or count % self.config.num_minibatches:
            raise ValueError("MAPPO batch cannot fill registered minibatches")

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
        old_lp = storage["log_prob"].reshape(count)
        adv = (advantages.reshape(-1) - advantages.mean()) / max(float(advantages.std()), 1.0e-6)
        ret = returns.reshape(-1)
        records: list[dict[str, float]] = []
        self.actor.train()
        self.critic.train()
        for _ in range(self.config.update_epochs):
            order = np.random.permutation(count)
            for start in range(0, count, minibatch):
                index = order[start : start + minibatch]
                self_t = torch.as_tensor(self_f[index], dtype=torch.float32)
                task_t = torch.as_tensor(task_f[index], dtype=torch.float32)
                comm_t = torch.as_tensor(comm_f[index], dtype=torch.float32)
                neigh_t = torch.as_tensor(neigh[index], dtype=torch.float32)
                near_t = torch.as_tensor(near[index], dtype=torch.float32)
                glob_t = torch.as_tensor(glob[index], dtype=torch.float32)
                pre_t = torch.as_tensor(pre[index], dtype=torch.float32)
                old = torch.as_tensor(old_lp[index], dtype=torch.float32)
                advantage = torch.as_tensor(adv[index], dtype=torch.float32)
                target = torch.as_tensor(ret[index], dtype=torch.float32)
                mu, log_std = self.actor(self_t, task_t, comm_t, neigh_t, near_t)
                log_prob, entropy = tanh_log_prob(mu, log_std, pre_t)
                value = self.critic(glob_t)
                ratio = (log_prob - old).exp()
                clipped = ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
                policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                value_loss = 0.5 * (target - value).square().mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.config.max_grad_norm).detach().cpu())
                self.optimizer.step()
                records.append(
                    {
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "entropy": float(entropy.mean().detach().cpu()),
                        "approx_kl": float((old - log_prob).mean().detach().cpu()),
                        "clip_fraction": float(((ratio - 1.0).abs() > self.config.clip_eps).float().mean().detach().cpu()),
                        "gradient_norm_preclip": grad_norm,
                        "action_std": float(log_std.exp().mean().detach().cpu()),
                    }
                )
        explained = 1.0 - float(np.var(ret - storage["values"].reshape(-1)) / max(np.var(ret), 1.0e-8))
        if not np.isfinite(explained) or not all(np.isfinite(v) for row in records for v in row.values()):
            raise RuntimeError("MAPPO update produced NaN/Inf")
        return {key: float(np.mean([row[key] for row in records])) for key in records[0]} | {
            "explained_variance": explained,
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loaded_phase101_ppo_weights": False,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.optimizer.load_state_dict(payload["optimizer"])
