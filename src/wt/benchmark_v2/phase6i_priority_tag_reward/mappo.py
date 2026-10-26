"""MAPPO update with the extra observation. Optimizer coefficients stay on the locked PPO config."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.lowlevel_v2.phase1b.mappo import ICARMAPPOLocked, tanh_log_prob_uav
from wt.round30.ppo import PPOConfig, compute_gae


class ExperimentMAPPO(ICARMAPPOLocked):
    def __init__(self, actor, critic, config: PPOConfig) -> None:
        super().__init__(actor, critic, config)

    def update(self, storage: dict[str, np.ndarray]) -> dict[str, float]:
        steps, envs = storage["rewards"].shape
        advantages, returns = compute_gae(
            storage["rewards"],
            storage["values"],
            storage["dones"],
            storage["last_values"],
            self.config.gamma,
            self.config.gae_lambda,
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
        priority = flat("priority")
        glob = flat("global_state")
        pre = flat("pre_tanh")
        old_lp = flat("log_prob_uav")
        mask_np = flat("policy_active_mask")
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
                priority_t = torch.as_tensor(priority[index], dtype=torch.float32)
                glob_t = torch.as_tensor(glob[index], dtype=torch.float32)
                pre_t = torch.as_tensor(pre[index], dtype=torch.float32)
                old = torch.as_tensor(old_lp[index], dtype=torch.float32)
                mask = torch.as_tensor(mask_np[index], dtype=torch.float32)
                advantage = torch.as_tensor(adv[index], dtype=torch.float32).unsqueeze(-1)
                target = torch.as_tensor(ret[index], dtype=torch.float32)
                mu, log_std = self.actor(self_t, task_t, comm_t, neigh_t, near_t, priority_t)
                log_prob, entropy = tanh_log_prob_uav(mu, log_std, pre_t)
                value = self.critic(glob_t)
                ratio = (log_prob - old).exp()
                clipped = ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
                per_uav = -torch.minimum(ratio * advantage, clipped * advantage)
                denom = mask.sum().clamp_min(1.0)
                policy_loss = (per_uav * mask).sum() / denom
                value_loss = 0.5 * (target - value).square().mean()
                entropy_term = (entropy * mask).sum() / denom
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_term
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(
                    clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.config.max_grad_norm).detach().cpu()
                )
                if not np.isfinite(grad_norm):
                    raise RuntimeError("NONFINITE")
                self.optimizer.step()
                kl = ((old - log_prob) * mask).sum() / denom
                clip_frac = (((ratio - 1.0).abs() > self.config.clip_eps).float() * mask).sum() / denom
                records.append(
                    {
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "entropy": float(entropy_term.detach().cpu()),
                        "approx_kl": float(kl.detach().cpu()),
                        "clip_fraction": float(clip_frac.detach().cpu()),
                        "gradient_norm_preclip": grad_norm,
                        "action_std": float(log_std.exp().mean().detach().cpu()),
                        "policy_active_fraction": float(mask.mean().detach().cpu()),
                    }
                )
        explained = 1.0 - float(np.var(ret - storage["values"].reshape(-1)) / max(np.var(ret), 1.0e-8))
        if not np.isfinite(explained) or not all(np.isfinite(value) for row in records for value in row.values()):
            raise RuntimeError("NONFINITE")
        return {key: float(np.mean([row[key] for row in records])) for key in records[0]} | {
            "explained_variance": explained,
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
        }
