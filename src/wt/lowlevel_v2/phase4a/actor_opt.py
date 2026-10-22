"""Active-sample MAPPO plus collision-distill auxiliary. PPO ratio unchanged."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.lowlevel_v2.phase1b.mappo import tanh_log_prob_uav
from wt.lowlevel_v2.phase2d.actor_opt import active_actor_advantage, actor_minibatch_sizes, split_active_indices
from wt.lowlevel_v2.phase2d.protocol import MAX_ACTOR_MINIBATCHES
from wt.lowlevel_v2.phase4a.protocol import DISTILL_EPS, LAMBDA_COLLISION_DISTILL
from wt.lowlevel_v2.phase4a.transform import residual_mean_used
from wt.lowlevel_v2.phase2b.trust_region import action_max
from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor
from wt.round30.ppo import PPOConfig, compute_gae


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().float().norm(2).cpu() ** 2)
    return float(total ** 0.5)


class ICARMAPPOCollisionDistill:
    def __init__(self, actor: ResidualActor, critic: CentralizedCritic, config: PPOConfig, *, lambda_distill: float = LAMBDA_COLLISION_DISTILL) -> None:
        self.actor = actor
        self.critic = critic
        self.config = config
        self.lambda_distill = float(lambda_distill)
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
        dist_np = flat("collision_distill_mask")
        a_base_f = flat("a_base")
        teacher_f = flat("r_teacher")
        lock_f = flat("service_lock")
        freeze_f = flat("base_zero_lock")
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
        a_max2 = float(action_max()) ** 2 + DISTILL_EPS
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
                ppo_actor = policy_loss - self.config.entropy_coef * entropy_term
                distill_hit = (dist_np[ei, ui] > 0.5)
                n_dist = int(np.sum(distill_hit))
                if n_dist == 0:
                    distill_loss = ppo_actor * 0.0
                    skipped = True
                else:
                    skipped = False
                    sel = torch.as_tensor(distill_hit, dtype=torch.bool)
                    a_base_t = torch.as_tensor(a_base_f[ei, ui], dtype=torch.float32)
                    teacher = torch.as_tensor(teacher_f[ei, ui], dtype=torch.float32).detach()
                    lock_t = torch.as_tensor(lock_f[ei, ui], dtype=torch.float32)
                    freeze_t = torch.as_tensor(freeze_f[ei, ui], dtype=torch.float32)
                    r_mean = residual_mean_used(mu, a_base_t, lock_t, freeze_t)
                    diff = r_mean[sel] - teacher[sel]
                    distill_loss = (diff.square().sum(dim=-1) / a_max2).mean()
                actor_loss = ppo_actor + self.lambda_distill * distill_loss
                params = list(self.actor.parameters())
                self.actor_optimizer.zero_grad(set_to_none=True)
                ppo_actor.backward(retain_graph=True)
                ppo_gn = _grad_norm(params)
                self.actor_optimizer.zero_grad(set_to_none=True)
                if skipped:
                    distill_gn = 0.0
                else:
                    (self.lambda_distill * distill_loss).backward(retain_graph=True)
                    distill_gn = _grad_norm(params)
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                total_pre = _grad_norm(params)
                actor_gn = float(clip_grad_norm_(params, self.config.max_grad_norm).detach().cpu())
                self.actor_optimizer.step()
                actor_steps += 1
                kl = (old - log_prob).mean()
                clip_frac = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean()
                if skipped:
                    mse = float("nan")
                    cosine = float("nan")
                    agree = float("nan")
                else:
                    r_det = r_mean[sel].detach()
                    t_det = teacher[sel]
                    mse = float((r_det - t_det).square().sum(dim=-1).mean().cpu())
                    rn = torch.nn.functional.normalize(r_det, dim=-1, eps=1e-8)
                    tn = torch.nn.functional.normalize(t_det, dim=-1, eps=1e-8)
                    cosine = float((rn * tn).sum(dim=-1).mean().cpu())
                    agree = float(((r_det * t_det).sum(dim=-1) > 0).float().mean().cpu())
                actor_rows.append(
                    {
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "entropy": float(entropy_term.detach().cpu()),
                        "approx_kl": float(kl.detach().cpu()),
                        "clip_fraction": float(clip_frac.detach().cpu()),
                        "actor_gradient_norm": actor_gn,
                        "ppo_actor_gradient_norm": ppo_gn,
                        "distill_gradient_norm": distill_gn,
                        "total_actor_gradient_norm_preclip": total_pre,
                        "action_std": float(log_std.exp().mean().detach().cpu()),
                        "actor_minibatch_size": float(len(batch)),
                        "collision_distill_loss": float(distill_loss.detach().cpu()) if not skipped else 0.0,
                        "n_collision_teacher": float(n_dist),
                        "teacher_mse": mse,
                        "teacher_cosine": cosine,
                        "teacher_direction_agreement": agree,
                        "distill_skipped": float(skipped),
                    }
                )
        explained = 1.0 - float(np.var(ret - storage["values"].reshape(-1)) / max(np.var(ret), 1.0e-8))
        if critic_rows and not all(np.isfinite(v) for row in critic_rows for v in row.values()):
            raise RuntimeError("Critic update produced NaN/Inf")
        finite_actor = []
        for row in actor_rows:
            for k, v in row.items():
                if k in {"teacher_mse", "teacher_cosine", "teacher_direction_agreement"} and not np.isfinite(v):
                    continue
                finite_actor.append(np.isfinite(v))
        if actor_rows and not all(finite_actor):
            raise RuntimeError("Actor update produced NaN/Inf")
        total_uav = int(mask_np.size)
        n_teacher = int(np.sum((mask_np > 0.5) & (dist_np > 0.5)))
        distill_rows = [r for r in actor_rows if r["n_collision_teacher"] > 0]
        ppo_g = float(np.mean([r["ppo_actor_gradient_norm"] for r in actor_rows])) if actor_rows else 0.0
        d_g = float(np.mean([r["distill_gradient_norm"] for r in actor_rows])) if actor_rows else 0.0
        out = {
            "policy_loss": float(np.mean([r["policy_loss"] for r in actor_rows])) if actor_rows else 0.0,
            "entropy": float(np.mean([r["entropy"] for r in actor_rows])) if actor_rows else 0.0,
            "approx_kl": float(np.mean([r["approx_kl"] for r in actor_rows])) if actor_rows else 0.0,
            "clip_fraction": float(np.mean([r["clip_fraction"] for r in actor_rows])) if actor_rows else 0.0,
            "actor_gradient_norm": float(np.mean([r["actor_gradient_norm"] for r in actor_rows])) if actor_rows else 0.0,
            "ppo_actor_gradient_norm": ppo_g,
            "distill_gradient_norm": d_g,
            "total_actor_gradient_norm_preclip": float(np.mean([r["total_actor_gradient_norm_preclip"] for r in actor_rows])) if actor_rows else 0.0,
            "distill_ppo_grad_ratio": float(d_g / max(ppo_g, 1.0e-12)),
            "action_std": float(np.mean([r["action_std"] for r in actor_rows])) if actor_rows else 0.0,
            "value_loss": float(np.mean([r["value_loss"] for r in critic_rows])),
            "critic_gradient_norm": float(np.mean([r["critic_gradient_norm"] for r in critic_rows])),
            "explained_variance": explained,
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
            "total_agent_samples": float(total_uav),
            "active_samples": float(n_active),
            "active_fraction": float(n_active / max(total_uav, 1)),
            "collision_teacher_samples": float(n_teacher),
            "collision_teacher_frac_of_active": float(n_teacher / max(n_active, 1)),
            "collision_distill_loss": float(np.mean([r["collision_distill_loss"] for r in actor_rows])) if actor_rows else 0.0,
            "teacher_mse": float(np.nanmean([r["teacher_mse"] for r in distill_rows])) if distill_rows else float("nan"),
            "teacher_cosine": float(np.nanmean([r["teacher_cosine"] for r in distill_rows])) if distill_rows else float("nan"),
            "teacher_direction_agreement": float(np.nanmean([r["teacher_direction_agreement"] for r in distill_rows])) if distill_rows else float("nan"),
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
            "lambda_collision_distill": self.lambda_distill,
            "loaded_phase3_weights": False,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if "critic_optimizer" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])


__all__ = ["ICARMAPPOCollisionDistill"]
