"""Pilot update. The loss matches ExperimentMAPPO; only the clip grouping changes."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import PPO
from wt.lowlevel_v2.phase1b.mappo import tanh_log_prob_uav
from wt.round30.ppo import compute_gae

from .clip import apply_clip
from .protocol import SHADOW_UPDATES


def _snapshot(module) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in module.parameters()]


def _delta_norm(saved: list[torch.Tensor], module) -> float:
    total = 0.0
    for before, parameter in zip(saved, module.parameters()):
        delta = parameter.detach().cpu().float() - before.float()
        total += float(torch.dot(delta.reshape(-1), delta.reshape(-1)))
    return float(total ** 0.5)


def _new_column_delta(saved_weight: torch.Tensor, module) -> float:
    weight = dict(module.named_parameters())["body.0.weight"].detach().cpu().float()
    delta = weight - saved_weight.float()
    return float(torch.linalg.vector_norm(delta[:, -2:]).cpu())


def _moment_summary(optimizer, actor_count: int) -> dict:
    actor_m = actor_v = critic_m = critic_v = 0.0
    step_count = 0
    for index, parameter in enumerate(optimizer.param_groups[0]["params"]):
        state = optimizer.state.get(parameter)
        if not state or "exp_avg" not in state:
            continue
        step_count = int(state["step"])
        first = state["exp_avg"].detach().float().reshape(-1)
        second = state["exp_avg_sq"].detach().float().reshape(-1)
        first_sum = float(torch.dot(first, first))
        second_sum = float(torch.dot(second, second))
        if index < int(actor_count):
            actor_m += first_sum
            actor_v += second_sum
        else:
            critic_m += first_sum
            critic_v += second_sum
    group = optimizer.param_groups[0]
    return {
        "learning_rate": float(group["lr"]),
        "beta1": float(group["betas"][0]),
        "beta2": float(group["betas"][1]),
        "adam_eps": float(group["eps"]),
        "adam_step_count": int(step_count),
        "actor_moment_norm": float(actor_m ** 0.5),
        "actor_second_moment_norm": float(actor_v ** 0.5),
        "critic_moment_norm": float(critic_m ** 0.5),
        "critic_second_moment_norm": float(critic_v ** 0.5),
    }


class GroupedPilot:
    """One Adam over actor and critic. PILOT_J is the original joint clip."""

    def __init__(self, actor, critic, rule: str) -> None:
        if rule not in {"PILOT_J", "PILOT_S"}:
            raise RuntimeError(f"unknown pilot rule {rule}")
        self.actor = actor
        self.critic = critic
        self.rule = rule
        self.optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=float(PPO.learning_rate))
        self.actor_count = sum(1 for _ in actor.parameters())

    def update(self, storage: dict[str, np.ndarray], *, ppo_update: int, before_clip=None, step_sink=None, model_seed=None) -> tuple[dict, list[dict]]:
        steps, envs = storage["rewards"].shape
        advantages, returns = compute_gae(
            storage["rewards"],
            storage["values"],
            storage["dones"],
            storage["last_values"],
            float(PPO.gamma),
            float(PPO.gae_lambda),
        )
        count = steps * envs
        minibatch = count // int(PPO.num_minibatches)
        if minibatch < 1 or count % int(PPO.num_minibatches):
            raise ValueError("MAPPO batch cannot fill registered minibatches")

        def flat(key):
            value = storage[key]
            return value.reshape((count,) + value.shape[2:])

        arrays = {key: flat(key) for key in ("self", "task", "communication", "neighbor_set", "d_nearest", "priority", "global_state", "pre_tanh", "log_prob_uav", "policy_active_mask")}
        adv = (advantages.reshape(-1) - advantages.mean()) / max(float(advantages.std()), 1.0e-6)
        ret = returns.reshape(-1)
        records = []
        steps_out = []
        self.actor.train()
        self.critic.train()
        optimizer_step = 0
        for epoch in range(int(PPO.update_epochs)):
            order = np.random.permutation(count)
            for start in range(0, count, minibatch):
                index = order[start : start + minibatch]
                batch = {key: torch.as_tensor(arrays[key][index], dtype=torch.float32) for key in arrays}
                advantage = torch.as_tensor(adv[index], dtype=torch.float32).unsqueeze(-1)
                target = torch.as_tensor(ret[index], dtype=torch.float32)
                mu, log_std = self.actor(batch["self"], batch["task"], batch["communication"], batch["neighbor_set"], batch["d_nearest"], batch["priority"])
                log_prob, entropy = tanh_log_prob_uav(mu, log_std, batch["pre_tanh"])
                value = self.critic(batch["global_state"])
                old = batch["log_prob_uav"]
                mask = batch["policy_active_mask"]
                ratio = (log_prob - old).exp()
                clipped = ratio.clamp(1.0 - float(PPO.clip_eps), 1.0 + float(PPO.clip_eps))
                per_uav = -torch.minimum(ratio * advantage, clipped * advantage)
                denom = mask.sum().clamp_min(1.0)
                policy_loss = (per_uav * mask).sum() / denom
                value_loss = 0.5 * (target - value).square().mean()
                entropy_term = (entropy * mask).sum() / denom
                loss = policy_loss + float(PPO.value_coef) * value_loss - float(PPO.entropy_coef) * entropy_term
                if not bool(torch.isfinite(loss).all()):
                    raise RuntimeError("NONFINITE")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if before_clip is not None and int(ppo_update) in SHADOW_UPDATES and epoch == 0 and start == 0:
                    before_clip(
                        {
                            "actor": self.actor,
                            "critic": self.critic,
                            "optimizer": self.optimizer,
                            "batch": batch,
                            "ppo_update": int(ppo_update),
                            "epoch": int(epoch),
                            "minibatch": 0,
                        }
                    )
                actor_before = _snapshot(self.actor)
                critic_before = _snapshot(self.critic)
                weight_before = dict(self.actor.named_parameters())["body.0.weight"].detach().cpu().clone()
                clipped_info = apply_clip(self.actor, self.critic, self.rule)
                self.optimizer.step()
                optimizer_step += 1
                kl = ((old - log_prob) * mask).sum() / denom
                clip_fraction = (((ratio - 1.0).abs() > float(PPO.clip_eps)).float() * mask).sum() / denom
                row = {
                    "ppo_update": int(ppo_update),
                    "epoch": int(epoch),
                    "minibatch": int(start // minibatch),
                    "optimizer_step": int(optimizer_step),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy_term.detach().cpu()),
                    "approx_kl": float(kl.detach().cpu()),
                    "clip_fraction": float(clip_fraction.detach().cpu()),
                    "action_std": float(log_std.exp().mean().detach().cpu()),
                    "actor_delta_norm": _delta_norm(actor_before, self.actor),
                    "critic_delta_norm": _delta_norm(critic_before, self.critic),
                    "new_column_delta_norm": _new_column_delta(weight_before, self.actor),
                    **clipped_info,
                    **_moment_summary(self.optimizer, self.actor_count),
                }
                if not all(np.isfinite(value) for key, value in row.items() if isinstance(value, float)):
                    raise RuntimeError("NONFINITE")
                if model_seed is not None:
                    row["model_seed"] = int(model_seed)
                row["variant"] = self.rule
                records.append(row)
                steps_out.append(row)
                if step_sink is not None:
                    step_sink.append(row)
        explained = 1.0 - float(np.var(ret - storage["values"].reshape(-1)) / max(float(np.var(ret)), 1.0e-8))
        if not np.isfinite(explained):
            raise RuntimeError("NONFINITE")
        summary = {key: float(np.mean([item[key] for item in records])) for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "action_std", "actor_delta_norm", "critic_delta_norm", "joint_pre_norm", "joint_post_norm")}
        summary["explained_variance"] = float(explained)
        summary["advantage_mean"] = float(advantages.mean())
        summary["advantage_std"] = float(advantages.std())
        summary["optimizer_steps"] = int(optimizer_step)
        return summary, steps_out
