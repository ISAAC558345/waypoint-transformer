"""Small, auditable standard PPO implementation for the Round-30 baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal

from .models import DeepSetsActorCritic


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    learning_rate: float = 3.0e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_length: int = 256
    num_envs: int = 8
    num_minibatches: int = 4
    update_epochs: int = 4
    checkpoint_interval_steps: int = 30720
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]: return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _tensor(features: Mapping[str, np.ndarray], device: str) -> dict[str, Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.bool if key.endswith("mask") else torch.float32, device=device) for key, value in features.items()}


def tanh_log_prob(mu: Tensor, log_std: Tensor, pre_tanh: Tensor) -> tuple[Tensor, Tensor]:
    normal, action = Normal(mu, log_std.exp()), torch.tanh(pre_tanh)
    correction = torch.log((1.0 - action.square()).clamp_min(1.0e-6))
    return (normal.log_prob(pre_tanh) - correction).sum(dim=(-1, -2)), normal.entropy().sum(dim=(-1, -2))


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_values: np.ndarray, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    if rewards.shape != values.shape or rewards.shape != dones.shape or last_values.shape != rewards.shape[1:]:
        raise ValueError("GAE inputs have inconsistent shapes")
    advantages = np.zeros_like(rewards, dtype=np.float32); next_advantage = np.zeros_like(last_values, dtype=np.float32)
    next_value = last_values.astype(np.float32)
    for index in range(rewards.shape[0] - 1, -1, -1):
        nonterminal = 1.0 - dones[index].astype(np.float32)
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages, advantages + values


class PPOAgent:
    def __init__(self, model: DeepSetsActorCritic, config: PPOConfig) -> None:
        self.model, self.config = model.to(config.device), config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def act(self, features: Mapping[str, np.ndarray], *, deterministic: bool = False) -> dict[str, np.ndarray]:
        batch = _tensor(features, self.config.device)
        self.model.eval()
        with torch.no_grad():
            mu, log_std, value = self.model(**batch)
            pre_tanh = mu if deterministic else Normal(mu, log_std.exp()).sample()
            action = torch.tanh(pre_tanh)
            log_prob, entropy = tanh_log_prob(mu, log_std, pre_tanh)
        return {"action": action.cpu().numpy().astype(np.float32), "pre_tanh": pre_tanh.cpu().numpy().astype(np.float32), "log_prob": log_prob.cpu().numpy().astype(np.float32), "value": value.cpu().numpy().astype(np.float32), "entropy": entropy.cpu().numpy().astype(np.float32), "action_std": log_std.exp().mean(dim=(-1, -2)).cpu().numpy().astype(np.float32)}

    def update(self, storage: Mapping[str, np.ndarray]) -> dict[str, float]:
        steps, envs = storage["rewards"].shape
        advantages, returns = compute_gae(storage["rewards"], storage["values"], storage["dones"], storage["last_values"], self.config.gamma, self.config.gae_lambda)
        flat = {key: value.reshape((steps * envs,) + value.shape[2:]) if value.ndim >= 2 else value.reshape(steps * envs) for key, value in storage.items() if key not in {"last_values"}}
        flat["advantages"], flat["returns"] = advantages.reshape(-1), returns.reshape(-1)
        flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / max(float(flat["advantages"].std()), 1.0e-6)
        count, minibatch = steps * envs, steps * envs // self.config.num_minibatches
        if minibatch < 1 or count % self.config.num_minibatches:
            raise ValueError("PPO batch cannot be divided into the registered minibatches")
        records: list[dict[str, float]] = []
        self.model.train()
        for epoch in range(self.config.update_epochs):
            order = np.random.permutation(count)
            for start in range(0, count, minibatch):
                index = order[start:start + minibatch]
                features = _tensor({key: flat[key][index] for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global")}, self.config.device)
                old_log_prob = torch.as_tensor(flat["log_prob"][index], dtype=torch.float32, device=self.config.device)
                action_z = torch.as_tensor(flat["pre_tanh"][index], dtype=torch.float32, device=self.config.device)
                advantage = torch.as_tensor(flat["advantages"][index], dtype=torch.float32, device=self.config.device)
                target = torch.as_tensor(flat["returns"][index], dtype=torch.float32, device=self.config.device)
                mu, log_std, value = self.model(**features)
                log_prob, entropy = tanh_log_prob(mu, log_std, action_z)
                ratio = (log_prob - old_log_prob).exp()
                clipped = ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
                policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                value_loss = 0.5 * (target - value).square().mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean()
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm).detach().cpu())
                self.optimizer.step()
                records.append({"policy_loss": float(policy_loss.detach().cpu()), "value_loss": float(value_loss.detach().cpu()), "entropy": float(entropy.mean().detach().cpu()), "approx_kl": float((old_log_prob - log_prob).mean().detach().cpu()), "clip_fraction": float(((ratio - 1.0).abs() > self.config.clip_eps).float().mean().detach().cpu()), "gradient_norm_preclip": grad_norm, "action_std": float(log_std.exp().mean().detach().cpu())})
        predicted, target = storage["values"].reshape(-1), returns.reshape(-1)
        explained = 1.0 - float(np.var(target - predicted) / max(np.var(target), 1.0e-8))
        if not np.isfinite(explained) or not all(np.isfinite(value) for row in records for value in row.values()):
            raise RuntimeError("PPO update produced NaN/Inf")
        return {key: float(np.mean([row[key] for row in records])) for key in records[0]} | {"explained_variance": explained, "advantage_mean": float(advantages.mean()), "advantage_std": float(advantages.std())}

    def checkpoint(self, *, seed: int, steps: int, extra: Mapping[str, Any]) -> dict[str, Any]:
        return {"seed": seed, "environment_steps": steps, "ppo_config": self.config.to_dict(), "model_shape": self.model.shape.to_dict(), "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(), "extra": dict(extra), "uses_task_waypoint": False, "uses_planner_trigger": False, "uses_hgnn": False, "uses_transformer": False, "raw_logprob_semantics": "tanh-squashed policy sample before environment safety projection"}


def load_agent(payload: Mapping[str, Any], config: PPOConfig, model: DeepSetsActorCritic) -> PPOAgent:
    agent = PPOAgent(model, config); agent.model.load_state_dict(payload["model_state_dict"]); agent.model.eval(); return agent


__all__ = ["PPOAgent", "PPOConfig", "compute_gae", "load_agent", "seed_everything", "tanh_log_prob"]
