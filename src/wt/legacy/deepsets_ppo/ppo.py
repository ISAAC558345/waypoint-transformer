"""Small, auditable standard PPO implementation for the Round-30 baseline."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal

from wt.algorithms.ppo_common import PPOConfig, compute_gae, seed_everything, tanh_log_prob

from .models import DeepSetsActorCritic


def _tensor(features: Mapping[str, np.ndarray], device: str) -> dict[str, Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.bool if key.endswith("mask") else torch.float32, device=device) for key, value in features.items()}


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