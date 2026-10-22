"""Discrete high-level PPO with optional graph edges. Does not edit round30/ppo.py."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch.distributions import Categorical

from wt.htw_phase63.high_ppo import assignment_to_indices, indices_to_assignment
from wt.l2a.env import project_exclusive
from wt.round30.ppo import PPOConfig, compute_gae


def _tensor(features: Mapping[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    batch = {}
    for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge"):
        if key not in features:
            continue
        dtype = torch.bool if key.endswith("mask") else torch.float32
        batch[key] = torch.as_tensor(features[key], dtype=dtype, device=device)
    return batch


class GraphDiscretePPOAgent:
    def __init__(self, model, config: PPOConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def _forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            batch["uav"],
            batch["buoy"],
            batch["uav_mask"],
            batch["buoy_mask"],
            batch["global"],
            batch.get("edge"),
        )

    def act(self, features: Mapping[str, np.ndarray], *, deterministic: bool = False) -> dict[str, np.ndarray]:
        batch = _tensor(features, self.config.device)
        self.model.eval()
        with torch.no_grad():
            logits, value = self._forward(batch)
            dist = Categorical(logits=logits)
            z = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        z_np = z.cpu().numpy().astype(np.int64)
        positions = np.asarray(features["_uav_positions"], dtype=np.float64)
        buoys = np.asarray(features["_buoy_positions"], dtype=np.float64)
        executed = np.zeros_like(z_np)
        for env_index in range(z_np.shape[0]):
            assignment = project_exclusive(indices_to_assignment(z_np[env_index]), positions[env_index], buoys[env_index])
            executed[env_index] = assignment_to_indices(assignment)
        z_exec = torch.as_tensor(executed, dtype=torch.int64, device=self.config.device)
        with torch.no_grad():
            log_prob = dist.log_prob(z_exec).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        return {
            "z": executed.astype(np.int64),
            "log_prob": log_prob.cpu().numpy().astype(np.float32),
            "value": value.cpu().numpy().astype(np.float32),
            "entropy": entropy.cpu().numpy().astype(np.float32),
        }

    def update(self, storage: Mapping[str, np.ndarray]) -> dict[str, float]:
        steps, envs = storage["rewards"].shape
        advantages, returns = compute_gae(storage["rewards"], storage["values"], storage["dones"], storage["last_values"], self.config.gamma, self.config.gae_lambda)
        skip = {"last_values"}
        flat = {key: value.reshape((steps * envs,) + value.shape[2:]) if value.ndim >= 2 else value.reshape(steps * envs) for key, value in storage.items() if key not in skip}
        flat["advantages"], flat["returns"] = advantages.reshape(-1), returns.reshape(-1)
        flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / max(float(flat["advantages"].std()), 1.0e-6)
        count, minibatch = steps * envs, steps * envs // self.config.num_minibatches
        records: list[dict[str, float]] = []
        self.model.train()
        keys = ["uav", "buoy", "uav_mask", "buoy_mask", "global"]
        if "edge" in flat:
            keys.append("edge")
        for _epoch in range(self.config.update_epochs):
            order = np.random.permutation(count)
            for start in range(0, count, minibatch):
                index = order[start : start + minibatch]
                features = _tensor({key: flat[key][index] for key in keys}, self.config.device)
                old_log_prob = torch.as_tensor(flat["log_prob"][index], dtype=torch.float32, device=self.config.device)
                z = torch.as_tensor(flat["z"][index], dtype=torch.int64, device=self.config.device)
                advantage = torch.as_tensor(flat["advantages"][index], dtype=torch.float32, device=self.config.device)
                target = torch.as_tensor(flat["returns"][index], dtype=torch.float32, device=self.config.device)
                logits, value = self._forward(features)
                dist = Categorical(logits=logits)
                log_prob = dist.log_prob(z).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                ratio = (log_prob - old_log_prob).exp()
                clipped = ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
                policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                value_loss = 0.5 * (target - value).square().mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm).detach().cpu())
                self.optimizer.step()
                records.append(
                    {
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "entropy": float(entropy.mean().detach().cpu()),
                        "approx_kl": float((old_log_prob - log_prob).mean().detach().cpu()),
                        "clip_fraction": float(((ratio - 1.0).abs() > self.config.clip_eps).float().mean().detach().cpu()),
                        "gradient_norm_preclip": grad_norm,
                    }
                )
        predicted, target_np = storage["values"].reshape(-1), returns.reshape(-1)
        explained = 1.0 - float(np.var(target_np - predicted) / max(np.var(target_np), 1.0e-8))
        return {key: float(np.mean([row[key] for row in records])) for key in records[0]} | {
            "explained_variance": explained,
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
        }


__all__ = ["GraphDiscretePPOAgent"]
