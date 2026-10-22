"""Discrete high-level option policy. Uses Round-30 GAE/clip constants via import; does not edit ppo.py."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from wt.htw_phase61.features import BUOY_DIM, GLOBAL_DIM, IDLE_INDEX, MAX_UAVS, OPTION_DIM, UAV_DIM
from wt.l2a.env import project_exclusive
from wt.round30.models import CentralDeepSetsCritic, DeepSetsPPOShape, masked_mean
from wt.round30.ppo import PPOConfig, compute_gae


def _mlp(inputs: int, outputs: int, width: int = 128) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))


def assignment_to_indices(assignment: np.ndarray) -> np.ndarray:
    indices = np.full(MAX_UAVS, IDLE_INDEX, dtype=np.int64)
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if uav >= MAX_UAVS:
            break
        indices[uav] = IDLE_INDEX if int(buoy) < 0 else int(buoy)
    return indices


def indices_to_assignment(indices: np.ndarray) -> np.ndarray:
    assignment = np.full(MAX_UAVS, -1, dtype=np.int64)
    for uav, index in enumerate(np.asarray(indices, dtype=np.int64)):
        if uav >= MAX_UAVS:
            break
        assignment[uav] = -1 if int(index) >= IDLE_INDEX else int(index)
    return assignment


class HighLevelOptionModel(nn.Module):
    """π_H(z|s): per-UAV categorical over buoy+idle. No Task-Waypoint, no HGNN."""

    def __init__(self) -> None:
        super().__init__()
        self.shape = DeepSetsPPOShape(uav_dim=UAV_DIM, buoy_dim=BUOY_DIM, global_dim=GLOBAL_DIM, max_uavs=MAX_UAVS)
        self.uav_encoder = _mlp(self.shape.uav_dim, self.shape.width)
        self.buoy_encoder = _mlp(self.shape.buoy_dim, self.shape.width)
        self.actor_head = _mlp(3 * self.shape.width + self.shape.global_dim, OPTION_DIM, self.shape.width)
        self.critic = CentralDeepSetsCritic(self.shape)

    def forward(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor) -> tuple[Tensor, Tensor]:
        h_u, h_b = self.uav_encoder(uav), self.buoy_encoder(buoy)
        g_u, g_b = masked_mean(h_u, uav_mask, 1), masked_mean(h_b, buoy_mask, 1)
        count = uav.shape[1]
        token = torch.cat((h_u, g_u[:, None].expand(-1, count, -1), g_b[:, None].expand(-1, count, -1), global_[:, None].expand(-1, count, -1)), dim=-1)
        logits = self.actor_head(token)
        logits = logits.masked_fill(~uav_mask.unsqueeze(-1), -1.0e9)
        value = self.critic(uav, buoy, uav_mask, buoy_mask, global_)
        return logits, value


def _tensor(features: Mapping[str, np.ndarray], device: str) -> dict[str, Tensor]:
    keys = ("uav", "buoy", "uav_mask", "buoy_mask", "global")
    return {key: torch.as_tensor(features[key], dtype=torch.bool if key.endswith("mask") else torch.float32, device=device) for key in keys}


class DiscretePPOAgent:
    """Clipped PPO for discrete z. Optimizer/loss/GAE match Round-30 formulas without editing ppo.py."""

    def __init__(self, model: HighLevelOptionModel, config: PPOConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def act(self, features: Mapping[str, np.ndarray], *, deterministic: bool = False) -> dict[str, np.ndarray]:
        batch = _tensor(features, self.config.device)
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(batch["uav"], batch["buoy"], batch["uav_mask"], batch["buoy_mask"], batch["global"])
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
        flat = {key: value.reshape((steps * envs,) + value.shape[2:]) if value.ndim >= 2 else value.reshape(steps * envs) for key, value in storage.items() if key not in {"last_values"}}
        flat["advantages"], flat["returns"] = advantages.reshape(-1), returns.reshape(-1)
        flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / max(float(flat["advantages"].std()), 1.0e-6)
        count, minibatch = steps * envs, steps * envs // self.config.num_minibatches
        if minibatch < 1 or count % self.config.num_minibatches:
            raise ValueError("high-level PPO batch cannot be divided into the registered minibatches")
        records: list[dict[str, float]] = []
        self.model.train()
        for _epoch in range(self.config.update_epochs):
            order = np.random.permutation(count)
            for start in range(0, count, minibatch):
                index = order[start : start + minibatch]
                features = _tensor({key: flat[key][index] for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global")}, self.config.device)
                old_log_prob = torch.as_tensor(flat["log_prob"][index], dtype=torch.float32, device=self.config.device)
                z = torch.as_tensor(flat["z"][index], dtype=torch.int64, device=self.config.device)
                advantage = torch.as_tensor(flat["advantages"][index], dtype=torch.float32, device=self.config.device)
                target = torch.as_tensor(flat["returns"][index], dtype=torch.float32, device=self.config.device)
                logits, value = self.model(features["uav"], features["buoy"], features["uav_mask"], features["buoy_mask"], features["global"])
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
        predicted, target = storage["values"].reshape(-1), returns.reshape(-1)
        explained = 1.0 - float(np.var(target - predicted) / max(np.var(target), 1.0e-8))
        if not np.isfinite(explained) or not all(np.isfinite(value) for row in records for value in row.values()):
            raise RuntimeError("high-level PPO update produced NaN/Inf")
        return {key: float(np.mean([row[key] for row in records])) for key in records[0]} | {
            "explained_variance": explained,
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
        }


__all__ = [
    "DiscretePPOAgent",
    "HighLevelOptionModel",
    "assignment_to_indices",
    "indices_to_assignment",
]
