"""PPO helpers moved out of the historical DeepSets baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def tanh_log_prob(mu: Tensor, log_std: Tensor, pre_tanh: Tensor) -> tuple[Tensor, Tensor]:
    normal, action = Normal(mu, log_std.exp()), torch.tanh(pre_tanh)
    correction = torch.log((1.0 - action.square()).clamp_min(1.0e-6))
    return (normal.log_prob(pre_tanh) - correction).sum(dim=(-1, -2)), normal.entropy().sum(dim=(-1, -2))


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    if rewards.shape != values.shape or rewards.shape != dones.shape or last_values.shape != rewards.shape[1:]:
        raise ValueError("GAE inputs have inconsistent shapes")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_advantage = np.zeros_like(last_values, dtype=np.float32)
    next_value = last_values.astype(np.float32)
    for index in range(rewards.shape[0] - 1, -1, -1):
        nonterminal = 1.0 - dones[index].astype(np.float32)
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages, advantages + values