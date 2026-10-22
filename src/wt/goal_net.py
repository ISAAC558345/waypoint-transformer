"""
Goal and reward waypoint network implementation.

By Anonymous Authors
"""

import gym
from gym import spaces
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F

from wt import dataset, layers, step, util

class KForwardGoalNetwork(pl.LightningModule):
    def __init__(self, obs_dim, goal_dim, hidden_dim, max_T, recurrent = False,
                learning_rate=1e-3, batch_size=1024, reward = False,
                reward_representation=None,
                reward_normalization_const=None,
                reward_episode_horizon=None):
        super().__init__()
        if obs_dim <= 0 or goal_dim <= 0 or hidden_dim <= 0:
            raise ValueError("obs_dim, goal_dim, and hidden_dim must be positive.")
        if max_T is None or int(max_T) < 0:
            raise ValueError("max_T must be a non-negative integer.")
        self.recurrent = recurrent 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.goal_dim = int(goal_dim)
        self.max_T = int(max_T)
        self.reward = bool(reward)
        if self.reward:
            if reward_representation is None:
                raise ValueError(
                    "Reward waypoint checkpoints require an explicit "
                    "reward_representation."
                )
            self.reward_representation = dataset.resolve_reward_representation(
                reward_representation
            )
            expected_goal_dim = (
                2 if self.reward_representation == "both" else 1
            )
            if self.goal_dim != expected_goal_dim:
                raise ValueError(
                    "Reward waypoint goal_dim conflicts with "
                    f"reward_representation: {self.goal_dim} != "
                    f"{expected_goal_dim}."
                )
            if reward_normalization_const is None:
                raise ValueError(
                    "Reward waypoint checkpoints require the training-split "
                    "reward_normalization_const."
                )
            self.reward_normalization_const = float(
                reward_normalization_const
            )
            if (
                not np.isfinite(self.reward_normalization_const)
                or self.reward_normalization_const == 0
            ):
                raise ValueError(
                    "reward_normalization_const must be finite and non-zero."
                )
            if reward_episode_horizon is None:
                raise ValueError(
                    "Reward waypoint checkpoints require "
                    "reward_episode_horizon."
                )
            self.reward_episode_horizon = int(reward_episode_horizon)
            if self.reward_episode_horizon <= 0:
                raise ValueError(
                    "reward_episode_horizon must be positive."
                )
        else:
            self.reward_representation = None
            self.reward_normalization_const = None
            self.reward_episode_horizon = None
        self.save_hyperparameters(
            {
                "obs_dim": self.obs_dim,
                "goal_dim": self.goal_dim,
                "hidden_dim": self.hidden_dim,
                "max_T": self.max_T,
                "recurrent": self.recurrent,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "reward": self.reward,
                "reward_representation": self.reward_representation,
                "reward_normalization_const": (
                    self.reward_normalization_const
                ),
                "reward_episode_horizon": self.reward_episode_horizon,
            }
        )

        self.net = nn.Sequential(nn.Linear(self.obs_dim + self.goal_dim, self.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.goal_dim))

    def forward(self, obs_goal):
        if self.reward:
            return obs_goal[..., -self.goal_dim:] + self.net(obs_goal)

        return obs_goal[..., :self.goal_dim] + self.net(obs_goal)

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared error over valid timesteps only."""
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: {prediction.shape} != {target.shape}"
            )
        valid = mask.to(device=prediction.device, dtype=torch.bool)
        if valid.ndim == prediction.ndim and valid.shape[-1] == 1:
            valid = valid.squeeze(-1)
        if valid.shape != prediction.shape[:-1]:
            raise ValueError(
                "mask must match the batch/time dimensions of prediction: "
                f"{valid.shape} != {prediction.shape[:-1]}"
            )

        per_timestep_loss = F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
        if not torch.any(valid):
            # Keep the zero connected to the model graph so backward remains valid.
            return prediction.sum() * 0.0
        return per_timestep_loss[valid].mean()

    def training_step(
        self,
        batch,
        batch_idx: int,
        log_prefix: str = "train",
    ) -> torch.Tensor:
        """Compute waypoint loss against episode-safe dataset labels.

        The dataset owns horizon alignment for both goal and reward waypoints,
        including terminal clamping. This keeps the model loss independent of
        window padding and guarantees that every reduction sees the timestep
        mask before averaging.
        """
        obs_goal, waypoint, mask = batch
        loss = self._masked_mse(self(obs_goal), waypoint, mask)

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ) -> torch.Tensor:
        """Computes loss for a validation batch."""
        loss = self.training_step(batch, batch_idx, log_prefix="val")
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        """Configures the optimizer used by PyTorch Lightning."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class Manager:
    def __init__(self, global_goal, epsilon = 0.01, K = 20, goal_columns = (0, 1)):
        self.global_goal = global_goal
        self.goal_columns = goal_columns
        self.epsilon, self.K = epsilon, K
        self._observations = []
        self._actions = []

    def update_obs(self, obs):
        self._observations.append(obs)
        self._observations = self._observations[-self.K:]

    def update_act(self, act):
        self._actions.append(act)
        history_length = max(self.K - 1, 0)
        self._actions = (
            self._actions[-history_length:] if history_length else []
        )

    @property
    def goal(self):
        if self._stuttering(self._observations):
            return self.global_goal #proposed_goal.detach().numpy()[0]
        return self.global_goal
        
    @property
    def actions(self):
        return None if not self._actions else self._actions + [np.zeros_like(self._actions[-1])]

    @property
    def observations(self):
        if self._stuttering(self._observations):
            return self._observations# [-1:]
        return self._observations

    def _stuttering(self, observations):
        if len(observations) < self.K:
            return False
        observations = np.concatenate([o[None, self.goal_columns] for o in observations], axis = 0)
        velocity = np.linalg.norm(observations[1:] - observations[:-1], axis = -1)
        return np.mean(velocity) <= self.epsilon

class Manager:
    def __init__(self, global_goal, epsilon = 0.01, K = 20, goal_columns = (0, 1), goal_append = False):
        self.global_goal = self.original_goal = global_goal
        self.goal_columns = goal_columns
        self.epsilon, self.K = epsilon, K
        self._observations = []
        self._actions = []
        self._goals = []
        self.goal_append = goal_append
        self.t = 0

    def update_obs(self, obs):
        self._observations.append(obs)
        self._observations = self._observations[-self.K:]
        self.t += 1

    def update_goal(self, goal):
        if self.goal_append:
            self._goals.append(goal)
            self._goals = self._goals[-self.K:]
        else:
            self.global_goal = goal

    def update_act(self, act):
        self._actions.append(act)
        history_length = max(self.K - 1, 0)
        self._actions = (
            self._actions[-history_length:] if history_length else []
        )

    def step_goal(self, lambd, thres):
        assert not self.goal_append
        # self.global_goal = thres / (1 + np.exp(-lambd * (self.t - 60))) + self.original_goal
        self.global_goal = np.minimum(lambd * self.global_goal, thres)
        #self.global_goal = np.minimum(lambd + self.global_goal, thres)

    @property
    def goal(self):
        if self.goal_append:
            return self._goals
        else:
            return [self.global_goal] * len(self._observations)
        
    @property
    def actions(self):
        return None if not self._actions else self._actions + [np.zeros_like(self._actions[-1])]

    @property
    def observations(self):
        if self._stuttering(self._observations):
            return self._observations# [-1:]
        return self._observations

    def _stuttering(self, observations):
        if len(observations) < self.K:
            return False
        observations = np.concatenate([o[None, self.goal_columns] for o in observations], axis = 0)
        velocity = np.linalg.norm(observations[1:] - observations[:-1], axis = -1)
        return np.mean(velocity) <= self.epsilon

class ManualGoalNetwork(nn.Module):
    LARGE_GOALS = torch.tensor([[12, 0], [12, 7], [0, 7], [4, 15], [0, 22], [20, 7], [20, 15], [20, 22], [12, 22], [12, 15], [20, 0],
                                [28, 0], [28, 7], [36, 0], [36, 7], [36, 15], [28, 15], [28, 22], [36, 24]])

    # LARGE_GOALS = torch.tensor([[0, 8], [12, 7], [20, 7], [20, 15], [28, 15], [28, 24], [36, 24]])

    def __init__(self, obs_dim, goal_dim, level = 'large', **unused):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.dummy_param = nn.Parameter(torch.zeros(1))
        if level == 'large':
            self.goals = ManualGoalNetwork.LARGE_GOALS.to(self.dummy_param.device)

    def forward(self, obs_goal):
        self.goals = self.goals.to(obs_goal.device)
        loc, global_goals = obs_goal[..., :self.goal_dim], obs_goal[..., -self.goal_dim:]
        goal_dist = torch.linalg.norm(loc.unsqueeze(-2) - self.goals.unsqueeze(0), dim = -1)
        sorted_idx = goal_dist.argsort(dim = -1)
        global_prox_cond = torch.linalg.norm(self.goals[sorted_idx] - global_goals.unsqueeze(-2), dim = -1) < torch.linalg.norm(loc - global_goals, dim = -1).unsqueeze(-1)
        selected = global_prox_cond.int().argmax(dim = -1)
        return self.goals[selected]

