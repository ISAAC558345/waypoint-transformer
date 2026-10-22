"""
Builds the data modules used to train the policy.

Sources:
    - RvS GitHub repository (https://github.com/scottemmons/rvs)
    - DT GitHub repository (https://github.com/kzl/decision-transformer)
    - Edits by Anonymous Authors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
import random
from typing import Dict, Iterator, List, Optional, Tuple, Union

from d4rl import offline_env
import gym
import numpy as np
import pytorch_lightning as pl
import torch
import collections
from collections import defaultdict
from torch.utils import data

from wt import step, util

max_num_workers = 16


def create_data_module(
    env: gym.Env,
    env_name: str,
    rollout_directory: str,
    batch_size: int = 256,
    val_frac: float = 0.1,
    unconditional_policy: bool = False,
    reward_conditioning: bool = False,
    average_reward_to_go: bool = True,
    reward_representation: Optional[str] = None,
    seed: Optional[int] = None,
    K = 20,
    train_goal_net = False,
    waypoint_horizon: int = 30,
    reward_episode_horizon: Optional[int] = None,
) -> AbstractDataModule:
    """Creates the data module used for training."""
    if unconditional_policy and reward_conditioning:
        raise ValueError("Cannot condition on reward with an unconditional policy.")

    if env_name in step.d4rl_env_names:
        if unconditional_policy:
            data_module = D4RLBCDataModule(
                env,
                batch_size=batch_size,
                val_frac=val_frac,
                seed=seed,
            )
        elif reward_conditioning:
            data_module = D4RLRvSRDataModule(
                env,
                batch_size=batch_size,
                val_frac=val_frac,
                average_reward_to_go=average_reward_to_go,
                reward_representation=reward_representation,
                seed=seed,
                train_goal_net = train_goal_net,
                K = K,
                waypoint_horizon=waypoint_horizon,
                reward_episode_horizon=reward_episode_horizon,
            )
        else:
            data_module = D4RLRvSGDataModule(
                env,
                batch_size=batch_size,
                seed=seed,
                train_goal_net = train_goal_net,
                val_frac=val_frac,
                K = K,
                waypoint_horizon=waypoint_horizon,
            )
    else:
        if unconditional_policy:
            raise NotImplementedError
        else:
            data_module = GCSLDataModule(
                rollout_directory,
                batch_size=batch_size,
                val_frac=val_frac,
                seed=seed,
                num_workers=os.cpu_count(),
            )

    return data_module

def s_g_pair_iter(
    s_obs_vecs: np.ndarray,
    s_ach_goal_vecs: np.ndarray,
    a_vecs: np.ndarary,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Use hindsight to iterate over all states and future achieved goals."""
    for episode in range(len(a_vecs)):
        s_obs_vec = s_obs_vecs[episode]
        s_ach_goal_vec = s_ach_goal_vecs[episode]
        a_vec = a_vecs[episode]
        for i in range(len(a_vec)):
            for j in range(i + 1, len(s_obs_vec)):
                s = s_obs_vec[i]
                a = a_vec[i]
                g = s_ach_goal_vec[j]
                yield s, a, g


def make_s_g_tensor(states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    """Combine observations and goals into the same tensor."""
    s_tensor = torch.tensor(states)
    g_tensor = torch.tensor(goals)
    s_g_tensor = torch.cat((s_tensor, g_tensor), dim=-1)

    return s_g_tensor


def to_tensor_dataset(
    data_vec: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> data.TensorDataset:
    """Convert a list of data into a tensor dataset."""
    states, goals, actions = zip(*data_vec)
    states = np.array(states)
    goals = np.array(goals)

    s_g_tensor = make_s_g_tensor(states, goals)
    a_tensor = torch.tensor(actions)

    assert not s_g_tensor.requires_grad
    assert not a_tensor.requires_grad

    return data.TensorDataset(s_g_tensor, a_tensor)


def load_tensor_dataset(rollout_directory: str) -> data.TensorDataset:
    """Load saved GCSL rollouts and return a tensor dataset."""
    s_obs_vecs = np.load(os.path.join(rollout_directory, step.s_obs_vecs_file))
    s_ach_goal_vecs = np.load(
        os.path.join(rollout_directory, step.s_ach_goal_vecs_file),
    )
    a_vecs = np.load(os.path.join(rollout_directory, step.a_vecs_file))

    data_vec = [
        (s, g, a) for s, a, g in s_g_pair_iter(s_obs_vecs, s_ach_goal_vecs, a_vecs)
    ]
    tensor_dataset = to_tensor_dataset(data_vec)

    return tensor_dataset


def seed_worker(worker_id: int) -> None:
    """Unique random seed for each parallel data worker to prevent duplicate batches."""
    # torch.initial_seed() is the base torch seed plus a unique offset for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _environment_flag(name: str) -> bool:
    """Parse a legacy boolean environment variable without truthy-string bugs."""
    value = os.environ.get(name)
    return (
        value is not None
        and value.strip().lower() not in ("", "0", "false", "no", "off")
    )


def environment_episode_horizon(env: gym.Env) -> Optional[int]:
    """Read an episode horizon from Gym without depending on one wrapper type."""
    horizon = getattr(env, "_max_episode_steps", None)
    if horizon is None:
        horizon = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    if horizon is None:
        return None
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("Environment episode horizon must be positive.")
    return horizon


def resolve_reward_episode_horizon(
    explicit_horizon: Optional[int] = None,
    checkpoint_horizon: Optional[int] = None,
    environment_horizon: Optional[int] = None,
) -> int:
    """Resolve reward horizon metadata and reject silent distribution shifts."""
    horizons = {
        name: None if value is None else int(value)
        for name, value in {
            "explicit": explicit_horizon,
            "checkpoint": checkpoint_horizon,
            "environment": environment_horizon,
        }.items()
    }
    for name, value in horizons.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name} reward episode horizon must be positive.")
    if (
        horizons["explicit"] is not None
        and horizons["checkpoint"] is not None
        and horizons["explicit"] != horizons["checkpoint"]
    ):
        raise ValueError(
            "Explicit reward_episode_horizon conflicts with the policy "
            f"checkpoint: {horizons['explicit']} != "
            f"{horizons['checkpoint']}."
        )
    if (
        horizons["checkpoint"] is not None
        and horizons["environment"] is not None
        and horizons["checkpoint"] != horizons["environment"]
    ):
        raise ValueError(
            "Policy reward_episode_horizon conflicts with the evaluation "
            f"environment: {horizons['checkpoint']} != "
            f"{horizons['environment']}."
        )
    if (
        horizons["explicit"] is not None
        and horizons["environment"] is not None
        and horizons["explicit"] != horizons["environment"]
    ):
        raise ValueError(
            "Explicit reward_episode_horizon conflicts with the evaluation "
            f"environment: {horizons['explicit']} != "
            f"{horizons['environment']}."
        )
    resolved = (
        horizons["explicit"]
        if horizons["explicit"] is not None
        else horizons["checkpoint"]
    )
    if resolved is None:
        resolved = horizons["environment"]
    if resolved is None:
        raise ValueError(
            "Reward-conditioned evaluation needs reward_episode_horizon "
            "metadata or an environment TimeLimit."
        )
    return resolved


class D4RLIterableDataset(data.IterableDataset):
    """Used for goal-conditioned learning in D4RL."""

    def __init__(
        self,
        trajectories: List[Dict],
        epoch_size: int = 1000000,
        index_batch_size: int = 64,
        goal_columns: Optional[Union[Tuple[int], List[int], np.ndarray]] = None,
        config: Optional[Dict] = None,
        augment: bool = True,
        normalization_const: Optional[float] = None,
        **kwargs
    ):
        """Initializes the dataset.

        Args:
            observations: The observations for the dataset.
            actions: The actions for the dataset.
            dones: The dones for the dataset.
            epoch_size: For PyTorch Lightning to count epochs.
            index_batch_size: This has no effect on the functionality of the dataset,
                but it is used internally as the batch size to fetch random indices.
            goal_columns: If not None, then only use these columns of the
                observation_space for the goal conditioning.
        """
        super().__init__()

        if len(trajectories) == 0:
            raise ValueError("D4RLIterableDataset requires at least one trajectory.")
        self.trajectories = trajectories
        self.epoch_size = epoch_size
        self.index_batch_size = index_batch_size
        self.goal_columns = goal_columns
        self.augment = augment
        self.config = config if config is not None else defaultdict(bool)
        self.config.update(kwargs)
        if "waypoint_horizon" not in self.config:
            self.config["waypoint_horizon"] = 30
        if int(self.config["waypoint_horizon"]) < 0:
            raise ValueError("waypoint_horizon must be non-negative.")
        self.lengths = np.array([len(self.trajectories[i]['observations']) for i in range(len(trajectories))])
        if np.any(self.lengths <= 0):
            raise ValueError("D4RLIterableDataset trajectories must be non-empty.")
        configured_episode_horizon = self.config.get(
            "reward_episode_horizon",
            self.config.get("max_ep_len"),
        )
        self.reward_episode_horizon = (
            int(self.lengths.max())
            if configured_episode_horizon is None
            else int(configured_episode_horizon)
        )
        if self.reward_episode_horizon <= 0:
            raise ValueError("reward_episode_horizon must be positive.")
        if np.any(self.lengths > self.reward_episode_horizon):
            raise ValueError(
                "reward_episode_horizon cannot be shorter than a trajectory: "
                f"{self.reward_episode_horizon} < {int(self.lengths.max())}."
            )
        self.rewards = np.array([sum(self.trajectories[i]['rewards']) for i in range(len(trajectories))])
        self.normalization_const = (
            round(self.rewards.max(), -3)
            if normalization_const is None
            else float(normalization_const)
        )
        if not np.isfinite(self.normalization_const):
            raise ValueError("normalization_const must be finite.")
        if self.normalization_const == 0:
            self.normalization_const = 1.0
        print("Normalization constant:", self.normalization_const)


    def _sample_indices(
        self,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Credit to Dibya Ghosh's GCSL codebase for the logic in the following block:
        # https://github.com/dibyaghosh/gcsl/blob/
        # cfae5609cee79e5a2228fb7653451023c41a64cb/gcsl/algo/buffer.py#L78
        sample_size = self.index_batch_size if batch_size is None else batch_size
        trajectory_indices = np.random.choice(len(self.trajectories), sample_size,
                                              p = self.lengths / sum(self.lengths))
        lengths = np.array([len(self.trajectories[i]['observations']) for i in trajectory_indices])
        proportional_indices_1 = np.random.rand(sample_size)
        proportional_indices_2 = np.random.rand(sample_size)
        time_indices_1 = np.floor(
            proportional_indices_1 * lengths,
        ).astype(int)
        time_indices_2 = np.floor(
            proportional_indices_2 * lengths,
        ).astype(int)

        start_indices = np.minimum(
            time_indices_1,
            time_indices_2,
        )
        goal_indices = np.maximum(
            time_indices_1,
            time_indices_2,
        )

        return trajectory_indices, start_indices, goal_indices

    @staticmethod
    def _left_pad(array: np.ndarray, target_length: int, value: float = 0) -> np.ndarray:
        """Left-pad a [1, T, ...] sequence to target_length."""
        pad_length = target_length - array.shape[1]
        if pad_length < 0:
            raise ValueError("Cannot pad a sequence longer than the target length.")
        if pad_length == 0:
            return array
        padding = np.full(
            (array.shape[0], pad_length, *array.shape[2:]),
            value,
            dtype=array.dtype,
        )
        return np.concatenate([padding, array], axis=1)

    def _fetch_trajectory_components(
        self,
        trajectory_indices: np.ndarray,
        start_indices: np.ndarray,
        goal_indices: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """Build a structured batch while keeping the legacy DataLoader tuple intact.

        Every returned sequence is left-padded to ``K``. ``goals`` contains the
        global condition and ``waypoints`` contains the achieved target
        ``waypoint_horizon`` steps in the future. Future indices are clamped to
        the final observation of the selected trajectory, so labels never cross
        an episode boundary.
        """
        K = int(self.config['K'])
        if K <= 0:
            raise ValueError("K must be positive.")
        waypoint_horizon = int(self.config["waypoint_horizon"])
        state_dim, act_dim = self.trajectories[0]['observations'].shape[-1], self.trajectories[0]['actions'].shape[-1]
        goal_dim = len(self.goal_columns) if self.goal_columns is not None else state_dim
        trajs = [self.trajectories[i] for i in trajectory_indices]
        states, actions, rewards, masks = [], [], [], []
        goals, waypoints = [], []

        for (si, gi), traj in zip(zip(start_indices, goal_indices), trajs):
            observations = np.asarray(traj['observations'])
            trajectory_actions = np.asarray(traj['actions'])
            trajectory_rewards = np.asarray(traj['rewards'])
            if not 0 <= si < len(observations):
                raise IndexError(f"start index {si} is outside a trajectory of length {len(observations)}.")
            if not 0 <= gi < len(observations):
                raise IndexError(f"goal index {gi} is outside a trajectory of length {len(observations)}.")

            state_sequence = observations[si:si + K].reshape(1, -1, state_dim)
            action_sequence = trajectory_actions[si:si + K].reshape(1, -1, act_dim)
            reward_sequence = trajectory_rewards[si:si + K].reshape(1, -1, 1)
            sequence_length = state_sequence.shape[1]
            future_indices = np.minimum(
                si + np.arange(sequence_length) + waypoint_horizon,
                len(observations) - 1,
            )

            if self.config['goal_conditioned']:
                global_goal = (
                    observations[gi]
                    if self.goal_columns is None
                    else observations[gi, self.goal_columns]
                )
                future_observations = observations[future_indices]
                future_goals = (
                    future_observations
                    if self.goal_columns is None
                    else future_observations[:, self.goal_columns]
                )
                goal_sequence = np.repeat(
                    global_goal.reshape(1, 1, goal_dim),
                    sequence_length,
                    axis=1,
                )
                waypoint_sequence = future_goals.reshape(1, sequence_length, goal_dim)
            elif self.config['reward_conditioned']:
                full_rtg = np.asarray(
                    discount_cumsum(
                        trajectory_rewards[si:],
                        normalize=self.normalization_const,
                        trim=len(trajectory_rewards) - si,
                        representation=self.config.get("reward_representation"),
                        remaining_horizon=self.reward_episode_horizon - si,
                    )
                )
                if full_rtg.ndim == 1:
                    full_rtg = full_rtg[:, None]
                reward_dim = full_rtg.shape[-1]
                goal_sequence = np.repeat(
                    full_rtg[0].reshape(1, 1, reward_dim),
                    sequence_length,
                    axis=1,
                )
                relative_future_indices = np.minimum(
                    np.arange(sequence_length) + waypoint_horizon,
                    len(full_rtg) - 1,
                )
                waypoint_sequence = full_rtg[relative_future_indices].reshape(
                    1,
                    sequence_length,
                    reward_dim,
                )
            else:
                raise ValueError("Dataset must be goal-conditioned or reward-conditioned.")

            states.append(self._left_pad(state_sequence, K))
            actions.append(self._left_pad(action_sequence, K, value=-10.0))
            rewards.append(self._left_pad(reward_sequence, K))
            goals.append(self._left_pad(goal_sequence, K))
            waypoints.append(self._left_pad(waypoint_sequence, K))
            masks.append(
                np.concatenate(
                    [np.zeros((1, K - sequence_length)), np.ones((1, sequence_length))],
                    axis=1,
                )
            )

        components = {
            "states": torch.from_numpy(np.concatenate(states, axis=0)).to(dtype=torch.float32),
            "actions": torch.from_numpy(np.concatenate(actions, axis=0)).to(dtype=torch.float32),
            "rewards": torch.from_numpy(np.concatenate(rewards, axis=0)).to(dtype=torch.float32),
            "goals": torch.from_numpy(np.concatenate(goals, axis=0)).to(dtype=torch.float32),
            "waypoints": torch.from_numpy(np.concatenate(waypoints, axis=0)).to(dtype=torch.float32),
            "mask": torch.from_numpy(np.concatenate(masks, axis=0)).to(dtype=torch.bool),
        }
        return components

    def sample_components(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Sample a structured batch for tests and data-pipeline inspection."""
        trajectory_indices, start_indices, goal_indices = self._sample_indices(batch_size)
        return self._fetch_trajectory_components(
            trajectory_indices,
            start_indices,
            goal_indices,
        )

    def _fetch_trajectories(self, trajectory_indices, start_indices, goal_indices):
        """Return the original four-tensor training tuple."""
        components = self._fetch_trajectory_components(
            trajectory_indices,
            start_indices,
            goal_indices,
        )
        if self.config.get('train_goal_net'):
            return (
                components["states"],
                components["waypoints"],
                components["goals"],
                components["mask"],
            )
        if self.config['goal_conditioned']:
            return (
                components["states"],
                components["actions"],
                components["goals"],
                components["mask"],
            )
        return (
            components["states"],
            components["actions"],
            components["goals"],
            components["mask"],
        )

    def _sample_batch(self, batch_size = None) -> Tuple:
        trajectory_indices, start_indices, goal_indices = self._sample_indices(batch_size)
        s, a, cond, mask = self._fetch_trajectories(trajectory_indices,
                                                                     start_indices,
                                                                     goal_indices)
        return s, a, cond, mask

    def __iter__(self) -> Iterator[Tuple[torch.tensor, torch.tensor]]:
        """Yield each training example."""
        examples_yielded = 0
        while examples_yielded < self.epoch_size:
            s, a, cond, mask = self._sample_batch()
            for i in range(s.shape[0]):
                yield torch.cat([s[i], cond[i]], dim = -1), a[i], mask[i]
                examples_yielded += 1
                if examples_yielded >= self.epoch_size:
                    break

    def __len__(self) -> int:
        """The number of examples in an epoch. Used by the trainer to count epochs."""
        return self.epoch_size

REWARD_REPRESENTATIONS = ("average", "cumulative", "both")


def resolve_reward_representation(
    representation: Optional[str] = None,
    *,
    average_reward_to_go: Optional[bool] = None,
) -> str:
    """Resolve reward channels with explicit configuration taking precedence."""
    if representation is not None:
        if representation not in REWARD_REPRESENTATIONS:
            raise ValueError(
                "reward_representation must be average, cumulative, or both."
            )
        return representation
    if average_reward_to_go is False:
        return "cumulative"

    average_flag = _environment_flag("AVG_REWARD")
    cumulative_flag = _environment_flag("CM_REWARD")
    if average_flag and cumulative_flag:
        raise ValueError(
            "AVG_REWARD and CM_REWARD cannot both be set; use "
            "reward_representation instead."
        )
    if average_flag:
        return "average"
    if cumulative_flag:
        return "cumulative"
    return "both"


def reward_target_channels(
    target: float,
    representation: str,
    normalization_const: float,
    episode_horizon: int,
    *,
    target_is_average: bool = True,
) -> np.ndarray:
    """Convert one scalar return command to the dataset's reward channels."""
    representation = resolve_reward_representation(representation)
    normalization_const = float(normalization_const)
    remaining_horizon = int(episode_horizon)
    if not np.isfinite(normalization_const) or normalization_const == 0:
        raise ValueError("normalization_const must be finite and non-zero.")
    if remaining_horizon <= 0:
        raise ValueError("remaining_horizon must be positive.")

    target = float(target)
    if target_is_average:
        average_target = target
        cumulative_target = target * remaining_horizon
    else:
        cumulative_target = target
        average_target = target / remaining_horizon
    cumulative_target /= normalization_const

    if representation == "average":
        channels = [average_target]
    elif representation == "cumulative":
        channels = [cumulative_target]
    else:
        channels = [average_target, cumulative_target]
    return np.asarray(channels, dtype=np.float32)


def reward_window_command(
    initial_cumulative_target: float,
    reward_before_window: float,
    representation: str,
    normalization_const: float,
    episode_horizon: int,
    window_start: int,
) -> np.ndarray:
    """Build the fixed command repeated within one sliding context window."""
    episode_horizon = int(episode_horizon)
    window_start = int(window_start)
    if window_start < 0:
        raise ValueError("window_start must be non-negative.")
    if episode_horizon <= 0:
        raise ValueError("episode_horizon must be positive.")
    if window_start >= episode_horizon:
        raise ValueError(
            "window_start must be smaller than episode_horizon."
        )
    remaining_horizon = episode_horizon - window_start
    remaining_target = (
        float(initial_cumulative_target) - float(reward_before_window)
    )
    return reward_target_channels(
        remaining_target,
        representation,
        normalization_const,
        remaining_horizon,
        target_is_average=False,
    )


def flat_reward_normalization(dataset_dict: Dict[str, np.ndarray]) -> float:
    """Compute the official rounded max-return scale from flat trajectories."""
    rewards = np.asarray(dataset_dict["rewards"])
    terminals = np.asarray(dataset_dict["terminals"], dtype=bool)
    if rewards.ndim != 1 or terminals.shape != rewards.shape:
        raise ValueError("rewards and terminals must be matching 1-D arrays.")
    timeouts = np.asarray(
        dataset_dict.get("timeouts", np.zeros_like(terminals)), dtype=bool
    )
    if timeouts.shape != rewards.shape:
        raise ValueError("timeouts must match rewards when provided.")

    returns, running_return, open_steps = [], 0.0, 0
    for index, reward in enumerate(rewards):
        running_return += float(reward)
        open_steps += 1
        if terminals[index] or timeouts[index]:
            returns.append(running_return)
            running_return, open_steps = 0.0, 0
    if open_steps or not returns:
        returns.append(running_return)
    normalization_const = float(round(max(returns), -3))
    return 1.0 if normalization_const == 0 else normalization_const


def discount_cumsum(
    x,
    normalize=1,
    trim=None,
    representation: Optional[str] = None,
    remaining_horizon: Optional[int] = None,
):
    trim = len(x) if trim is None else int(trim)
    csum = np.cumsum(x[::-1])[::-1][:trim]
    output_length = len(csum)
    if remaining_horizon is None:
        remaining_horizon = x.shape[0]
    remaining_horizon = int(remaining_horizon)
    if remaining_horizon < x.shape[0]:
        raise ValueError(
            "remaining_horizon cannot be shorter than the reward suffix."
        )
    denominators = remaining_horizon - np.arange(output_length)
    avg, norm = csum / denominators, csum / normalize
    representation = resolve_reward_representation(representation)
    if representation == "average":
        return avg
    elif representation == "cumulative":
        return norm
    elif representation == "both":
        return np.vstack([avg[None], norm[None]]).T
    raise AssertionError("unreachable reward representation")

class AbstractDataModule(pl.LightningDataModule, ABC):
    """Abstract class that serves as parent for all DataModules."""

    def __init__(
        self,
        batch_size: int = 256,
        val_frac: float = 0.1,
        num_workers: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Initialization for the abstract class.

        Args:
            batch_size: How many examples to return per batch.
            val_frac: What fraction of examples to use as a validation set.
            num_workers: How many cpu workers to fetch data. If not specified, takes
                the minimum of os.cpu_count() and max_num_workers (defined at the top of
                this file).
            seed: A seed for the random dataset samples.
        """
        super().__init__()
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.num_workers = num_workers or min(os.cpu_count(), max_num_workers)

        # These should be created in self.setup()
        self.data_train = None
        self.data_val = None

        if seed is None:
            seed = np.random.randint(2**31 - 1)
        self.seed = int(seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    @abstractmethod
    def setup(self, *args, **kwargs) -> None:
        """Create the training and validation data."""
        pass

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self) -> data.DataLoader:
        """Make the training dataloader."""
        return data.DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=self.generator,
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self) -> data.DataLoader:
        """Make the validation dataloader."""
        return data.DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=self.generator,
            worker_init_fn=seed_worker,
        )



def d4rl_trajectory_split(
    paths: List[Dict],
    val_frac: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Divides the D4RL trajectories into training and validation splits.

    Args:
        paths: Complete trajectories (or another indexable collection).
        val_frac: What fraction of the trajectories to put in the validation set.
        seed: Seed used only for this deterministic split.

    Returns:
        Indices indicating which trajectories go in the training set and which go in
            the validation set.
    """
    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must be between 0 (inclusive) and 1 (exclusive).")
    if len(paths) == 0:
        raise ValueError("Cannot split an empty trajectory collection.")
    if val_frac > 0 and len(paths) < 2:
        raise ValueError(
            "An independent validation split requires at least two trajectories."
        )

    n_val = int(val_frac * len(paths))
    if 0 < val_frac < 1 and len(paths) > 1:
        # A requested validation split must contain a trajectory, while still
        # leaving at least one independent trajectory for training.
        n_val = min(max(n_val, 1), len(paths) - 1)
    n_train = len(paths) - n_val

    rng = np.random.default_rng(seed)
    traj_indices = rng.permutation(len(paths))
    train_traj_indices = traj_indices[:n_train]
    val_traj_indices = traj_indices[n_train:]

    return train_traj_indices, val_traj_indices


def d4rl_timestep_split(
    dones: np.ndarray,
    val_frac: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split flat D4RL arrays by whole trajectories, then return timesteps."""
    dones = np.asarray(dones, dtype=bool).copy()
    if dones.ndim != 1 or len(dones) == 0:
        raise ValueError("dones must be a non-empty one-dimensional array.")
    # Some datasets omit a terminal marker on the final truncated episode.
    dones[-1] = True
    _, _, lengths = util.extract_done_markers(dones)
    train_trajectories, val_trajectories = d4rl_trajectory_split(
        list(range(len(lengths))),
        val_frac,
        seed=seed,
    )
    train_timesteps = util.collect_timestep_indices(dones, train_trajectories)
    val_timesteps = util.collect_timestep_indices(dones, val_trajectories)
    return train_timesteps.astype(np.int64), val_timesteps.astype(np.int64)


def reward_to_go(
    dataset: Dict[str, np.ndarray],
    average: bool = True,
    episode_horizon: Optional[int] = None,
) -> np.ndarray:
    """Compute the reward to go for each timestep.

    The implementation is iterative because when I wrote a vectorized version, np.cumsum
    cauased numerical instability.
    """
    terminals = np.asarray(dataset["terminals"], dtype=bool)
    timeouts = np.asarray(
        dataset.get("timeouts", np.zeros_like(terminals)),
        dtype=bool,
    )
    if terminals.shape != timeouts.shape:
        raise ValueError("terminals and timeouts must have matching shapes.")
    dones = np.logical_or(terminals, timeouts)
    dones[-1] = True
    _, _, lengths = util.extract_done_markers(dones)
    max_trajectory_length = int(np.max(lengths))
    if episode_horizon is None:
        episode_horizon = max_trajectory_length
    episode_horizon = int(episode_horizon)
    if episode_horizon < max_trajectory_length:
        raise ValueError(
            "episode_horizon cannot be shorter than a trajectory: "
            f"{episode_horizon} < {max_trajectory_length}."
        )

    reverse_reward_to_go = np.inf * np.ones_like(dataset["rewards"])
    running_reward = 0
    for i, (reward, done) in enumerate(zip(dataset["rewards"][::-1], dones[::-1])):
        if done:
            running_reward = 0
        running_reward += reward
        reverse_reward_to_go[i] = running_reward
    cum_reward_to_go = reverse_reward_to_go[::-1].copy()

    avg_reward_to_go = np.inf * np.ones_like(cum_reward_to_go)
    elapsed_time = 0
    for i, (cum_reward, done) in enumerate(zip(cum_reward_to_go, dones)):
        avg_reward_to_go[i] = cum_reward / (
            episode_horizon - elapsed_time
        )
        elapsed_time += 1
        if done:
            elapsed_time = 0

    return avg_reward_to_go if average else cum_reward_to_go


class D4RLTensorDatasetDataModule(AbstractDataModule):
    """Abstract class for D4RL datasets that can be stored as a TensorDataset."""

    def __init__(
        self,
        env: offline_env.OfflineEnv,
        batch_size: int,
        val_frac: float = 0.1,
        num_workers: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Custom initialization that saves the environment."""
        super().__init__(
            batch_size=batch_size,
            val_frac=val_frac,
            num_workers=num_workers,
            seed=seed,
        )
        self.env = env

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the training and validation data."""
        dataset = self.env.get_dataset()
        observation_tensor = self._get_observation_tensor(dataset)
        action_tensor = torch.tensor(dataset["actions"])
        dones = np.logical_or(dataset["terminals"], dataset["timeouts"])

        train_indices, val_indices = d4rl_timestep_split(
            dones,
            self.val_frac,
            seed=self.seed,
        )
        train_dataset = data.TensorDataset(
            observation_tensor[train_indices],
            action_tensor[train_indices],
        )
        val_dataset = (
            data.TensorDataset(
                observation_tensor[val_indices],
                action_tensor[val_indices],
            )
            if self.val_frac > 0
            else None
        )

        if stage == "fit" or stage is None:
            self.data_train, self.data_val = train_dataset, val_dataset

    @abstractmethod
    def _get_observation_tensor(self, dataset: Dict[str, np.ndarray]) -> torch.Tensor:
        pass


class D4RLBCDataModule(D4RLTensorDatasetDataModule):
    """Data module for unconditional behavior cloning in D4RL."""

    def _get_observation_tensor(self, dataset: Dict[str, np.ndarray]) -> torch.Tensor:
        return torch.tensor(dataset["observations"])


class D4RLRvSRDataModule(D4RLTensorDatasetDataModule):
    """Data module for RvS-R (reward-conditioned) learning in D4RL."""

    def __init__(
        self,
        env: offline_env.OfflineEnv,
        batch_size: int,
        val_frac: float = 0.1,
        num_workers: Optional[int] = None,
        average_reward_to_go: bool = True,
        reward_representation: Optional[str] = None,
        seed: Optional[int] = None,
        train_goal_net = False,
        K = 20,
        waypoint_horizon: int = 30,
        reward_episode_horizon: Optional[int] = None,
    ):
        """Custom initialization that sets the average_reward_to_go."""
        super().__init__(
            env,
            batch_size,
            val_frac=val_frac,
            num_workers=num_workers,
            seed=seed,
        )
        self.average_reward_to_go = average_reward_to_go
        self.reward_representation = resolve_reward_representation(
            reward_representation,
            average_reward_to_go=average_reward_to_go,
        )
        self.train_goal_net = train_goal_net
        self.K = K
        self.waypoint_horizon = waypoint_horizon
        environment_horizon = environment_episode_horizon(env)
        if (
            reward_episode_horizon is not None
            and environment_horizon is not None
            and int(reward_episode_horizon) != environment_horizon
        ):
            raise ValueError(
                "Explicit reward_episode_horizon conflicts with the "
                f"environment: {reward_episode_horizon} != "
                f"{environment_horizon}."
            )
        self.reward_episode_horizon = (
            environment_horizon
            if reward_episode_horizon is None
            else int(reward_episode_horizon)
        )
        if (
            self.reward_episode_horizon is not None
            and self.reward_episode_horizon <= 0
        ):
            raise ValueError("reward_episode_horizon must be positive.")
        self.state_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))
        self.goal_dim = (
            2 if self.reward_representation == "both" else 1
        )
        self.waypoint_dim = self.goal_dim

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the training and validation data."""
        if self.data_train is not None:
            return
        dataset = self.env.get_dataset()
        paths = self._get_paths(dataset)
        train_indices, val_indices = d4rl_trajectory_split(
            paths,
            self.val_frac,
            seed=self.seed,
        )
        self.train_trajectory_indices = train_indices.copy()
        self.val_trajectory_indices = val_indices.copy()
        if self.reward_episode_horizon is None:
            self.reward_episode_horizon = max(
                len(paths[index]["observations"])
                for index in train_indices
            )

        train_dataset = D4RLIterableDataset(
            paths[train_indices],
            goal_columns=None,
            K=self.K,
            reward_episode_horizon=self.reward_episode_horizon,
            goal_conditioned=False,
            reward_conditioned=True,
            train_goal_net = self.train_goal_net,
            reward_representation=self.reward_representation,
            waypoint_horizon=self.waypoint_horizon,
        )
        val_dataset = (
            D4RLIterableDataset(
                paths[val_indices],
                goal_columns=None, #self.goal_columns,
                K=self.K,
                reward_episode_horizon=self.reward_episode_horizon,
                goal_conditioned=False,
                reward_conditioned=True,
                train_goal_net = self.train_goal_net,
                reward_representation=self.reward_representation,
                waypoint_horizon=self.waypoint_horizon,
                normalization_const=train_dataset.normalization_const,

            )
            if self.val_frac > 0
            else None
        )
        self.reward_normalization_const = train_dataset.normalization_const

        if stage == "fit" or stage is None:
            self.data_train, self.data_val = train_dataset, val_dataset

    def _get_paths(self, dataset):
        N = dataset['rewards'].shape[0]
        data_ = collections.defaultdict(list)

        use_timeouts = 'timeouts' in dataset

        episode_step = 0
        paths = []
        for i in range(N):
            terminal_bool = bool(dataset["terminals"][i])
            timeout_bool = bool(dataset["timeouts"][i]) if use_timeouts else False
            if step.is_antmaze_env(self.env) and use_timeouts:
                done_bool = timeout_bool
            else:
                done_bool = terminal_bool or timeout_bool

            for k in ['observations', 'actions', 'rewards', 'terminals']:
                data_[k].append(dataset[k][i])
            if done_bool:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                paths.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1
        if data_["observations"]:
            paths.append({key: np.array(values) for key, values in data_.items()})
        return np.array(paths)

    def _get_observation_tensor(self, dataset: Dict[str, np.ndarray]) -> torch.Tensor:
        return make_s_g_tensor(
            dataset["observations"],
            reward_to_go(
                dataset,
                average=self.average_reward_to_go,
                episode_horizon=self.reward_episode_horizon,
            ).reshape(-1, 1),
        )


class D4RLRvSGDataModule(AbstractDataModule):
    """Data module for RvS-G (goal-conditioned) learning in D4RL."""

    def __init__(
        self,
        env: offline_env.OfflineEnv,
        batch_size: int = 32,
        val_frac: float = 0.1,
        num_workers: Optional[int] = None,
        seed: Optional[int] = None,
        K = 20,
        train_goal_net = False,
        waypoint_horizon: int = 30,
    ):
        """Custom initialization.

        Saves the environment and conditions on the (x, y) coordinate of the goal in
        AntMaze.
        """
        super().__init__(
            batch_size=batch_size,
            val_frac=val_frac,
            num_workers=num_workers,
            seed=seed,
        )
        self.env = env
        self.goal_columns = (0, 1) if step.is_antmaze_env(env) else None
        self.K = K
        self.train_goal_net = train_goal_net
        self.waypoint_horizon = waypoint_horizon
        self.state_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))
        self.goal_dim = (
            len(self.goal_columns)
            if self.goal_columns is not None
            else self.state_dim
        )
        self.waypoint_dim = self.goal_dim

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the training and validation data."""
        dataset = self.env.get_dataset()
        paths = self._get_paths(dataset)
        train_indices, val_indices = d4rl_trajectory_split(
            paths,
            self.val_frac,
            seed=self.seed,
        )
        self.train_trajectory_indices = train_indices.copy()
        self.val_trajectory_indices = val_indices.copy()

        train_dataset = D4RLIterableDataset(
            paths[train_indices],
            goal_columns=self.goal_columns,
            K=self.K,
            goal_conditioned=True,
            reward_conditioned=False,
            train_goal_net=self.train_goal_net,
            waypoint_horizon=self.waypoint_horizon,
        )
        val_dataset = (
            D4RLIterableDataset(
                paths[val_indices],
                goal_columns=self.goal_columns,
                K=self.K,
                goal_conditioned=True,
                reward_conditioned=False,
                train_goal_net=self.train_goal_net,
                waypoint_horizon=self.waypoint_horizon,
                normalization_const=train_dataset.normalization_const,
            )
            if self.val_frac > 0
            else None
        )

        if stage == "fit" or stage is None:
            self.data_train, self.data_val = train_dataset, val_dataset

    def _get_paths(self, dataset):
        N = dataset['rewards'].shape[0]
        data_ = collections.defaultdict(list)

        use_timeouts = 'timeouts' in dataset

        episode_step = 0
        paths = []
        for i in range(N):
            terminal_bool = bool(dataset["terminals"][i])
            timeout_bool = bool(dataset["timeouts"][i]) if use_timeouts else False
            if step.is_antmaze_env(self.env) and use_timeouts:
                done_bool = timeout_bool
            else:
                done_bool = terminal_bool or timeout_bool

            for k in ['observations', 'actions', 'rewards', 'terminals']:
                data_[k].append(dataset[k][i])
            if done_bool:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                paths.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1
        if data_["observations"]:
            paths.append({key: np.array(values) for key, values in data_.items()})
        return np.array(paths)
