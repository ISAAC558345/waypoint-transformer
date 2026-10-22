"""
Defines the policy class, including the main training step logic.

Source: RvS GitHub repository (https://github.com/scottemmons/rvs)
Edits by Anonymous Authors
"""

from typing import Dict, Optional, Tuple, Type, List

import gym
import os
from gym import spaces
import numpy as np
import pytorch_lightning as pl
from stable_baselines3.common import policies, type_aliases, utils, distributions
import torch
from torch import nn, optim

from wt import dataset, layers, step, util
from wt.transformer import TransformerExtractor


def _flat_observation_dim(space: gym.Space) -> int:
    """Return the flattened observation width represented in dataset tensors."""
    if getattr(space, "shape", None):
        return int(np.prod(space.shape))
    return int(spaces.utils.flatdim(space))


def _action_tensor_dim(space: gym.Space) -> int:
    """Return the width of one action as consumed by the policy tensor."""
    if isinstance(space, spaces.Discrete):
        return 1
    if isinstance(space, spaces.MultiDiscrete):
        return int(len(space.nvec))
    if isinstance(space, spaces.MultiBinary):
        return int(np.prod(space.shape))
    if getattr(space, "shape", None):
        return int(np.prod(space.shape))
    raise ValueError(f"Unsupported action space for Transformer actions: {space}")


def make_obs_goal_space(
    observation_space: gym.Space,
    unconditional_policy: bool = False,
    reward_conditioning: bool = False,
    xy_conditioning: bool = False,
    reward_condition_dim: Optional[int] = None,
) -> gym.Space:
    """Create the policy's input space.

    This includes the observation as well as a possible goal.

    Args:
        observation_space: The observation space of the environment. By default, it's
            duplicated to create the goal space.
        unconditional_policy: If True, do not use any goals, and only use the
            observation_space.
        reward_conditioning: If True, condition on a reward scalar appended to the
            observation_space.
        xy_conditioning: If True, condition on (x, y) coordinates appended to the
            observation_space.

    Returns:
        The new space including observation and goal.

    Raises:
        ValueError: If conflicting types of spaces are specified.
    """
    if sum([unconditional_policy, reward_conditioning, xy_conditioning]) > 1:
        raise ValueError("You must choose at most one policy conditioning setting.")

    if unconditional_policy:
        return observation_space
    elif reward_conditioning:
        if reward_condition_dim is None:
            reward_condition_dim = (
                2
                if dataset.resolve_reward_representation() == "both"
                else 1
            )
        if reward_condition_dim <= 0:
            raise ValueError("reward_condition_dim must be positive.")
        conditioned_space = observation_space
        for _ in range(int(reward_condition_dim)):
            conditioned_space = util.add_scalar_to_space(conditioned_space)
        return conditioned_space
    elif xy_conditioning:
        return util.add_scalar_to_space(util.add_scalar_to_space(observation_space))
    else:
        return util.create_observation_goal_space(observation_space)


class RvS(pl.LightningModule):
    """A Reinforcement Learning via Supervised Learning (RvS) policy."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_size: int = 1024,
        depth: int = 2,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        activation_fn: Type[nn.Module] = nn.ReLU,
        dropout_p: float = 0.1,
        unconditional_policy: bool = False,
        reward_conditioning: bool = False,
        env_name: Optional[str] = None,
        goal_network_ckpt: Optional[str] = None,
        goal_network_config: Optional[Dict] = None,
        transformer_config: Optional[Dict] = None,
        waypoint_horizon: int = 30,
        context_length: int = 20,
        waypoint_hidden_size: int = 1024,
        reward_condition_dim: Optional[int] = None,
        reward_representation: Optional[str] = None,
        reward_normalization_const: Optional[float] = None,
        reward_episode_horizon: Optional[int] = None,
        _goal_network_restore_mode: Optional[str] = None,
    ):
        """Builds RvS.

        Args:
            observation_space: The policy's observation space
            action_space: The policy's action space
            hidden_size: The width of each hidden layer
            depth: The number of hidden layers
            learning_rate: A learning rate held constant throughout training
            batch_size: The batch size for each gradient step
            activation_fn: The network's activation function
            dropout_p: The dropout probability
            unconditional_policy: If True, ignore goals and act only based on
                observations
            reward_conditioning: If True, condition on a reward scalar instead of future
                observations
            env_name: The name of the environment for which to configure the policy
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.unconditional_policy = unconditional_policy
        self.batch_size = batch_size
        self.context_length = int(context_length)
        if self.context_length <= 0:
            raise ValueError("context_length must be positive.")
        self.reward_conditioning = bool(reward_conditioning)
        if reward_conditioning:
            if reward_representation is None:
                if reward_condition_dim == 2:
                    reward_representation = "both"
                elif reward_condition_dim == 1:
                    inferred_representation = (
                        dataset.resolve_reward_representation()
                    )
                    if inferred_representation == "both":
                        raise ValueError(
                            "A one-channel reward-conditioned policy must specify "
                            "reward_representation='average' or 'cumulative'."
                        )
                    reward_representation = inferred_representation
                else:
                    reward_representation = (
                        dataset.resolve_reward_representation()
                    )
            reward_representation = dataset.resolve_reward_representation(
                reward_representation
            )
            expected_reward_dim = (
                2 if reward_representation == "both" else 1
            )
            if (
                reward_condition_dim is not None
                and int(reward_condition_dim) != expected_reward_dim
            ):
                raise ValueError(
                    "reward_condition_dim conflicts with reward_representation: "
                    f"{reward_condition_dim} != {expected_reward_dim}."
                )
            reward_condition_dim = expected_reward_dim
            if reward_normalization_const is not None:
                reward_normalization_const = float(
                    reward_normalization_const
                )
                if (
                    not np.isfinite(reward_normalization_const)
                    or reward_normalization_const == 0
                ):
                    raise ValueError(
                        "reward_normalization_const must be finite and non-zero."
                    )
            if reward_episode_horizon is not None:
                reward_episode_horizon = int(reward_episode_horizon)
                if reward_episode_horizon <= 0:
                    raise ValueError(
                        "reward_episode_horizon must be positive."
                    )
        else:
            reward_representation = None
            reward_normalization_const = None
            reward_episode_horizon = None
        self.reward_representation = reward_representation
        self.reward_normalization_const = reward_normalization_const
        self.reward_episode_horizon = reward_episode_horizon
        self.goal_network_ckpt = goal_network_ckpt
        if _goal_network_restore_mode not in (None, "embedded"):
            raise ValueError(
                "_goal_network_restore_mode must be None or 'embedded'."
            )

        xy_conditioning = (
            env_name in step.d4rl_antmaze
            and not unconditional_policy
            and not reward_conditioning
        )
        observation_goal_space = make_obs_goal_space(
            observation_space,
            unconditional_policy=unconditional_policy,
            reward_conditioning=reward_conditioning,
            xy_conditioning=xy_conditioning,
            reward_condition_dim=reward_condition_dim,
        )
        state_dim = _flat_observation_dim(observation_space)
        policy_input_dim = _flat_observation_dim(observation_goal_space)
        condition_dim = policy_input_dim - state_dim
        if condition_dim < 0:
            raise ValueError(
                "Conditioned observation space is smaller than the state space"
            )
        derived_goal_network_config = {
            "state_dim": state_dim,
            "goal_dim": condition_dim,
            "waypoint_dim": condition_dim,
            "hidden_dim": int(waypoint_hidden_size),
            "max_T": int(waypoint_horizon),
            "reward": reward_conditioning,
            "reward_representation": reward_representation,
            "reward_normalization_const": reward_normalization_const,
            "reward_episode_horizon": reward_episode_horizon,
            "network_type": (
                "manual" if goal_network_ckpt == "manual" else "learned"
            ),
        }
        if goal_network_config is not None:
            protected_goal_fields = {
                "state_dim",
                "goal_dim",
                "waypoint_dim",
                "reward",
                "reward_representation",
                "reward_normalization_const",
                "reward_episode_horizon",
                "network_type",
            }
            supplied_state_dim = goal_network_config.get("obs_dim")
            if (
                supplied_state_dim is not None
                and int(supplied_state_dim) != state_dim
            ):
                raise ValueError(
                    "goal_network_config obs_dim conflicts with the "
                    f"environment: {supplied_state_dim} != {state_dim}."
                )
            for name in protected_goal_fields & goal_network_config.keys():
                supplied = goal_network_config[name]
                expected = derived_goal_network_config[name]
                matches = (
                    np.isclose(supplied, expected)
                    if name == "reward_normalization_const"
                    and supplied is not None
                    and expected is not None
                    else supplied == expected
                )
                if not matches:
                    raise ValueError(
                        "goal_network_config conflicts with policy/dataset "
                        f"metadata: {name}={supplied!r}, "
                        f"expected {expected!r}."
                    )
            derived_goal_network_config.update(goal_network_config)
        model_actions = dataset._environment_flag("MODEL_ACTIONS")
        action_dim = _action_tensor_dim(action_space)
        derived_transformer_config = {
            "context_length": int(context_length),
            "n_layer": int(os.environ.get("NUM_LAYERS", 2)),
            "n_head": 16,
            "n_embd": 128,
            "model_actions": model_actions,
            "action_dim": action_dim,
            "resid_pdrop": float(os.environ.get("DROPOUT", 0.15)),
            "attn_pdrop": float(os.environ.get("DROPOUT", 0.15)),
        }
        if transformer_config is not None:
            derived_transformer_config.update(transformer_config)
        goal_network_config = derived_goal_network_config
        transformer_config = derived_transformer_config
        self.goal_network_config = dict(derived_goal_network_config)
        self.transformer_config = dict(derived_transformer_config)
        self.save_hyperparameters(
            "hidden_size",
            "depth",
            "learning_rate",
            "batch_size",
            "activation_fn",
            "dropout_p",
            "unconditional_policy",
            "reward_conditioning",
            "env_name",
            "goal_network_ckpt",
            "goal_network_config",
            "transformer_config",
            "waypoint_horizon",
            "context_length",
            "waypoint_hidden_size",
            "reward_condition_dim",
            "reward_representation",
            "reward_normalization_const",
            "reward_episode_horizon",
        )
        lr_schedule = utils.constant_fn(learning_rate)
        net_arch = [hidden_size] * depth
        layers.DropoutActivation.activation_fn = activation_fn
        layers.DropoutActivation.p = dropout_p
        self.model = ExtendedActorCriticPolicy(
            observation_goal_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=layers.DropoutActivation,
            features_extractor_class=layers.IdentityExtractor,
            goal_network_ckpt=goal_network_ckpt,
            goal_network_config=derived_goal_network_config,
            transformer_config=derived_transformer_config,
            goal_network_restore_mode=_goal_network_restore_mode,
        )

    @classmethod
    def _load_model_state(cls, checkpoint, strict=True, **overrides):
        """Restore embedded waypoint weights without reopening their source file.

        Lightning 1.5 constructs the module before applying ``state_dict``.
        Detecting registered waypoint keys here lets that constructor build only
        the saved topology; strict outer loading then supplies the actual frozen
        weights. Fresh policy construction still requires a valid source path.
        """
        state_dict = checkpoint.get("state_dict", {})
        embedded_waypoint = any(
            ".goal_network." in name
            or name.startswith("goal_network.")
            for name in state_dict
        )
        if embedded_waypoint:
            overrides.setdefault(
                "_goal_network_restore_mode", "embedded"
            )
        return super()._load_model_state(
            checkpoint,
            strict=strict,
            **overrides,
        )

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute a forward pass with the model."""
        return self.model.forward(*args, **kwargs)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        log_prefix: str = "train",
    ) -> torch.Tensor:
        """Computes loss for a training batch."""
        obs_goal, action, mask = batch
        action = action.view(action.shape[0] * action.shape[1], -1)
        _, log_probs, _, prediction = self.model.evaluate_and_predict(
            obs_goal,
            action,
            mask,
        )
        # dynamics_preds = self.model.predict_dynamics(cached = True)

        valid_mask = mask.view(-1).bool()
        if valid_mask.numel() != log_probs.numel():
            raise ValueError(
                "Padding mask and action log-probabilities have different "
                f"sizes: {valid_mask.numel()} != {log_probs.numel()}"
            )
        if not valid_mask.any():
            raise ValueError("A training batch contains no valid timesteps")
        log_probs = log_probs[valid_mask]

        prob_true_act = torch.exp(log_probs).mean()
        loss = -log_probs.mean()

        self.log(f"{log_prefix}_prob_true_act", prob_true_act)
        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        try:
            self.log(f"{log_prefix}_std", torch.exp(self.model.log_std).mean())
            self.log(f"{log_prefix}_log_std", self.model.log_std.mean())
        except AttributeError:
            pass
        if prediction is not None:
            prediction_error = (prediction - action) ** 2
            prediction_error = prediction_error[valid_mask]
            self.log(f"{log_prefix}_mse", prediction_error.mean())

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Computes loss for a validation batch."""
        loss = self.training_step(batch, batch_idx, log_prefix="val")
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        """Configures the optimizer used by PyTorch Lightning."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ##################
    # POLICY FUNCTIONS
    ##################

    def get_probabilities(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        actions: np.ndarray,
    ) -> torch.Tensor:
        """Get the policy's probabilities.

        Returns a probability for each action given the corresponding observation and
        goal.
        """
        assert actions.shape[0] == observations.shape[0] == goals.shape[0]
        s_g_tensor = dataset.make_s_g_tensor(observations, goals)
        a_tensor = torch.tensor(actions)

        self.model.eval()
        with torch.no_grad():
            _, log_probs, _ = self.model.evaluate_actions(s_g_tensor, a_tensor)
        probs = torch.exp(log_probs)
        return probs

    def get_action(
        self,
        observation: List[np.ndarray],
        goal: np.ndarray,
        deterministic: bool = True,
        actions = None
    ) -> np.ndarray:
        """Get an action for a single observation / goal pair."""
        return self.get_actions(
            np.concatenate([obs[np.newaxis, np.newaxis] for obs in observation], axis = 1),
            np.concatenate([g[np.newaxis, np.newaxis] for g in goal], axis = 1),
            deterministic=deterministic,
            actions = np.concatenate([act[np.newaxis, np.newaxis] for act in actions], axis = 1) if actions is not None else actions
        )[-1]


    def get_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        deterministic: bool = True,
        actions = None
    ) -> np.ndarray:
        """Get actions for each observation / goal pair."""
        assert observations.shape[0] == goals.shape[0]

        self.model.eval()
        with torch.no_grad():
            if self.unconditional_policy:
                s_tensor = torch.tensor(observations)
                actions = self.model._predict(s_tensor, deterministic=deterministic)
            else:
                s_g_tensor = dataset.make_s_g_tensor(observations, goals).float()
                actions = self.model._predict([s_g_tensor, torch.tensor(actions).float() if actions is not None else actions], deterministic=deterministic)

        return actions.cpu().numpy()


class ExtendedActorCriticPolicy(policies.ActorCriticPolicy):
    """Extends the functionality of stable-baseline3's ActorCriticPolicy.

    The extended functionality includes:
    - Action and value predictions at the same time as evaluating probabilities.
    - The option to skip value function computation.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: type_aliases.Schedule,
        compute_values: bool = False,
        goal_network_ckpt: str = None,
        goal_network_config: Optional[Dict] = None,
        transformer_config: Optional[Dict] = None,
        goal_network_restore_mode: Optional[str] = None,
        **kwargs,
    ):
        """Builds ExtendedActorCriticPolicy.

        Args:
            observation_space: The policy's observation space.
            action_space: The policy's action space.
            lr_schedule: A learning rate schedule.
            compute_values: We'll skip value function computation unless this is True.
            **kwargs: Keyword arguments passed along to parent class.
        """
        self.compute_values = compute_values
        self.goal_network_ckpt = goal_network_ckpt
        self.goal_network_config = (
            None if goal_network_config is None else dict(goal_network_config)
        )
        if (
            self.goal_network_ckpt == "manual"
            and self.goal_network_config is not None
        ):
            # ``manual`` is an explicit network selector, not a source of
            # dimensional metadata.
            self.goal_network_config["network_type"] = "manual"
        self.transformer_config = transformer_config
        self.goal_network_restore_mode = goal_network_restore_mode

        super(ExtendedActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

        # self.dynamics_dist = DiagGaussianDistribution(self.features_dim)
        # self.dynamics_dist.proba_distribution_net(latent_dim = self.mlp_extractor.latent_dim_pi, 0.0)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """Keep eagerly registered, frozen waypoint weights pretrained."""
        direct_parameters = list(module.parameters(recurse=False))
        if direct_parameters and all(
            not parameter.requires_grad for parameter in direct_parameters
        ):
            return
        policies.ActorCriticPolicy.init_weights(module, gain=gain)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = TransformerExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            config=self.transformer_config,
            goal_network_ckpt=self.goal_network_ckpt,
            goal_network_config=self.goal_network_config,
            goal_network_restore_mode=self.goal_network_restore_mode,
            action_dim=_action_tensor_dim(self.action_space),
            #'wandb/goal_network_try3_30/files/checkpoints/gcsl-antmaze-large-play-v2-epoch=016-val_loss=1.9702e-01.ckpt'
        )

    def extract_features(self, obs):
        return obs

    def _build(self, lr_schedule: type_aliases.Schedule) -> None:
        super(ExtendedActorCriticPolicy, self)._build(lr_schedule)
        # Skip unused value function computation
        if not self.compute_values:
            self.value_net = nn.Sequential()  # Identity function
            self.optimizer = self.optimizer_class(
                self.parameters(),
                lr=lr_schedule(1),
                **self.optimizer_kwargs,
            )

    def evaluate_and_predict(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate probability of actions and provide action and value prediction."""
        latent_pi, latent_vf, latent_sde = self._get_latent(
            [obs, actions, attention_mask]
        )
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        if isinstance(self.action_space, spaces.Box):
            predictions = distribution.get_actions(deterministic=True)
        else:
            predictions = None
        return values, log_prob, distribution.entropy(), predictions

    def predict_dynamics(cached: bool):
        if not cached:
            raise NotImplementedError("must cache state,action features")
        # self.mlp_extractor.action_features
