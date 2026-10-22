"""CPU-only regression tests for the stage 0.6 training closure.

The official environment eagerly imports D4RL, GCSL, MuJoCo, Lightning and
Stable-Baselines3. These tests replace only those integration boundaries with
small import stubs, while exercising the repository's real dataset, waypoint
network, policy loss and GPT implementations.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def _install_import_stubs() -> None:
    """Install minimal interfaces for dependencies unrelated to tensor smoke."""
    lightning = types.ModuleType("pytorch_lightning")

    class LightningModule(nn.Module):
        def save_hyperparameters(self, parameters=None, *args, **kwargs):
            if isinstance(parameters, dict):
                self._saved_hyperparameters = dict(parameters)
            return None

        def log(self, *args, **kwargs):
            return None

    class LightningDataModule:
        pass

    lightning.LightningModule = LightningModule
    lightning.LightningDataModule = LightningDataModule
    sys.modules["pytorch_lightning"] = lightning

    d4rl = types.ModuleType("d4rl")
    offline_env = types.ModuleType("d4rl.offline_env")

    class OfflineEnv:
        pass

    offline_env.OfflineEnv = OfflineEnv
    d4rl.offline_env = offline_env
    sys.modules["d4rl"] = d4rl
    sys.modules["d4rl.offline_env"] = offline_env

    step = types.ModuleType("wt.step")
    step.d4rl_env_names = []
    step.d4rl_antmaze = []
    step.gym_goal_envs = []
    step.GCSLToGym = object
    step.is_antmaze_env = lambda env: bool(getattr(env, "is_antmaze", False))
    sys.modules["wt.step"] = step
    sys.modules["wt.train"] = types.ModuleType("wt.train")

    wandb = types.ModuleType("wandb")
    wandb_run = types.ModuleType("wandb.wandb_run")

    class Run:
        pass

    wandb_run.Run = Run
    sys.modules["wandb"] = wandb
    sys.modules["wandb.wandb_run"] = wandb_run

    stable_baselines3 = types.ModuleType("stable_baselines3")
    common = types.ModuleType("stable_baselines3.common")
    torch_layers = types.ModuleType("stable_baselines3.common.torch_layers")
    preprocessing = types.ModuleType("stable_baselines3.common.preprocessing")
    sb3_policies = types.ModuleType("stable_baselines3.common.policies")
    type_aliases = types.ModuleType("stable_baselines3.common.type_aliases")
    utils = types.ModuleType("stable_baselines3.common.utils")
    distributions = types.ModuleType("stable_baselines3.common.distributions")

    class BaseFeaturesExtractor(nn.Module):
        def __init__(self, observation_space, features_dim):
            super().__init__()
            self.features_dim = features_dim

    class MlpExtractor(nn.Module):
        def __init__(self, feature_dim, net_arch, activation_fn, device="auto"):
            super().__init__()
            self.shared_net = nn.Identity()
            self.policy_net = nn.Identity()
            self.value_net = nn.Identity()
            self.latent_dim_pi = feature_dim
            self.latent_dim_vf = feature_dim

    class ActorCriticPolicy(nn.Module):
        pass

    def get_flattened_obs_dim(space):
        return int(np.prod(space.shape))

    torch_layers.BaseFeaturesExtractor = BaseFeaturesExtractor
    torch_layers.MlpExtractor = MlpExtractor
    preprocessing.get_flattened_obs_dim = get_flattened_obs_dim
    sb3_policies.ActorCriticPolicy = ActorCriticPolicy
    type_aliases.Schedule = object
    utils.constant_fn = lambda value: lambda _: value
    common.policies = sb3_policies
    common.type_aliases = type_aliases
    common.utils = utils
    common.distributions = distributions
    stable_baselines3.common = common

    sys.modules["stable_baselines3"] = stable_baselines3
    sys.modules["stable_baselines3.common"] = common
    sys.modules["stable_baselines3.common.torch_layers"] = torch_layers
    sys.modules["stable_baselines3.common.preprocessing"] = preprocessing
    sys.modules["stable_baselines3.common.policies"] = sb3_policies
    sys.modules["stable_baselines3.common.type_aliases"] = type_aliases
    sys.modules["stable_baselines3.common.utils"] = utils
    sys.modules["stable_baselines3.common.distributions"] = distributions


_install_import_stubs()

from wt import dataset, goal_net, policies  # noqa: E402
from wt.transformer import GPT, GPTConfig, TransformerExtractor  # noqa: E402
from gym import spaces as gym_spaces  # noqa: E402


def _load_real_step_module():
    """Load wt.step under an alias while keeping heavyweight envs stubbed."""
    import wt as wt_package

    d4rl = sys.modules["d4rl"]
    infos = types.ModuleType("d4rl.infos")
    infos.REF_MIN_SCORE = {}
    infos.REF_MAX_SCORE = {}
    d4rl.infos = infos
    sys.modules["d4rl.infos"] = infos

    kitchen = types.ModuleType("d4rl.kitchen")
    kitchen_envs = types.ModuleType("d4rl.kitchen.kitchen_envs")
    kitchen_envs.KitchenBase = object
    kitchen_envs.OBS_ELEMENT_INDICES = {}
    kitchen_envs.OBS_ELEMENT_GOALS = {}
    adept_envs = types.ModuleType("d4rl.kitchen.adept_envs")
    franka = types.ModuleType("d4rl.kitchen.adept_envs.franka")
    kitchen_multitask = types.ModuleType(
        "d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0"
    )
    kitchen_multitask.KitchenTaskRelaxV1 = object
    kitchen.kitchen_envs = kitchen_envs
    franka.kitchen_multitask_v0 = kitchen_multitask
    sys.modules["d4rl.kitchen"] = kitchen
    sys.modules["d4rl.kitchen.kitchen_envs"] = kitchen_envs
    sys.modules["d4rl.kitchen.adept_envs"] = adept_envs
    sys.modules["d4rl.kitchen.adept_envs.franka"] = franka
    sys.modules[
        "d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0"
    ] = kitchen_multitask

    locomotion = types.ModuleType("d4rl.locomotion")
    ant = types.ModuleType("d4rl.locomotion.ant")
    ant.AntMazeEnv = object
    locomotion.ant = ant
    sys.modules["d4rl.locomotion"] = locomotion
    sys.modules["d4rl.locomotion.ant"] = ant

    gcsl = types.ModuleType("gcsl")
    gcsl_envs = types.ModuleType("gcsl.envs")
    gcsl_envs.env_names = []
    gcsl_algo = types.ModuleType("gcsl.algo")
    variants = types.ModuleType("gcsl.algo.variants")
    goal_env = types.ModuleType("gcsl.envs.goal_env")
    goal_env.GoalEnv = object
    gcsl.envs = gcsl_envs
    gcsl.algo = gcsl_algo
    gcsl_algo.variants = variants
    gcsl_envs.goal_env = goal_env
    sys.modules["gcsl"] = gcsl
    sys.modules["gcsl.envs"] = gcsl_envs
    sys.modules["gcsl.algo"] = gcsl_algo
    sys.modules["gcsl.algo.variants"] = variants
    sys.modules["gcsl.envs.goal_env"] = goal_env

    visualize = types.ModuleType("wt.visualize")
    sys.modules["wt.visualize"] = visualize
    wt_package.visualize = visualize

    module_name = "wt._stage06_real_step"
    spec = importlib.util.spec_from_file_location(
        module_name, REPOSITORY_ROOT / "src" / "wt" / "step.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _trajectory(
    trajectory_id: int,
    length: int = 7,
    state_dim: int = 3,
    action_dim: int = 2,
):
    observations = np.arange(length * state_dim, dtype=np.float32).reshape(
        length, state_dim
    )
    observations = observations + trajectory_id * 1000
    return {
        "observations": observations,
        "actions": np.full(
            (length, action_dim), trajectory_id, dtype=np.float32
        ),
        "rewards": np.arange(1, length + 1, dtype=np.float32),
        "terminals": np.array(
            [False] * (length - 1) + [True], dtype=bool
        ),
    }


class DatasetSplitTests(unittest.TestCase):
    def test_trajectory_split_is_seeded_disjoint_and_complete(self):
        paths = [_trajectory(index) for index in range(10)]
        train_a, val_a = dataset.d4rl_trajectory_split(paths, 0.2, seed=17)
        train_b, val_b = dataset.d4rl_trajectory_split(paths, 0.2, seed=17)

        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(val_a, val_b)
        self.assertEqual(set(train_a) & set(val_a), set())
        self.assertEqual(set(train_a) | set(val_a), set(range(10)))
        self.assertEqual((len(train_a), len(val_a)), (8, 2))

    def test_independent_validation_rejects_single_trajectory(self):
        with self.assertRaisesRegex(ValueError, "at least two trajectories"):
            dataset.d4rl_trajectory_split([_trajectory(0)], 0.2, seed=1)

    def test_flat_dataset_split_keeps_episodes_independent(self):
        dones = np.array(
            [False, False, True, False, True, False, False, False, True]
        )
        train, val = dataset.d4rl_timestep_split(dones, 1 / 3, seed=3)
        episodes = [set(range(0, 3)), set(range(3, 5)), set(range(5, 9))]

        for episode in episodes:
            self.assertFalse(bool(episode & set(train)) and bool(episode & set(val)))
        self.assertEqual(set(train) | set(val), set(range(len(dones))))

    def test_rvs_data_modules_build_validation_from_validation_paths(self):
        class Space:
            def __init__(self, shape):
                self.shape = shape

        class SyntheticEnv:
            observation_space = Space((3,))
            action_space = Space((2,))
            is_antmaze = False

            def get_dataset(self):
                trajectories = [_trajectory(index, length=3) for index in range(10)]
                for index, path in enumerate(trajectories):
                    path["rewards"] = path["rewards"] + index * 1000
                terminals = np.concatenate(
                    [path["terminals"] for path in trajectories]
                )
                timeouts = np.tile(
                    np.array([False, False, True]), len(trajectories)
                )
                # Exercise the explicit final partial-trajectory flush.
                terminals[-1] = False
                timeouts[-1] = False
                return {
                    "observations": np.concatenate(
                        [path["observations"] for path in trajectories]
                    ),
                    "actions": np.concatenate(
                        [path["actions"] for path in trajectories]
                    ),
                    "rewards": np.concatenate(
                        [path["rewards"] for path in trajectories]
                    ),
                    "terminals": terminals,
                    "timeouts": timeouts,
                }

        for module_type in (
            dataset.D4RLRvSRDataModule,
            dataset.D4RLRvSGDataModule,
        ):
            with self.subTest(module=module_type.__name__):
                module = module_type(
                    SyntheticEnv(),
                    batch_size=2,
                    val_frac=0.2,
                    num_workers=1,
                    seed=19,
                    K=3,
                )
                module.setup("fit")
                train_ids = {
                    int(path["observations"][0, 0] // 1000)
                    for path in module.data_train.trajectories
                }
                val_ids = {
                    int(path["observations"][0, 0] // 1000)
                    for path in module.data_val.trajectories
                }

                self.assertEqual(train_ids & val_ids, set())
                self.assertEqual(train_ids | val_ids, set(range(10)))
                self.assertEqual(len(val_ids), 2)
                self.assertEqual(
                    module.data_train.normalization_const,
                    module.data_val.normalization_const,
                )
                expected_normalization = round(
                    max(
                        sum(path["rewards"])
                        for path in module.data_train.trajectories
                    ),
                    -3,
                )
                self.assertEqual(
                    module.data_train.normalization_const,
                    expected_normalization,
                )
                if isinstance(module, dataset.D4RLRvSRDataModule):
                    self.assertEqual(module.reward_representation, "both")
                    self.assertEqual(module.goal_dim, 2)
                    self.assertEqual(module.waypoint_dim, 2)
                    self.assertEqual(module.reward_episode_horizon, 3)
                    self.assertEqual(
                        module.data_train.reward_episode_horizon,
                        module.data_val.reward_episode_horizon,
                    )
                    self.assertEqual(
                        module.reward_normalization_const,
                        expected_normalization,
                    )


class WaypointDatasetTests(unittest.TestCase):
    def test_structured_goal_batch_shapes_and_terminal_clamp(self):
        source = _trajectory(0, length=7)
        iterable = dataset.D4RLIterableDataset(
            [source],
            index_batch_size=1,
            K=5,
            goal_conditioned=True,
            reward_conditioned=False,
            goal_columns=(0, 1),
            waypoint_horizon=3,
        )
        components = iterable._fetch_trajectory_components(
            np.array([0]), np.array([4]), np.array([6])
        )

        self.assertEqual(
            set(components),
            {"states", "actions", "rewards", "goals", "waypoints", "mask"},
        )
        expected_shapes = {
            "states": (1, 5, 3),
            "actions": (1, 5, 2),
            "rewards": (1, 5, 1),
            "goals": (1, 5, 2),
            "waypoints": (1, 5, 2),
            "mask": (1, 5),
        }
        for name, shape in expected_shapes.items():
            self.assertEqual(tuple(components[name].shape), shape)

        # The valid source indices are 4, 5 and 6. All t+3 targets clamp to
        # the last state (index 6), never to another trajectory.
        valid_waypoints = components["waypoints"][0, components["mask"][0]]
        expected_terminal = torch.as_tensor(source["observations"][6, :2])
        torch.testing.assert_close(
            valid_waypoints,
            expected_terminal.repeat(3, 1),
        )

    def test_reward_waypoint_uses_configured_horizon(self):
        source = _trajectory(0, length=6)
        iterable = dataset.D4RLIterableDataset(
            [source],
            index_batch_size=1,
            K=4,
            goal_conditioned=False,
            reward_conditioned=True,
            train_goal_net=True,
            waypoint_horizon=2,
        )
        components = iterable._fetch_trajectory_components(
            np.array([0]), np.array([2]), np.array([5])
        )
        full_rtg = dataset.discount_cumsum(
            source["rewards"][2:],
            normalize=iterable.normalization_const,
            trim=4,
        )
        expected = torch.as_tensor(
            full_rtg[[2, 3, 3, 3]], dtype=torch.float32
        )
        torch.testing.assert_close(components["waypoints"][0], expected)
        torch.testing.assert_close(
            components["goals"][0],
            torch.as_tensor(full_rtg[0], dtype=torch.float32).repeat(4, 1),
        )
        self.assertEqual(tuple(components["waypoints"].shape), (1, 4, 2))

    def test_reward_representation_controls_condition_dimension(self):
        source = _trajectory(0, length=6)
        for representation, expected_dim in (
            ("average", 1),
            ("cumulative", 1),
            ("both", 2),
        ):
            with self.subTest(representation=representation):
                iterable = dataset.D4RLIterableDataset(
                    [source],
                    index_batch_size=1,
                    K=4,
                    goal_conditioned=False,
                    reward_conditioned=True,
                    reward_representation=representation,
                    waypoint_horizon=2,
                )
                components = iterable._fetch_trajectory_components(
                    np.array([0]), np.array([1]), np.array([5])
                )
                self.assertEqual(components["goals"].shape[-1], expected_dim)
                self.assertEqual(
                    components["waypoints"].shape[-1], expected_dim
                )

    def test_policy_reward_condition_space_matches_dataset_dimension(self):
        state_space = gym_spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        for condition_dim in (1, 2):
            with self.subTest(condition_dim=condition_dim):
                conditioned = policies.make_obs_goal_space(
                    state_space,
                    reward_conditioning=True,
                    reward_condition_dim=condition_dim,
                )
                self.assertEqual(conditioned.shape, (3 + condition_dim,))

    def test_reward_target_channels_are_environment_independent(self):
        expected = {
            "average": np.array([2.0], dtype=np.float32),
            "cumulative": np.array([0.5], dtype=np.float32),
            "both": np.array([2.0, 0.5], dtype=np.float32),
        }
        for representation, expected_channels in expected.items():
            with self.subTest(representation=representation):
                channels = dataset.reward_target_channels(
                    2.0,
                    representation,
                    normalization_const=4000,
                    episode_horizon=1000,
                    target_is_average=True,
                )
                np.testing.assert_allclose(channels, expected_channels)

    def test_reward_representation_rejects_conflicting_legacy_flags(self):
        with mock.patch.dict(
            os.environ,
            {"AVG_REWARD": "1", "CM_REWARD": "1"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "cannot both be set"):
                dataset.resolve_reward_representation()
            self.assertEqual(
                dataset.resolve_reward_representation("cumulative"),
                "cumulative",
            )

    def test_false_legacy_reward_flags_are_not_enabled(self):
        with mock.patch.dict(
            os.environ,
            {"AVG_REWARD": "false", "CM_REWARD": "0"},
            clear=False,
        ):
            self.assertEqual(
                dataset.resolve_reward_representation(),
                "both",
            )

    def test_average_reward_uses_nominal_horizon_after_early_terminal(self):
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        targets = dataset.discount_cumsum(
            rewards,
            normalize=10.0,
            representation="both",
            remaining_horizon=5,
        )
        np.testing.assert_allclose(
            targets,
            np.array(
                [[6 / 5, 0.6], [5 / 4, 0.5], [3 / 3, 0.3]],
                dtype=np.float32,
            ),
        )

    def test_sliding_reward_command_matches_training_window(self):
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        training_window = dataset.discount_cumsum(
            rewards[1:],
            normalize=10.0,
            representation="both",
            remaining_horizon=4,
        )[0]
        online_window = dataset.reward_window_command(
            initial_cumulative_target=6.0,
            reward_before_window=1.0,
            representation="both",
            normalization_const=10.0,
            episode_horizon=5,
            window_start=1,
        )
        np.testing.assert_allclose(online_window, training_window)
        np.testing.assert_allclose(
            online_window,
            np.array([5 / 4, 0.5], dtype=np.float32),
        )

    def test_reward_horizon_conflicts_are_rejected(self):
        self.assertEqual(
            dataset.resolve_reward_episode_horizon(
                checkpoint_horizon=5,
                environment_horizon=5,
            ),
            5,
        )
        with self.assertRaisesRegex(ValueError, "evaluation environment"):
            dataset.resolve_reward_episode_horizon(
                checkpoint_horizon=5,
                environment_horizon=6,
            )

    def test_reward_dataset_rejects_horizon_shorter_than_trajectory(self):
        with self.assertRaisesRegex(ValueError, "shorter than a trajectory"):
            dataset.D4RLIterableDataset(
                [_trajectory(0, length=5)],
                K=3,
                goal_conditioned=False,
                reward_conditioned=True,
                reward_episode_horizon=4,
            )


class PaddingLossTests(unittest.TestCase):
    def test_sb3_initialization_preserves_frozen_waypoint_parameters(self):
        layer = nn.Linear(3, 2)
        for parameter in layer.parameters():
            parameter.requires_grad = False
        before = {
            name: value.detach().clone()
            for name, value in layer.state_dict().items()
        }

        policies.ExtendedActorCriticPolicy.init_weights(layer, gain=2.0)

        for name, value in layer.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_policy_checkpoint_records_reward_and_waypoint_source_metadata(self):
        class FakeExtendedPolicy(nn.Module):
            def __init__(
                self,
                observation_space,
                action_space,
                lr_schedule,
                **kwargs,
            ):
                super().__init__()
                self.goal_network_ckpt = kwargs["goal_network_ckpt"]
                self.goal_network_config = kwargs["goal_network_config"]
                self.transformer_config = kwargs["transformer_config"]

        def capture_hyperparameter_names(module, *names):
            module._saved_hyperparameter_names = set(names)

        with mock.patch.dict(
            os.environ, {"GOAL_NETWORK_CKPT": "legacy.ckpt"}
        ), mock.patch.object(
            policies, "ExtendedActorCriticPolicy", FakeExtendedPolicy
        ), mock.patch.object(
            policies.RvS,
            "save_hyperparameters",
            capture_hyperparameter_names,
        ):
            model = policies.RvS(
                gym_spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                gym_spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
                reward_conditioning=True,
                reward_condition_dim=1,
                reward_representation="cumulative",
                reward_normalization_const=9000,
                reward_episode_horizon=1000,
                goal_network_ckpt="explicit.ckpt",
            )

        self.assertEqual(model.goal_network_ckpt, "explicit.ckpt")
        self.assertEqual(model.model.goal_network_ckpt, "explicit.ckpt")
        self.assertEqual(model.reward_representation, "cumulative")
        self.assertEqual(model.reward_normalization_const, 9000)
        self.assertEqual(model.reward_episode_horizon, 1000)
        self.assertEqual(
            model.model.goal_network_config["reward_representation"],
            "cumulative",
        )
        self.assertEqual(
            model.model.goal_network_config["reward_normalization_const"],
            9000,
        )
        self.assertEqual(
            model.model.goal_network_config["reward_episode_horizon"],
            1000,
        )
        self.assertIn("n_layer", model.model.transformer_config)
        self.assertIn("model_actions", model.model.transformer_config)
        self.assertIn(
            "goal_network_ckpt", model._saved_hyperparameter_names
        )
        self.assertIn(
            "reward_representation", model._saved_hyperparameter_names
        )
        self.assertIn(
            "reward_normalization_const",
            model._saved_hyperparameter_names,
        )
        self.assertIn(
            "reward_episode_horizon",
            model._saved_hyperparameter_names,
        )

    def test_waypoint_checkpoint_metadata_is_saved(self):
        model = goal_net.KForwardGoalNetwork(
            obs_dim=3,
            goal_dim=2,
            hidden_dim=11,
            max_T=30,
            reward=True,
            reward_representation="both",
            reward_normalization_const=9000,
            reward_episode_horizon=1000,
        )
        self.assertEqual(
            model._saved_hyperparameters,
            {
                "obs_dim": 3,
                "goal_dim": 2,
                "hidden_dim": 11,
                "max_T": 30,
                "recurrent": False,
                "learning_rate": 1e-3,
                "batch_size": 1024,
                "reward": True,
                "reward_representation": "both",
                "reward_normalization_const": 9000.0,
                "reward_episode_horizon": 1000,
            },
        )

    def test_waypoint_loss_excludes_padding_before_reduce(self):
        torch.manual_seed(1)
        model = goal_net.KForwardGoalNetwork(
            obs_dim=3,
            goal_dim=2,
            hidden_dim=8,
            max_T=30,
        )
        obs_goal = torch.randn(1, 5, 5)
        target = torch.randn(1, 5, 2)
        mask = torch.tensor([[False, False, True, True, True]])

        changed_padding = target.clone()
        changed_padding[:, :2] = 1e6
        loss_a = model.training_step((obs_goal, target, mask), 0)
        loss_b = model.training_step((obs_goal, changed_padding, mask), 0)
        torch.testing.assert_close(loss_a, loss_b)

        loss_a.backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            )
        )

    def test_all_padding_waypoint_loss_is_finite_and_backward_safe(self):
        model = goal_net.KForwardGoalNetwork(
            obs_dim=3,
            goal_dim=2,
            hidden_dim=8,
            max_T=30,
        )
        obs_goal = torch.randn(2, 4, 5)
        target = torch.randn(2, 4, 2)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        loss = model.training_step((obs_goal, target, mask), 0)

        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

    def test_policy_loss_excludes_padding_before_reduce(self):
        class FakePolicyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.center = nn.Parameter(torch.tensor(0.25))

            def evaluate_and_predict(self, obs, actions, attention_mask):
                log_probs = -torch.square(actions[:, 0] - self.center)
                prediction = self.center.expand_as(actions)
                zeros = torch.zeros_like(log_probs)
                return zeros, log_probs, zeros, prediction

        module = policies.RvS.__new__(policies.RvS)
        nn.Module.__init__(module)
        module.model = FakePolicyModel()
        obs_goal = torch.randn(1, 4, 5)
        mask = torch.tensor([[False, False, True, True]])
        action_a = torch.tensor([[[1e4], [-1e4], [1.0], [2.0]]])
        action_b = action_a.clone()
        action_b[:, :2] = torch.tensor([[[7.0], [9.0]]])

        loss_a = module.training_step((obs_goal, action_a, mask), 0)
        loss_b = module.training_step((obs_goal, action_b, mask), 0)
        torch.testing.assert_close(loss_a, loss_b)
        loss_a.backward()
        self.assertTrue(torch.isfinite(module.model.center.grad).item())


class TransformerMaskTests(unittest.TestCase):
    @staticmethod
    def _model(model_actions: bool, context_length: int = 5):
        config = GPTConfig(
            vocab_size=6,
            block_size=context_length * (2 if model_actions else 1),
            n_layer=2,
            n_head=2,
            n_embd=8,
            feature_dim=5,
            model_actions=model_actions,
            act_dim=3,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        return GPT(config)

    def _assert_padding_invariance(self, model_actions: bool):
        torch.manual_seed(2)
        model = self._model(model_actions)
        model.eval()
        with torch.no_grad():
            # Padding outputs must remain zero after the head has learned bias.
            model.head[0].bias.fill_(0.75)
        observations_a = torch.randn(2, 5, 5)
        observations_b = observations_a.clone()
        observations_b[:, :2] = torch.randn_like(observations_b[:, :2]) * 1e5
        actions_a = torch.randn(2, 5, 3)
        actions_b = actions_a.clone()
        actions_b[:, :2] = torch.randn_like(actions_b[:, :2]) * 1e5
        mask = torch.tensor(
            [[False, False, True, True, True], [False] * 5]
        )

        output_a = model((observations_a, actions_a, mask)).reshape(2, 5, -1)
        output_b = model((observations_b, actions_b, mask)).reshape(2, 5, -1)
        self.assertTrue(torch.isfinite(output_a).all().item())
        torch.testing.assert_close(output_a[0, 2:], output_b[0, 2:])
        torch.testing.assert_close(output_a[1], torch.zeros_like(output_a[1]))

    def test_transformer_padding_mask_without_action_tokens(self):
        self._assert_padding_invariance(model_actions=False)

    def test_transformer_padding_mask_with_action_tokens(self):
        self._assert_padding_invariance(model_actions=True)

    def test_transformer_action_dim_and_context_come_from_config(self):
        previous = os.environ.get("MODEL_ACTIONS")
        os.environ["MODEL_ACTIONS"] = "1"
        try:
            extractor = TransformerExtractor(
                feature_dim=5,
                net_arch=[],
                activation_fn=nn.ReLU,
                config={
                    "vocab_size": 6,
                    "n_head": 2,
                    "n_embd": 8,
                    "context_length": 7,
                },
                action_dim=3,
            )
        finally:
            if previous is None:
                os.environ.pop("MODEL_ACTIONS", None)
            else:
                os.environ["MODEL_ACTIONS"] = previous

        transformer = extractor.shared_net[0]
        self.assertEqual(transformer.config.act_dim, 3)
        self.assertEqual(transformer.config.block_size, 14)

    def test_single_step_context_clears_action_history_with_action_tokens(self):
        manager = goal_net.Manager(
            np.zeros(2, dtype=np.float32),
            K=1,
        )
        manager.update_obs(np.zeros(3, dtype=np.float32))
        manager.update_act(np.ones(3, dtype=np.float32))
        manager.update_obs(np.ones(3, dtype=np.float32))

        # With K=1 there is no previous action in the current context. GPT
        # supplies the current zero action token when actions is None.
        self.assertIsNone(manager.actions)
        model = self._model(model_actions=True, context_length=1)
        output = model(
            (
                torch.ones(1, 1, 5),
                manager.actions,
                torch.ones(1, 1, dtype=torch.bool),
            )
        )
        self.assertEqual(output.shape, (1, 6))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_learned_waypoint_is_registered_before_strict_state_load(self):
        checkpoint_metadata = {
            "hyper_parameters": {
                "obs_dim": 3,
                "goal_dim": 2,
                "hidden_dim": 8,
                "max_T": 30,
                "reward": False,
            }
        }

        def load_waypoint(*args, **kwargs):
            return goal_net.KForwardGoalNetwork(
                obs_dim=3,
                goal_dim=2,
                hidden_dim=8,
                max_T=30,
            )

        extractor_kwargs = {
            "feature_dim": 5,
            "net_arch": [],
            "activation_fn": nn.ReLU,
            "config": {
                "vocab_size": 6,
                "n_head": 2,
                "n_embd": 8,
                "context_length": 3,
            },
            "goal_network_ckpt": "waypoint.ckpt",
            "goal_network_config": {
                "state_dim": 3,
                "goal_dim": 2,
                "waypoint_dim": 2,
                "hidden_dim": 8,
                "max_T": 30,
                "reward": False,
            },
            "action_dim": 2,
        }
        with mock.patch.object(
            torch, "load", return_value=checkpoint_metadata
        ), mock.patch.object(
            goal_net.KForwardGoalNetwork,
            "load_from_checkpoint",
            side_effect=load_waypoint,
            create=True,
        ):
            source = TransformerExtractor(**extractor_kwargs)
        embedded_kwargs = dict(extractor_kwargs)
        embedded_kwargs["goal_network_ckpt"] = "missing-waypoint.ckpt"
        embedded_kwargs["goal_network_restore_mode"] = "embedded"
        with mock.patch.object(
            torch,
            "load",
            side_effect=AssertionError(
                "embedded restore must not reopen the waypoint file"
            ),
        ):
            restored = TransformerExtractor(**embedded_kwargs)

        state = source.state_dict()
        self.assertTrue(
            any(name.startswith("goal_network.net.") for name in state)
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in source.goal_network.parameters()
            )
        )
        incompatible = restored.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_outer_checkpoint_selects_embedded_waypoint_restore(self):
        lightning_base = policies.RvS.__mro__[1]
        captured = {}

        def fake_load_model_state(cls, checkpoint, strict=True, **kwargs):
            captured.update(kwargs)
            captured["strict"] = strict
            return "restored"

        checkpoint = {
            "state_dict": {
                "model.mlp_extractor.goal_network.net.0.weight": (
                    torch.zeros(2, 2)
                )
            }
        }
        with mock.patch.object(
            lightning_base,
            "_load_model_state",
            classmethod(fake_load_model_state),
            create=True,
        ):
            result = policies.RvS._load_model_state(
                checkpoint, strict=True
            )

        self.assertEqual(result, "restored")
        self.assertEqual(
            captured["_goal_network_restore_mode"], "embedded"
        )
        self.assertTrue(captured["strict"])

    def test_checkpoint_metadata_conflict_is_not_overwritten(self):
        checkpoint_metadata = {
            "hyper_parameters": {
                "obs_dim": 3,
                "goal_dim": 2,
                "hidden_dim": 8,
                "max_T": 30,
                "reward": False,
            }
        }
        extractor_kwargs = {
            "feature_dim": 5,
            "net_arch": [],
            "activation_fn": nn.ReLU,
            "config": {
                "vocab_size": 6,
                "n_head": 2,
                "n_embd": 8,
                "context_length": 3,
            },
            "goal_network_ckpt": "waypoint.ckpt",
            "goal_network_config": {
                "state_dim": 3,
                "goal_dim": 2,
                "waypoint_dim": 2,
                "hidden_dim": 8,
                "max_T": 30,
                "reward": True,
                "reward_representation": "both",
                "reward_normalization_const": 10.0,
                "reward_episode_horizon": 100,
            },
            "action_dim": 2,
        }
        with mock.patch.object(
            torch, "load", return_value=checkpoint_metadata
        ), mock.patch.object(
            goal_net.KForwardGoalNetwork,
            "load_from_checkpoint",
            side_effect=AssertionError(
                "conflicting metadata must fail before weight loading"
            ),
            create=True,
        ):
            with self.assertRaisesRegex(ValueError, "reward=False"):
                TransformerExtractor(**extractor_kwargs)


class RewardRolloutTests(unittest.TestCase):
    def test_online_context_uses_window_start_reward_command(self):
        real_step = _load_real_step_module()

        class FakePolicy:
            reward_conditioning = True
            reward_representation = "both"
            reward_normalization_const = 10.0
            reward_episode_horizon = 5
            context_length = 2

            def __init__(self):
                self.commands = []

            def get_action(
                self,
                observations,
                goals,
                deterministic=True,
                actions=None,
            ):
                self.commands.append(
                    np.asarray(goals[0], dtype=np.float32).copy()
                )
                return np.zeros(1, dtype=np.float32)

        class FakeEnv:
            _max_episode_steps = 5

            def __init__(self):
                self.rewards = [1.0, 2.0, 3.0]
                self.t = 0

            def reset(self):
                self.t = 0
                return np.zeros(3, dtype=np.float32)

            def step(self, action):
                reward = self.rewards[self.t]
                self.t += 1
                return (
                    np.full(3, self.t, dtype=np.float32),
                    reward,
                    self.t == len(self.rewards),
                    {},
                )

        policy = FakePolicy()
        rewards = real_step.sample_cumulative_reward(
            policy,
            FakeEnv(),
            goals=np.array([[2.0]], dtype=np.float32),
            trajectory_samples=1,
            reward_targets_are_average=True,
        )

        np.testing.assert_allclose(rewards, np.array([6.0]))
        np.testing.assert_allclose(
            policy.commands,
            np.array(
                [[2.0, 1.0], [2.0, 1.0], [2.25, 0.9]],
                dtype=np.float32,
            ),
        )

    def test_explicit_false_conditioning_beats_legacy_environment(self):
        real_step = _load_real_step_module()

        class GoalPolicy:
            reward_conditioning = False
            reward_representation = None

        with mock.patch.dict(
            os.environ,
            {"AVG_REWARD": "1", "CM_REWARD": "0"},
            clear=False,
        ):
            self.assertFalse(
                real_step._is_reward_conditioned(GoalPolicy(), None)
            )


class TrainingEntryPointTests(unittest.TestCase):
    def test_cli_reward_horizon_is_forwarded_to_training(self):
        source = (
            REPOSITORY_ROOT / "src" / "wt" / "train.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_training"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        forwarded = keywords["reward_episode_horizon"]
        self.assertIsInstance(forwarded, ast.Attribute)
        self.assertEqual(forwarded.attr, "reward_episode_horizon")


class TinyTrainingClosureTests(unittest.TestCase):
    def test_dataset_forward_loss_backward_optimizer_step(self):
        torch.manual_seed(4)
        source = _trajectory(0, length=8)
        iterable = dataset.D4RLIterableDataset(
            [source],
            index_batch_size=2,
            K=5,
            goal_conditioned=False,
            reward_conditioned=True,
            train_goal_net=True,
            waypoint_horizon=3,
        )
        components = iterable._fetch_trajectory_components(
            np.array([0, 0]), np.array([1, 5]), np.array([7, 7])
        )
        obs_goal = torch.cat(
            [components["states"], components["goals"]], dim=-1
        )
        model = goal_net.KForwardGoalNetwork(
            obs_dim=3,
            goal_dim=2,
            hidden_dim=16,
            max_T=3,
            reward=True,
            reward_representation="both",
            reward_normalization_const=iterable.normalization_const,
            reward_episode_horizon=iterable.reward_episode_horizon,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss = model.training_step(
            (obs_goal, components["waypoints"], components["mask"]), 0
        )
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            )
        )

    def test_transformer_forward_masked_loss_backward_optimizer_step(self):
        torch.manual_seed(5)
        model = TransformerMaskTests._model(model_actions=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        observations = torch.randn(2, 5, 5)
        actions = torch.randn(2, 5, 3)
        mask = torch.tensor(
            [[False, True, True, True, True], [False, False, True, True, True]]
        )
        target = torch.randn(2, 5, 6)

        optimizer.zero_grad()
        prediction = model((observations, actions, mask)).reshape(2, 5, 6)
        per_timestep = torch.square(prediction - target).mean(dim=-1)
        loss = per_timestep[mask].mean()
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
