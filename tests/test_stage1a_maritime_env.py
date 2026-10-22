"""Regression tests for the stage 1A minimal maritime environment.

The suite intentionally uses only the Python standard library and NumPy.  Its
numbered tests map directly to the thirty mandatory checks in the stage 1A
specification; the remaining tests cover validation and interface invariants
that are needed before trajectory collection is designed in stage 1B.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from wt.maritime.channel import (  # noqa: E402
    compute_channel_metrics,
    theoretical_max_rates,
)
from wt.maritime.config import MaritimeEnvConfig  # noqa: E402
from wt.maritime.env import MaritimeEnv  # noqa: E402


def _channel_kwargs(config: MaritimeEnvConfig) -> dict[str, float]:
    """Return the common keyword arguments for a channel calculation."""
    return {
        "uav_altitude_m": config.uav_altitude,
        "total_bandwidth_hz": config.total_bandwidth_hz,
        "buoy_tx_power_w": config.buoy_tx_power_w,
        "reference_channel_gain": config.reference_channel_gain,
        "pathloss_exponent": config.pathloss_exponent,
        "noise_power_w": config.noise_power_w,
    }


class MaritimeConfigAndLayoutTests(unittest.TestCase):
    """Configuration validation, seeding, and buoy-layout checks."""

    def test_01_default_configuration_is_valid_and_frozen(self) -> None:
        config = MaritimeEnvConfig()
        self.assertEqual(config.num_buoys, 4)
        self.assertEqual(config.episode_horizon, 20)
        with self.assertRaises(FrozenInstanceError):
            config.num_buoys = 5  # type: ignore[misc]

    def test_02_invalid_area_dimensions_raise_value_error(self) -> None:
        for kwargs in (
            {"area_width": 0.0},
            {"area_width": -1.0},
            {"area_height": 0.0},
            {"area_height": -1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MaritimeEnvConfig(**kwargs)

    def test_03_invalid_buoy_count_raises_value_error(self) -> None:
        for value in (0, -1):
            with self.subTest(num_buoys=value), self.assertRaises(ValueError):
                MaritimeEnvConfig(num_buoys=value)

    def test_04_reset_is_deterministic_for_equal_seed(self) -> None:
        env = MaritimeEnv(MaritimeEnvConfig(randomize_buoys=True))
        observation_a, info_a = env.reset(seed=1977)
        observation_b, info_b = env.reset(seed=1977)
        np.testing.assert_array_equal(observation_a, observation_b)
        np.testing.assert_array_equal(
            info_a["buoy_positions_m"], info_b["buoy_positions_m"]
        )

    def test_05_random_layout_changes_for_different_seeds(self) -> None:
        env = MaritimeEnv(MaritimeEnvConfig(randomize_buoys=True))
        _, info_a = env.reset(seed=11)
        _, info_b = env.reset(seed=12)
        self.assertFalse(
            np.array_equal(info_a["buoy_positions_m"], info_b["buoy_positions_m"])
        )

    def test_06_default_four_buoy_layout_is_exact(self) -> None:
        env = MaritimeEnv()
        _, info = env.reset()
        expected = np.asarray(
            [[50.0, 50.0], [50.0, 150.0], [150.0, 50.0], [150.0, 150.0]]
        )
        np.testing.assert_array_equal(info["buoy_positions_m"], expected)

    def test_31_invalid_dynamics_parameters_raise_value_error(self) -> None:
        for kwargs in (
            {"episode_horizon": 0},
            {"delta_t": 0.0},
            {"delta_t": -1.0},
            {"uav_altitude": 0.0},
            {"uav_altitude": -1.0},
            {"max_speed": 0.0},
            {"max_speed": -1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MaritimeEnvConfig(**kwargs)

    def test_32_negative_energy_parameters_raise_value_error(self) -> None:
        for field in (
            "initial_energy",
            "hover_energy_per_step",
            "move_energy_per_meter",
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                MaritimeEnvConfig(**{field: -0.01})

    def test_33_invalid_communication_parameters_raise_value_error(self) -> None:
        for field in (
            "total_bandwidth_hz",
            "buoy_tx_power_w",
            "reference_channel_gain",
            "pathloss_exponent",
            "noise_power_w",
        ):
            for value in (0.0, -1.0):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        MaritimeEnvConfig(**{field: value})

    def test_34_invalid_margin_or_initial_position_raises_value_error(self) -> None:
        for kwargs in (
            {"buoy_margin": -1.0},
            {"area_width": 100.0, "buoy_margin": 51.0},
            {"area_height": 100.0, "buoy_margin": 51.0},
            {"initial_uav_x": -0.01},
            {"initial_uav_x": 200.01},
            {"initial_uav_y": -0.01},
            {"initial_uav_y": 200.01},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MaritimeEnvConfig(**kwargs)

    def test_35_nondefault_grid_is_deterministic_and_respects_margin(self) -> None:
        config = MaritimeEnvConfig(num_buoys=7, randomize_buoys=False)
        env = MaritimeEnv(config)
        _, info_a = env.reset(seed=1)
        _, info_b = env.reset(seed=999)
        positions = info_a["buoy_positions_m"]
        np.testing.assert_array_equal(positions, info_b["buoy_positions_m"])
        self.assertEqual(positions.shape, (7, 2))
        self.assertTrue(np.all(positions[:, 0] >= config.buoy_margin))
        self.assertTrue(
            np.all(positions[:, 0] <= config.area_width - config.buoy_margin)
        )
        self.assertTrue(np.all(positions[:, 1] >= config.buoy_margin))
        self.assertTrue(
            np.all(positions[:, 1] <= config.area_height - config.buoy_margin)
        )

    def test_36_environment_does_not_mutate_global_numpy_rng(self) -> None:
        previous_state = np.random.get_state()
        try:
            np.random.seed(12345)
            expected = np.random.random(5)
            np.random.seed(12345)
            env = MaritimeEnv(MaritimeEnvConfig(randomize_buoys=True))
            env.reset(seed=88)
            actual = np.random.random(5)
            np.testing.assert_array_equal(actual, expected)
        finally:
            np.random.set_state(previous_state)


class MaritimeChannelTests(unittest.TestCase):
    """Independent channel-model tests."""

    def setUp(self) -> None:
        self.config = MaritimeEnvConfig()
        self.buoy = np.asarray([[0.0, 0.0]])

    def _metrics(self, uav_xy: np.ndarray) -> dict[str, object]:
        return compute_channel_metrics(
            uav_xy,
            self.buoy,
            **_channel_kwargs(self.config),
        )

    def test_19_channel_gain_decreases_with_horizontal_distance(self) -> None:
        near = self._metrics(np.asarray([0.0, 0.0]))
        far = self._metrics(np.asarray([100.0, 0.0]))
        self.assertGreater(near["channel_gains"][0], far["channel_gains"][0])

    def test_20_rate_decreases_with_horizontal_distance(self) -> None:
        near = self._metrics(np.asarray([0.0, 0.0]))
        far = self._metrics(np.asarray([100.0, 0.0]))
        self.assertGreater(near["rates_bps"][0], far["rates_bps"][0])

    def test_21_all_channel_outputs_are_finite_and_nonnegative(self) -> None:
        buoys = np.asarray([[0.0, 0.0], [50.0, 75.0], [200.0, 200.0]])
        metrics = compute_channel_metrics(
            np.asarray([100.0, 100.0]),
            buoys,
            **_channel_kwargs(self.config),
        )
        for key in (
            "horizontal_distances_m",
            "distances_3d_m",
            "channel_gains",
            "snr_linear",
            "rates_bps",
        ):
            values = metrics[key]
            self.assertTrue(np.all(np.isfinite(values)), key)
            self.assertTrue(np.all(values >= 0.0), key)
        self.assertTrue(np.isfinite(metrics["total_rate_bps"]))
        self.assertGreaterEqual(metrics["total_rate_bps"], 0.0)

    def test_37_channel_result_shapes_match_buoy_count(self) -> None:
        count = 5
        metrics = compute_channel_metrics(
            np.asarray([1.0, 2.0]),
            np.zeros((count, 2), dtype=np.float64),
            **_channel_kwargs(self.config),
        )
        for key in (
            "horizontal_distances_m",
            "distances_3d_m",
            "channel_gains",
            "snr_linear",
            "rates_bps",
        ):
            self.assertEqual(metrics[key].shape, (count,), key)
        self.assertIsInstance(metrics["total_rate_bps"], float)

    def test_38_invalid_channel_input_shape_or_values_raise(self) -> None:
        valid_uav = np.asarray([0.0, 0.0])
        valid_buoys = np.zeros((2, 2))
        cases = (
            (np.asarray([0.0]), valid_buoys),
            (np.asarray([[0.0, 0.0]]), valid_buoys),
            (valid_uav, np.asarray([0.0, 0.0])),
            (valid_uav, np.zeros((2, 3))),
            (np.asarray([np.nan, 0.0]), valid_buoys),
            (valid_uav, np.asarray([[np.inf, 0.0]])),
        )
        for uav, buoys in cases:
            with self.subTest(uav=uav.shape, buoys=buoys.shape):
                with self.assertRaises((TypeError, ValueError)):
                    compute_channel_metrics(
                        uav, buoys, **_channel_kwargs(self.config)
                    )

    def test_39_theoretical_rate_uses_zero_horizontal_distance(self) -> None:
        single_rate, total_rate = theoretical_max_rates(
            num_buoys=self.config.num_buoys,
            **_channel_kwargs(self.config),
        )
        colocated = np.repeat(
            np.asarray([[100.0, 100.0]]), self.config.num_buoys, axis=0
        )
        metrics = compute_channel_metrics(
            np.asarray([100.0, 100.0]),
            colocated,
            **_channel_kwargs(self.config),
        )
        self.assertAlmostEqual(metrics["rates_bps"][0], single_rate)
        self.assertAlmostEqual(metrics["total_rate_bps"], total_rate)


class MaritimeEnvironmentTests(unittest.TestCase):
    """State, action, reward, and lifecycle checks."""

    def test_07_state_dim_tracks_number_of_buoys(self) -> None:
        for count in (1, 3, 7):
            with self.subTest(count=count):
                env = MaritimeEnv(MaritimeEnvConfig(num_buoys=count))
                self.assertEqual(env.state_dim, 6 + 4 * count)

    def test_08_default_state_dim_is_22(self) -> None:
        env = MaritimeEnv()
        self.assertEqual(env.state_dim, 22)
        self.assertEqual(env.action_dim, 2)
        self.assertEqual(env.num_buoys, 4)

    def test_09_observation_shape_and_dtype_are_exact(self) -> None:
        env = MaritimeEnv()
        observation, _ = env.reset()
        self.assertEqual(observation.shape, (env.state_dim,))
        self.assertEqual(observation.dtype, np.float32)

    def test_10_observations_have_no_nan_or_inf(self) -> None:
        env = MaritimeEnv()
        observation, _ = env.reset()
        self.assertTrue(np.all(np.isfinite(observation)))
        for action in (np.asarray([1.0, 1.0]), np.asarray([-1.0, 0.25])):
            observation, _, _, _, _ = env.step(action)
            self.assertTrue(np.all(np.isfinite(observation)))

    def test_11_invalid_action_shapes_raise_value_error(self) -> None:
        cases = (
            np.asarray(0.0),
            np.asarray([0.0]),
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([[0.0, 0.0]]),
        )
        for action in cases:
            env = MaritimeEnv()
            env.reset()
            with self.subTest(shape=action.shape), self.assertRaises(ValueError):
                env.step(action)

    def test_12_nan_action_raises_value_error(self) -> None:
        env = MaritimeEnv()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(np.asarray([np.nan, 0.0]))

    def test_13_inf_action_raises_value_error(self) -> None:
        env = MaritimeEnv()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(np.asarray([0.0, np.inf]))

    def test_14_action_norm_is_radially_projected(self) -> None:
        env = MaritimeEnv()
        env.reset()
        _, _, _, _, info = env.step(np.asarray([1.0, 1.0]))
        applied = info["normalized_applied_action"]
        self.assertAlmostEqual(float(np.linalg.norm(applied)), 1.0)
        self.assertTrue(info["action_norm_projected"])
        self.assertFalse(info["action_component_clipped"])

    def test_15_single_step_displacement_respects_speed_limit(self) -> None:
        env = MaritimeEnv()
        env.reset()
        _, _, _, _, info = env.step(np.asarray([9.0, -9.0]))
        maximum = env.config.max_speed * env.config.delta_t
        actual = float(np.linalg.norm(info["physical_displacement_m"]))
        self.assertLessEqual(actual, maximum + 1.0e-12)

    def test_16_out_of_bounds_position_is_projected_to_boundary(self) -> None:
        config = MaritimeEnvConfig(initial_uav_x=199.0)
        env = MaritimeEnv(config)
        env.reset()
        _, _, _, _, info = env.step(np.asarray([1.0, 0.0]))
        np.testing.assert_allclose(info["candidate_position_m"], [209.0, 100.0])
        np.testing.assert_allclose(info["applied_position_m"], [200.0, 100.0])
        self.assertAlmostEqual(info["boundary_projection_distance_m"], 9.0)

    def test_17_boundary_event_is_visible_in_info(self) -> None:
        env = MaritimeEnv(MaritimeEnvConfig(initial_uav_x=199.0))
        env.reset()
        _, _, terminated, _, info = env.step(np.asarray([1.0, 0.0]))
        self.assertTrue(info["boundary_violation"])
        self.assertFalse(terminated)
        for key in (
            "candidate_position_m",
            "applied_position_m",
            "boundary_projection_distance_m",
        ):
            self.assertIn(key, info)

    def test_18_boundary_violation_reduces_reward_by_configured_penalty(self) -> None:
        base = MaritimeEnvConfig(initial_uav_x=199.0, boundary_penalty=0.0)
        penalized = replace(base, boundary_penalty=0.75)
        env_base = MaritimeEnv(base)
        env_penalized = MaritimeEnv(penalized)
        env_base.reset()
        env_penalized.reset()
        _, reward_base, _, _, _ = env_base.step(np.asarray([1.0, 0.0]))
        _, reward_penalized, _, _, _ = env_penalized.step(
            np.asarray([1.0, 0.0])
        )
        self.assertAlmostEqual(reward_base - reward_penalized, 0.75)

    def test_22_zero_action_has_zero_movement_cost(self) -> None:
        env = MaritimeEnv()
        env.reset()
        _, _, _, _, info = env.step(np.zeros(2))
        self.assertEqual(info["movement_cost"], 0.0)

    def test_23_nonzero_action_decreases_remaining_energy(self) -> None:
        env = MaritimeEnv()
        env.reset()
        _, _, _, _, info = env.step(np.asarray([0.5, 0.0]))
        expected_cost = (
            env.config.hover_energy_per_step
            + env.config.move_energy_per_meter
            * np.linalg.norm(info["physical_displacement_m"])
        )
        self.assertAlmostEqual(
            env.config.initial_energy - info["remaining_energy"], expected_cost
        )

    def test_24_energy_depletion_sets_terminated(self) -> None:
        config = MaritimeEnvConfig(
            initial_energy=0.05,
            hover_energy_per_step=0.1,
            episode_horizon=10,
        )
        env = MaritimeEnv(config)
        env.reset()
        _, _, terminated, truncated, info = env.step(np.zeros(2))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["remaining_energy"], 0.0)

    def test_25_horizon_sets_truncated(self) -> None:
        env = MaritimeEnv(MaritimeEnvConfig(episode_horizon=2))
        env.reset()
        _, _, terminated_1, truncated_1, _ = env.step(np.zeros(2))
        _, _, terminated_2, truncated_2, _ = env.step(np.zeros(2))
        self.assertFalse(terminated_1)
        self.assertFalse(truncated_1)
        self.assertFalse(terminated_2)
        self.assertTrue(truncated_2)

    def test_26_step_after_episode_end_raises_runtime_error(self) -> None:
        env = MaritimeEnv(MaritimeEnvConfig(episode_horizon=1))
        env.reset()
        env.step(np.zeros(2))
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(2))

    def test_27_signed_reward_components_sum_to_reward(self) -> None:
        env = MaritimeEnv(MaritimeEnvConfig(initial_uav_x=199.0))
        env.reset()
        _, reward, _, _, info = env.step(np.asarray([1.0, 0.0]))
        self.assertIsInstance(reward, float)
        self.assertAlmostEqual(
            sum(float(value) for value in info["reward_components"].values()),
            reward,
        )
        for key in ("rate_utility", "movement_cost", "boundary_cost", "energy_cost"):
            self.assertIn(key, info)

    def test_28_random_actions_complete_one_episode(self) -> None:
        env = MaritimeEnv()
        observation, _ = env.reset(seed=2023)
        generator = np.random.default_rng(2023)
        steps = 0
        while True:
            observation, reward, terminated, truncated, _ = env.step(
                generator.uniform(-1.5, 1.5, size=2)
            )
            steps += 1
            self.assertTrue(np.all(np.isfinite(observation)))
            self.assertTrue(np.isfinite(reward))
            if terminated or truncated:
                break
        self.assertEqual(steps, env.config.episode_horizon)
        self.assertTrue(truncated)

    def test_29_one_hundred_short_episodes_remain_finite(self) -> None:
        config = MaritimeEnvConfig(
            num_buoys=3,
            episode_horizon=3,
            randomize_buoys=True,
        )
        env = MaritimeEnv(config)
        generator = np.random.default_rng(771)
        for episode in range(100):
            observation, info = env.reset(seed=episode)
            self.assertTrue(np.all(np.isfinite(observation)))
            self.assertTrue(np.isfinite(info["total_rate_bps"]))
            while True:
                observation, reward, terminated, truncated, info = env.step(
                    generator.uniform(-2.0, 2.0, size=2)
                )
                self.assertTrue(np.all(np.isfinite(observation)))
                self.assertTrue(np.isfinite(reward))
                self.assertTrue(np.all(np.isfinite(info["rates_bps"])))
                if terminated or truncated:
                    break

    def test_40_component_clipping_and_projection_are_fully_reported(self) -> None:
        env = MaritimeEnv()
        env.reset()
        raw = np.asarray([2.0, -3.0])
        _, _, _, _, info = env.step(raw)
        np.testing.assert_array_equal(info["raw_action"], raw)
        np.testing.assert_array_equal(
            info["component_clipped_action"], np.asarray([1.0, -1.0])
        )
        self.assertTrue(info["action_component_clipped"])
        self.assertTrue(info["action_norm_projected"])
        self.assertLessEqual(
            np.linalg.norm(info["normalized_applied_action"]), 1.0 + 1.0e-12
        )

    def test_41_reset_info_contains_physical_state_and_safe_copies(self) -> None:
        env = MaritimeEnv()
        observation, info = env.reset()
        required = {
            "time_step",
            "uav_position_m",
            "uav_velocity_mps",
            "remaining_energy",
            "buoy_positions_m",
            "rates_bps",
            "total_rate_bps",
            "channel_gains",
            "snr_linear",
            "state_dim",
            "action_dim",
        }
        self.assertTrue(required.issubset(info))
        self.assertEqual(info["time_step"], 0)
        self.assertEqual(info["state_dim"], observation.size)
        expected_uav_position = env.uav_position_m
        expected_buoy_positions = env.buoy_positions_m
        info["uav_position_m"][0] = -999.0
        info["buoy_positions_m"][0, 0] = -999.0
        np.testing.assert_array_equal(env.uav_position_m, expected_uav_position)
        np.testing.assert_array_equal(
            env.buoy_positions_m, expected_buoy_positions
        )

    def test_42_normalized_observation_fields_stay_in_their_ranges(self) -> None:
        env = MaritimeEnv()
        observation, _ = env.reset()
        self.assertTrue(np.all(observation[:4] >= -1.0))
        self.assertTrue(np.all(observation[:4] <= 1.0))
        self.assertGreaterEqual(observation[4], 0.0)
        self.assertLessEqual(observation[4], 1.0)
        self.assertGreaterEqual(observation[5], 0.0)
        self.assertLessEqual(observation[5], 1.0)
        buoy_features = observation[6:].reshape(env.num_buoys, 4)
        self.assertTrue(np.all(np.abs(buoy_features[:, :2]) <= 1.0))
        self.assertTrue(np.all((buoy_features[:, 2:] >= 0.0)))
        self.assertTrue(np.all((buoy_features[:, 2:] <= 1.0)))

    def test_43_buoys_remain_static_during_episode(self) -> None:
        env = MaritimeEnv(MaritimeEnvConfig(randomize_buoys=True))
        _, reset_info = env.reset(seed=123)
        original = reset_info["buoy_positions_m"].copy()
        for _ in range(3):
            _, _, _, _, step_info = env.step(np.asarray([0.2, -0.3]))
            np.testing.assert_array_equal(step_info["buoy_positions_m"], original)

    def test_44_stage1a_does_not_define_a_goal_interface(self) -> None:
        env = MaritimeEnv()
        _, info = env.reset()
        self.assertFalse(hasattr(env, "goal_dim"))
        self.assertNotIn("goal", info)
        self.assertNotIn("waypoint", info)


class MaritimeDependencyIsolationTests(unittest.TestCase):
    """Static and import-time checks for the stage 1A dependency boundary."""

    FORBIDDEN_IMPORTS = {
        "gym",
        "gymnasium",
        "torch",
        "pytorch_lightning",
        "d4rl",
        "mujoco",
        "stable_baselines3",
        "torch_geometric",
        "dgl",
        "cvxpy",
    }

    def test_30_maritime_sources_do_not_import_prohibited_dependencies(self) -> None:
        maritime_root = SOURCE_ROOT / "wt" / "maritime"
        for source_path in sorted(maritime_root.glob("*.py")):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            imported_roots: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(
                        alias.name.partition(".")[0] for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    imported_roots.add((node.module or "").partition(".")[0])
            with self.subTest(source=source_path.name):
                self.assertFalse(imported_roots & self.FORBIDDEN_IMPORTS)

    def test_45_clean_import_does_not_load_prohibited_dependencies(self) -> None:
        source_literal = repr(str(SOURCE_ROOT))
        forbidden_literal = repr(sorted(self.FORBIDDEN_IMPORTS))
        script = (
            "import sys\n"
            f"sys.path.insert(0, {source_literal})\n"
            "before = set(sys.modules)\n"
            "import wt.maritime\n"
            "import wt.maritime.channel\n"
            "import wt.maritime.config\n"
            "import wt.maritime.env\n"
            "import wt.maritime.smoke\n"
            "loaded = set(sys.modules) - before\n"
            f"forbidden = set({forbidden_literal})\n"
            "bad = sorted({name.partition('.')[0] for name in loaded} & forbidden)\n"
            "assert not bad, bad\n"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
