"""Round-3 queue/task, shared-trace, v1 storage, and Waypoint tests.

No test constructs or trains a learning model.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from wt.maritime.arrival import materialize_arrival_trace  # noqa: E402
from wt.maritime.baselines import (  # noqa: E402
    MAX_QUEUE,
    NEAREST_BUOY,
    QUEUE_RATE_ENERGY,
    RANDOM_VALID,
    available_baseline_policies,
    create_baseline_policy,
)
from wt.maritime.config import MaritimeEnvConfig  # noqa: E402
from wt.maritime.dataset_v1 import (  # noqa: E402
    assert_roundtrip_exact_v1,
    deterministic_episode_split_v1,
    load_dataset_v1,
    save_dataset_v1,
    validate_saved_dataset_v1,
)
from wt.maritime.env import MaritimeEnv  # noqa: E402
from wt.maritime.round3_analysis import (  # noqa: E402
    build_data_quality_audit,
    compute_action_agreement_matrix,
    compute_episode_metric_rows_v1,
)
from wt.maritime.round3_runner import (  # noqa: E402
    ROUND3_POLICY_NAMES,
    collect_round3_episodes,
)
from wt.maritime.waypoint_samples import (  # noqa: E402
    build_waypoint_sample_dataset,
    load_waypoint_samples,
    save_waypoint_samples,
    validate_saved_waypoint_samples,
    validate_waypoint_samples,
    waypoint_dim,
    waypoint_state_dim,
)


def _v1_config(**kwargs: object) -> MaritimeEnvConfig:
    values = {
        "task_semantics_version": "v1",
        "num_buoys": 3,
        "episode_horizon": 4,
        "arrival_mode": "constant",
        "constant_arrival_bits_per_step": 1000.0,
    }
    values.update(kwargs)
    return MaritimeEnvConfig(**values)


def _policy_probe_info() -> tuple[np.ndarray, dict[str, object]]:
    observation = np.zeros(30, dtype=np.float32)
    info: dict[str, object] = {
        "state_dim": 30,
        "action_dim": 2,
        "num_buoys": 3,
        "uav_position_m": np.array([0.0, 0.0]),
        "buoy_positions_m": np.array(
            [[1.0, 0.0], [0.0, 2.0], [-3.0, -3.0]]
        ),
        "horizontal_distances_m": np.array([1.0, 2.0, np.sqrt(18.0)]),
        "rates_bps": np.array([1.0, 1.0, 100.0]),
        "valid_node_mask": np.ones(3, dtype=np.bool_),
        "raw_queue_bits": np.array([1.0, 10.0, 5.0]),
        "one_step_movement_energy_proxy": np.array([0.2, 1.0, 0.1]),
    }
    return observation, info


class Round3EnvironmentSemanticsTests(unittest.TestCase):
    def test_01_v1_dimensions_are_dynamic(self) -> None:
        for count in (1, 3, 7):
            env = MaritimeEnv(_v1_config(num_buoys=count))
            self.assertEqual(env.state_dim, 9 + 7 * count)
            self.assertEqual(env.action_dim, 2)

    def test_02_observation_space_matches_observation(self) -> None:
        env = MaritimeEnv(_v1_config())
        observation, info = env.reset(seed=4)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["observation_space_shape"], observation.shape)
        self.assertEqual(
            info["observation_schema"]["state_dim"], observation.size
        )

    def test_03_constant_arrival_materializes_exact_bits(self) -> None:
        config = _v1_config(constant_arrival_bits_per_step=1234.0)
        trace, identifier = materialize_arrival_trace(config, seed=2)
        np.testing.assert_array_equal(trace, np.full((4, 3), 1234.0))
        self.assertEqual(len(identifier), 64)

    def test_04_poisson_arrival_is_seed_reproducible(self) -> None:
        config = replace(_v1_config(), arrival_mode="poisson")
        first, first_id = materialize_arrival_trace(config, seed=91)
        second, second_id = materialize_arrival_trace(config, seed=91)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_id, second_id)

    def test_05_different_poisson_seeds_have_different_trace_ids(self) -> None:
        config = replace(_v1_config(), arrival_mode="poisson")
        _, first_id = materialize_arrival_trace(config, seed=91)
        _, second_id = materialize_arrival_trace(config, seed=92)
        self.assertNotEqual(first_id, second_id)

    def test_06_service_before_arrival_does_not_send_future_data(self) -> None:
        config = _v1_config(
            num_buoys=1,
            initial_queue_bits=0.0,
            constant_arrival_bits_per_step=1000.0,
        )
        env = MaritimeEnv(config)
        env.reset(seed=1)
        _, _, _, _, info = env.step(np.zeros(2))
        self.assertEqual(float(info["transmitted_payload_bits"][0]), 0.0)
        self.assertEqual(float(info["raw_queue_bits"][0]), 1000.0)

    def test_07_queue_update_is_conservative(self) -> None:
        env = MaritimeEnv(_v1_config())
        _, before = env.reset(seed=1)
        _, _, _, _, after = env.step(np.array([0.2, -0.1]))
        expected = (
            before["raw_queue_bits"]
            + after["raw_arrival_bits"]
            - after["transmitted_payload_bits"]
            - after["dropped_arrival_bits"]
        )
        np.testing.assert_allclose(after["raw_queue_bits"], expected)
        self.assertLess(
            float(np.max(np.abs(after["queue_conservation_residual_bits"]))),
            1.0e-6,
        )

    def test_08_queue_never_becomes_negative(self) -> None:
        env = MaritimeEnv(_v1_config())
        env.reset(seed=2)
        for _ in range(4):
            _, _, terminated, truncated, info = env.step(np.zeros(2))
            self.assertTrue(np.all(info["raw_queue_bits"] >= 0.0))
            if terminated or truncated:
                break

    def test_09_queue_never_exceeds_capacity(self) -> None:
        config = _v1_config(
            queue_capacity_bits=2000.0,
            initial_queue_bits=2000.0,
            constant_arrival_bits_per_step=1.0e8,
        )
        env = MaritimeEnv(config)
        env.reset(seed=2)
        _, _, _, _, info = env.step(np.zeros(2))
        self.assertTrue(np.all(info["raw_queue_bits"] <= 2000.0))
        self.assertTrue(np.any(info["dropped_arrival_bits"] > 0.0))

    def test_10_payload_does_not_exceed_available_data(self) -> None:
        env = MaritimeEnv(_v1_config())
        env.reset(seed=3)
        _, _, _, _, info = env.step(np.zeros(2))
        self.assertTrue(
            np.all(
                info["transmitted_payload_bits"]
                <= info["available_data_bits"] + 1.0e-6
            )
        )

    def test_11_payload_does_not_exceed_capacity(self) -> None:
        env = MaritimeEnv(_v1_config())
        env.reset(seed=3)
        _, _, _, _, info = env.step(np.zeros(2))
        self.assertTrue(
            np.all(
                info["transmitted_payload_bits"]
                <= info["link_capacity_bits"] + 1.0e-6
            )
        )

    def test_12_raw_and_normalized_queue_are_inverse(self) -> None:
        config = _v1_config()
        _, info = MaritimeEnv(config).reset(seed=3)
        np.testing.assert_allclose(
            info["normalized_queue"] * config.queue_capacity_bits,
            info["raw_queue_bits"],
        )

    def test_13_observation_contains_only_last_realized_arrival(self) -> None:
        config = _v1_config(constant_arrival_bits_per_step=2000.0)
        env = MaritimeEnv(config)
        observation, _ = env.reset(seed=3)
        nodes = observation[9:].reshape(3, 7)
        np.testing.assert_array_equal(nodes[:, 5], np.zeros(3))
        next_observation, _, _, _, info = env.step(np.zeros(2))
        expected = 2000.0 / (2000.0 + config.queue_capacity_bits)
        np.testing.assert_allclose(
            next_observation[9:].reshape(3, 7)[:, 5], expected
        )
        np.testing.assert_array_equal(
            info["raw_arrival_bits"], np.full(3, 2000.0)
        )

    def test_14_cumulative_payload_matches_step_payload(self) -> None:
        env = MaritimeEnv(_v1_config())
        env.reset(seed=5)
        _, _, _, _, info = env.step(np.zeros(2))
        self.assertAlmostEqual(
            info["cumulative_payload_bits"],
            float(np.sum(info["transmitted_payload_bits"])),
        )

    def test_15_reward_components_sum_and_use_actual_payload(self) -> None:
        env = MaritimeEnv(_v1_config())
        env.reset(seed=5)
        _, reward, _, _, info = env.step(np.zeros(2))
        self.assertAlmostEqual(
            reward, sum(info["reward_components"].values())
        )
        self.assertAlmostEqual(
            info["payload_utility"],
            float(np.sum(info["transmitted_payload_bits"]))
            / (env._max_total_rate_bps * env.config.delta_t),
        )

    def test_16_horizon_terminal_reason_is_explicit(self) -> None:
        env = MaritimeEnv(_v1_config(episode_horizon=1))
        env.reset(seed=5)
        _, _, terminated, truncated, info = env.step(np.zeros(2))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["terminal_reason"], "horizon")

    def test_17_task_success_can_terminate_early(self) -> None:
        config = _v1_config(
            num_buoys=1,
            episode_horizon=5,
            initial_queue_bits=1.0,
            constant_arrival_bits_per_step=0.0,
            target_task_completion_ratio=1.0,
            maximum_allowed_residual_queue_ratio=0.0,
            minimum_terminal_energy_ratio=0.0,
        )
        env = MaritimeEnv(config)
        env.reset(seed=5)
        _, _, terminated, _, info = env.step(np.zeros(2))
        self.assertTrue(terminated)
        self.assertTrue(info["task_success"])
        self.assertEqual(info["terminal_reason"], "task_success")

    def test_18_energy_depletion_reason_is_explicit(self) -> None:
        config = _v1_config(
            initial_energy=0.01,
            hover_energy_per_step=0.1,
            terminate_on_task_success=False,
        )
        env = MaritimeEnv(config)
        env.reset(seed=5)
        _, _, terminated, _, info = env.step(np.zeros(2))
        self.assertTrue(terminated)
        self.assertEqual(info["terminal_reason"], "energy_depleted")

    def test_19_undefined_communication_energy_is_not_fabricated(self) -> None:
        _, info = MaritimeEnv(_v1_config()).reset(seed=1)
        self.assertFalse(info["communication_energy_available"])
        self.assertIn("not defined", info["communication_energy_status"])

    def test_20_desired_goal_has_stable_three_fields(self) -> None:
        _, info = MaritimeEnv(_v1_config()).reset(seed=1)
        self.assertEqual(info["desired_goal"].shape, (3,))
        self.assertEqual(len(info["desired_goal_field_names"]), 3)


class Round3BaselineTests(unittest.TestCase):
    def test_21_v1_registry_uses_queue_name_not_urgency_name(self) -> None:
        self.assertEqual(available_baseline_policies("v1"), ROUND3_POLICY_NAMES)
        self.assertIn(QUEUE_RATE_ENERGY, ROUND3_POLICY_NAMES)
        self.assertNotIn("urgency_rate_energy", ROUND3_POLICY_NAMES)

    def test_22_nearest_skips_empty_queue(self) -> None:
        observation, info = _policy_probe_info()
        info["raw_queue_bits"] = np.array([0.0, 2.0, 3.0])
        policy = create_baseline_policy(NEAREST_BUOY)
        policy.act(observation, info)
        self.assertEqual(
            policy.metadata["last_decision"]["selected_buoy_index"], 1
        )

    def test_23_max_queue_selects_real_largest_queue(self) -> None:
        observation, info = _policy_probe_info()
        policy = create_baseline_policy(MAX_QUEUE)
        policy.act(observation, info)
        self.assertEqual(
            policy.metadata["last_decision"]["selected_buoy_index"], 1
        )

    def test_24_max_queue_tie_breaks_by_distance_then_index(self) -> None:
        observation, info = _policy_probe_info()
        info["raw_queue_bits"] = np.array([0.0, 5.0, 5.0])
        info["horizontal_distances_m"] = np.array([9.0, 2.0, 1.0])
        policy = create_baseline_policy(MAX_QUEUE)
        policy.act(observation, info)
        self.assertEqual(
            policy.metadata["last_decision"]["selected_buoy_index"], 2
        )
        info["horizontal_distances_m"] = np.array([9.0, 1.0, 1.0])
        policy.act(observation, info)
        self.assertEqual(
            policy.metadata["last_decision"]["selected_buoy_index"], 1
        )

    def test_25_queue_rate_energy_matches_public_formula(self) -> None:
        observation, info = _policy_probe_info()
        policy = create_baseline_policy(QUEUE_RATE_ENERGY)
        policy.act(observation, info)
        self.assertEqual(
            policy.metadata["last_decision"]["selected_buoy_index"], 2
        )
        self.assertFalse(policy.metadata["urgency_used"])

    def test_26_four_policies_produce_distinct_constructed_actions(self) -> None:
        observation, info = _policy_probe_info()
        actions = [
            create_baseline_policy(name, seed=77).act(observation, info)
            for name in (
                RANDOM_VALID,
                NEAREST_BUOY,
                MAX_QUEUE,
                QUEUE_RATE_ENERGY,
            )
        ]
        for left in range(len(actions)):
            for right in range(left + 1, len(actions)):
                self.assertFalse(
                    np.allclose(actions[left], actions[right], atol=1.0e-8)
                )

    def test_26b_random_valid_respects_current_motion_bounds(self) -> None:
        config = _v1_config(initial_uav_x=0.0, initial_uav_y=0.0)
        env = MaritimeEnv(config)
        observation, info = env.reset(seed=7)
        policy = create_baseline_policy(RANDOM_VALID, seed=19)
        bounds = np.asarray(info["area_bounds_m"], dtype=np.float64)
        max_step = float(info["max_step_distance_m"])
        position = np.asarray(info["uav_position_m"], dtype=np.float64)
        for _ in range(256):
            action = policy.act(observation, info)
            candidate = position + action[:2] * max_step
            self.assertTrue(np.all(candidate >= -1.0e-12))
            self.assertTrue(np.all(candidate <= bounds + 1.0e-12))

    def test_26c_runner_preserves_boundary_safe_random_actions(self) -> None:
        collection = collect_round3_episodes(
            seeds=(9,),
            buoy_counts=(5,),
            horizon=128,
        )
        episode = next(
            value
            for value in collection.episodes
            if value.policy_id == RANDOM_VALID
        )
        self.assertEqual(int(np.count_nonzero(episode.boundary_violation)), 0)


class Round3PipelineAndWaypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.collection = collect_round3_episodes(
            seeds=(1, 2, 3),
            buoy_counts=(3, 5),
            horizon=32,
        )
        cls.episodes = list(cls.collection.episodes)
        cls.assignments = deterministic_episode_split_v1(
            cls.episodes, seed=44
        )
        cls.waypoint_result = build_waypoint_sample_dataset(
            cls.episodes,
            cls.assignments,
            max_num_buoys=5,
        )

    def test_27_shared_trace_is_identical_across_policies(self) -> None:
        groups: dict[tuple[int, int], set[str]] = {}
        for episode in self.episodes:
            groups.setdefault((episode.seed, episode.num_buoys), set()).add(
                episode.arrival_trace_id
            )
        self.assertTrue(all(len(values) == 1 for values in groups.values()))

    def test_28_different_scenario_seeds_use_different_traces(self) -> None:
        by_seed: dict[int, set[str]] = {}
        for episode in self.episodes:
            by_seed.setdefault(episode.seed, set()).add(
                episode.arrival_trace_id
            )
        flattened = {next(iter(values)) for values in by_seed.values()}
        self.assertEqual(len(flattened), 3)

    def test_29_v1_trajectory_payload_matches_environment_fields(self) -> None:
        for episode in self.episodes:
            self.assertEqual(episode.schema_version, "v1")
            self.assertTrue(
                np.all(
                    episode.transmitted_payload_bits
                    <= episode.raw_queue_bits + 1.0e-6
                )
            )
            self.assertTrue(
                np.all(
                    episode.transmitted_payload_bits
                    <= episode.link_capacity_bits + 1.0e-6
                )
            )

    def test_30_split_groups_do_not_leak(self) -> None:
        group_splits: dict[tuple[str, int], str] = {}
        for episode in self.episodes:
            key = (episode.scenario_id, episode.seed)
            split = self.assignments[episode.episode_id]
            self.assertEqual(group_splits.setdefault(key, split), split)

    def test_31_dataset_save_reload_is_exact(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="t", dir=REPOSITORY_ROOT / "round_03"
        ) as temporary:
            root = Path(temporary)
            save_dataset_v1(
                root,
                self.episodes,
                self.assignments,
                metadata={"training_performed": False},
            )
            loaded = load_dataset_v1(root)
            assert_roundtrip_exact_v1(self.episodes, loaded)
            report = validate_saved_dataset_v1(root)
            self.assertTrue(report["passed"])
            self.assertTrue(report["roundtrip_exact"])

    def test_32_v1_loader_rejects_round2_schema(self) -> None:
        legacy = (
            REPOSITORY_ROOT
            / "round_02"
            / "outputs"
            / "round02_smoke_20260730_212701_283649"
        )
        with self.assertRaises(ValueError):
            load_dataset_v1(legacy)

    def test_33_waypoint_exact_k_counts_for_full_episodes(self) -> None:
        totals: dict[int, int] = {}
        for row in self.waypoint_result.k_statistics:
            totals[int(row["k"])] = totals.get(int(row["k"]), 0) + int(
                row["valid_sample_count"]
            )
        self.assertEqual(
            totals,
            {
                1: 24 * 32,
                5: 24 * 28,
                10: 24 * 23,
                20: 24 * 13,
                30: 24 * 3,
            },
        )

    def test_34_k30_has_real_samples(self) -> None:
        count = sum(
            int(np.sum(batch.k == 30))
            for batch in self.waypoint_result.batches.values()
        )
        self.assertGreater(count, 0)

    def test_35_waypoint_padding_mask_zeroes_padded_nodes(self) -> None:
        batch = self.waypoint_result.batches["train"]
        rows = np.flatnonzero(batch.num_buoys == 3)
        self.assertGreater(rows.size, 0)
        n_max = 5
        queue_slice = slice(2, 2 + n_max)
        completion_slice = slice(2 + n_max, 2 + 2 * n_max)
        self.assertTrue(
            np.all(
                ~batch.waypoint_mask_t_plus_k[
                    rows[:, None], np.array([5, 6, 10, 11])[None, :]
                ]
            )
        )
        self.assertTrue(
            np.all(
                batch.waypoint_target_t_plus_k[rows, queue_slice][
                    :, 3:
                ]
                == 0.0
            )
        )
        self.assertTrue(
            np.all(
                batch.waypoint_target_t_plus_k[rows, completion_slice][
                    :, 3:
                ]
                == 0.0
            )
        )

    def test_36_waypoint_samples_never_cross_split(self) -> None:
        report = validate_waypoint_samples(
            self.waypoint_result.batches,
            expected_episode_splits=self.assignments,
        )
        self.assertEqual(report["split_leakage_count"], 0)

    def test_37_goal_sources_remain_distinct(self) -> None:
        desired = build_waypoint_sample_dataset(
            self.episodes,
            self.assignments,
            k_values=(1,),
            goal_source="desired_goal",
            max_num_buoys=5,
        )
        achieved_batch = self.waypoint_result.batches["train"]
        desired_batch = desired.batches["train"]
        self.assertTrue(np.all(desired_batch.goal_source == "desired_goal"))
        self.assertTrue(
            np.all(
                desired_batch.goal_mask[:, :2]
                == np.zeros_like(desired_batch.goal_mask[:, :2])
            )
        )
        self.assertFalse(
            np.array_equal(
                desired_batch.goal[0], achieved_batch.goal[0]
            )
        )

    def test_38_waypoint_save_load_checksums_and_no_pickle(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="t", dir=REPOSITORY_ROOT / "round_03"
        ) as temporary:
            root = Path(temporary)
            manifest = save_waypoint_samples(root, self.waypoint_result)
            loaded = load_waypoint_samples(root)
            report = validate_saved_waypoint_samples(root)
            self.assertEqual(
                manifest["sample_count"], report["sample_count"]
            )
            self.assertTrue(report["roundtrip_exact"])
            for path in root.glob("waypoint_samples_*.npz"):
                with np.load(path, allow_pickle=False) as archive:
                    self.assertTrue(
                        all(not archive[name].dtype.hasobject for name in archive.files)
                    )
            self.assertEqual(set(loaded), {"train", "validation", "test"})

    def test_39_all_episode_and_waypoint_arrays_are_finite(self) -> None:
        for episode in self.episodes:
            self.assertTrue(
                all(
                    np.all(np.isfinite(array))
                    for array in episode.arrays().values()
                )
            )
        for batch in self.waypoint_result.batches.values():
            self.assertTrue(np.all(np.isfinite(batch.state_t)))
            self.assertTrue(np.all(np.isfinite(batch.goal)))
            self.assertTrue(
                np.all(np.isfinite(batch.waypoint_target_t_plus_k))
            )

    def test_40_variable_scale_waypoint_dimensions_are_fixed(self) -> None:
        self.assertEqual(waypoint_dim(5), 13)
        self.assertEqual(waypoint_state_dim(5), 62)
        for batch in self.waypoint_result.batches.values():
            self.assertEqual(batch.state_t.shape[1], 62)
            self.assertEqual(batch.goal.shape[1], 13)
            self.assertEqual(
                batch.waypoint_target_t_plus_k.shape[1], 13
            )

    def test_41_nearest_and_queue_rate_are_not_aliases(self) -> None:
        rows = compute_action_agreement_matrix(self.episodes)
        matching = [
            row
            for row in rows
            if {
                row["policy_a"],
                row["policy_b"],
            }
            == {NEAREST_BUOY, QUEUE_RATE_ENERGY}
        ]
        self.assertEqual(len(matching), 2)
        off_diagonal = [
            row for row in matching if row["policy_a"] != row["policy_b"]
        ]
        self.assertEqual(len(off_diagonal), 2)
        self.assertTrue(
            all(float(row["agreement_rate"]) < 1.0 for row in off_diagonal)
        )

    def test_42_max_queue_is_not_default_hover(self) -> None:
        selected = [
            episode
            for episode in self.episodes
            if episode.policy_id == MAX_QUEUE
        ]
        actions = np.concatenate([episode.actions for episode in selected])
        nonhover = np.linalg.norm(actions, axis=1) > 1.0e-12
        self.assertGreater(float(np.mean(nonhover)), 0.0)

    def test_43_quality_audit_records_no_physical_integrity_error(self) -> None:
        metrics = compute_episode_metric_rows_v1(self.episodes)
        rows = build_data_quality_audit(
            self.episodes,
            episode_metrics=metrics,
            split_assignments=self.assignments,
            k_statistics=self.waypoint_result.k_statistics,
            action_agreement=compute_action_agreement_matrix(self.episodes),
        )
        integrity = [
            row
            for row in rows
            if row["metric"] == "numeric_mask_and_conservation_integrity"
        ]
        self.assertEqual(len(integrity), 1)
        value = integrity[0]["value"]
        self.assertEqual(value["nan_inf_count"], 0)
        self.assertEqual(value["negative_queue_count"], 0)
        self.assertEqual(value["queue_conservation_violation_count"], 0)

    def test_44_all_manifests_keep_training_false(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="t", dir=REPOSITORY_ROOT / "round_03"
        ) as temporary:
            root = Path(temporary)
            dataset_manifest = save_dataset_v1(
                root,
                self.episodes,
                self.assignments,
                metadata={"training_performed": False},
            )
            self.assertFalse(dataset_manifest["training_performed"])
            self.assertFalse(
                dataset_manifest["user_metadata"]["training_performed"]
            )
        self.assertEqual(
            set(ROUND3_POLICY_NAMES),
            {RANDOM_VALID, NEAREST_BUOY, MAX_QUEUE, QUEUE_RATE_ENERGY},
        )


if __name__ == "__main__":
    unittest.main()
