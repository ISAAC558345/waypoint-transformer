"""Round-02 tests for non-learning baselines and offline episode data.

The tests intentionally exercise only deterministic rules, environment
rollouts, serialization, auditing, metrics, and plotting.  No learning model,
optimizer, or backward pass is constructed.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from wt.maritime.baselines import (  # noqa: E402
    MAX_QUEUE,
    NEAREST_BUOY,
    RANDOM_VALID,
    URGENCY_RATE_ENERGY,
    available_baseline_policies,
    create_baseline_policy,
)
from wt.maritime.config import MaritimeEnvConfig  # noqa: E402
from wt.maritime.dataset_io import (  # noqa: E402
    assert_roundtrip_exact,
    deterministic_episode_split,
    load_dataset,
    save_dataset,
    validate_dataset,
    validate_saved_dataset,
)
from wt.maritime.env import MaritimeEnv  # noqa: E402
from wt.maritime.k_step_audit import (  # noqa: E402
    audit_k_steps,
    enumerate_k_step_pairs,
    get_k_step_state_pair,
)
from wt.maritime.metrics import (  # noqa: E402
    compute_episode_metrics,
    summarize_episode_metrics,
)
from wt.maritime.plotting import FIGURE_FILENAMES  # noqa: E402
from wt.maritime.runner import (  # noqa: E402
    collect_baseline_episodes,
    run_baseline_experiment,
)


POLICY_NAMES = (
    RANDOM_VALID,
    NEAREST_BUOY,
    MAX_QUEUE,
    URGENCY_RATE_ENERGY,
)


def _policy_info(
    config: MaritimeEnvConfig, info: dict[str, object]
) -> dict[str, object]:
    result = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in info.items()
    }
    result["valid_node_mask"] = np.ones(config.num_buoys, dtype=np.bool_)
    result["max_step_distance_m"] = config.max_speed * config.delta_t
    return result


class NonLearningBaselineTests(unittest.TestCase):
    """Policy legality, determinism, scale, mask, and fallback checks."""

    def test_01_all_required_policies_are_registered(self) -> None:
        self.assertEqual(available_baseline_policies(), POLICY_NAMES)
        for name in POLICY_NAMES:
            policy = create_baseline_policy(name, seed=17)
            self.assertFalse(policy.metadata["trainable"])
            self.assertFalse(policy.metadata["uses_optimizer"])
            self.assertFalse(policy.metadata["uses_future_information"])

    def test_02_actions_are_finite_and_legal_at_variable_scales(self) -> None:
        for num_buoys in (1, 3, 7):
            config = MaritimeEnvConfig(
                num_buoys=num_buoys, randomize_buoys=True
            )
            observation, reset_info = MaritimeEnv(config).reset(seed=23)
            info = _policy_info(config, reset_info)
            for name in POLICY_NAMES:
                with self.subTest(policy=name, num_buoys=num_buoys):
                    action = create_baseline_policy(name, seed=9).act(
                        observation, info
                    )
                    self.assertEqual(action.shape, (int(info["action_dim"]),))
                    self.assertTrue(np.all(np.isfinite(action)))
                    self.assertLessEqual(float(np.linalg.norm(action)), 1.0 + 1e-12)

    def test_03_policies_use_dynamic_action_dimension(self) -> None:
        config = MaritimeEnvConfig(num_buoys=2)
        observation, reset_info = MaritimeEnv(config).reset(seed=3)
        info = _policy_info(config, reset_info)
        info["action_dim"] = 5
        for name in POLICY_NAMES:
            with self.subTest(policy=name):
                action = create_baseline_policy(name, seed=4).act(
                    observation, info
                )
                self.assertEqual(action.shape, (5,))
                self.assertTrue(np.all(np.isfinite(action)))
                self.assertLessEqual(float(np.linalg.norm(action)), 1.0 + 1e-12)

    def test_04_deterministic_policies_reproduce_exact_actions(self) -> None:
        config = MaritimeEnvConfig(num_buoys=5, randomize_buoys=True)
        observation, reset_info = MaritimeEnv(config).reset(seed=91)
        info = _policy_info(config, reset_info)
        for name in (NEAREST_BUOY, MAX_QUEUE, URGENCY_RATE_ENERGY):
            with self.subTest(policy=name):
                first = create_baseline_policy(name, seed=13)
                second = create_baseline_policy(name, seed=13)
                np.testing.assert_array_equal(
                    first.act(observation, info),
                    second.act(observation, info),
                )

    def test_05_random_policy_has_private_reproducible_rng(self) -> None:
        config = MaritimeEnvConfig(num_buoys=3)
        observation, reset_info = MaritimeEnv(config).reset(seed=8)
        info = _policy_info(config, reset_info)
        first = create_baseline_policy(RANDOM_VALID, seed=101)
        second = create_baseline_policy(RANDOM_VALID, seed=101)
        first_values = [first.act(observation, info) for _ in range(5)]
        second_values = [second.act(observation, info) for _ in range(5)]
        for left, right in zip(first_values, second_values):
            np.testing.assert_array_equal(left, right)

        np.random.seed(777)
        expected_global_draw = np.random.random()
        np.random.seed(777)
        create_baseline_policy(RANDOM_VALID, seed=4).act(observation, info)
        actual_global_draw = np.random.random()
        self.assertEqual(expected_global_draw, actual_global_draw)

    def test_06_no_valid_node_returns_safe_targeting_actions(self) -> None:
        config = MaritimeEnvConfig(num_buoys=4)
        observation, reset_info = MaritimeEnv(config).reset(seed=2)
        info = _policy_info(config, reset_info)
        info["valid_node_mask"] = np.zeros(config.num_buoys, dtype=np.bool_)
        info["remaining_queue"] = np.ones(config.num_buoys)
        for name in (NEAREST_BUOY, MAX_QUEUE, URGENCY_RATE_ENERGY):
            with self.subTest(policy=name):
                action = create_baseline_policy(name, seed=6).act(
                    observation, info
                )
                np.testing.assert_array_equal(
                    action, np.zeros(int(info["action_dim"]))
                )

    def test_07_masks_and_stable_queue_ties_are_respected(self) -> None:
        config = MaritimeEnvConfig(num_buoys=3)
        observation, reset_info = MaritimeEnv(config).reset(seed=2)
        info = _policy_info(config, reset_info)
        info["uav_position_m"] = np.array([0.0, 0.0])
        info["buoy_positions_m"] = np.array(
            [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]
        )
        info["horizontal_distances_m"] = np.array([1.0, 2.0, 3.0])
        info["rates_bps"] = np.array([100.0, 10.0, 20.0])
        info["valid_node_mask"] = np.array([False, True, True])

        nearest = create_baseline_policy(NEAREST_BUOY, seed=0)
        nearest.act(observation, info)
        self.assertEqual(
            nearest.metadata["last_decision"]["selected_buoy_index"], 1
        )

        info["remaining_queue"] = np.array([100.0, 7.0, 7.0])
        maximum = create_baseline_policy(MAX_QUEUE, seed=0)
        maximum.act(observation, info)
        self.assertEqual(
            maximum.metadata["last_decision"]["selected_buoy_index"], 1
        )

        heuristic = create_baseline_policy(URGENCY_RATE_ENERGY, seed=0)
        heuristic.act(observation, info)
        selected = heuristic.metadata["last_decision"]["selected_buoy_index"]
        self.assertIn(selected, (1, 2))
        self.assertNotEqual(selected, 0)

    def test_08_missing_queue_is_never_fabricated(self) -> None:
        config = MaritimeEnvConfig(num_buoys=3)
        observation, reset_info = MaritimeEnv(config).reset(seed=2)
        info = _policy_info(config, reset_info)
        policy = create_baseline_policy(MAX_QUEUE, seed=0)
        action = policy.act(observation, info)
        np.testing.assert_array_equal(
            action, np.zeros(int(info["action_dim"]))
        )
        decision = policy.metadata["last_decision"]
        self.assertFalse(decision["queue_available"])
        self.assertEqual(decision["status"], "hover_queue_unavailable")


class OfflineTrajectoryPipelineTests(unittest.TestCase):
    """Shared runner, storage, split, K-step, and metric checks."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = MaritimeEnvConfig(
            episode_horizon=6,
            randomize_buoys=True,
        )
        cls.episodes = collect_baseline_episodes(
            env_config=cls.config,
            policy_names=POLICY_NAMES,
            scenario_id="round02_unit",
            episodes_per_policy=3,
            seeds=(7, 11, 13),
            buoy_counts=(2, 4),
            max_steps=3,
        )

    def test_09_runner_uses_the_same_comparison_schedule(self) -> None:
        self.assertEqual(len(self.episodes), 24)
        schedules: dict[str, list[tuple[int, int]]] = {}
        for name in POLICY_NAMES:
            schedules[name] = sorted(
                (episode.num_buoys, episode.seed)
                for episode in self.episodes
                if episode.policy_id == name
            )
        for name in POLICY_NAMES[1:]:
            self.assertEqual(schedules[name], schedules[POLICY_NAMES[0]])

        grouped: dict[tuple[int, int], list[object]] = {}
        for episode in self.episodes:
            grouped.setdefault((episode.num_buoys, episode.seed), []).append(
                episode
            )
        for case_episodes in grouped.values():
            reference = case_episodes[0]
            for episode in case_episodes[1:]:
                np.testing.assert_array_equal(
                    episode.observations[0], reference.observations[0]
                )
                np.testing.assert_array_equal(
                    episode.buoy_positions_m[0],
                    reference.buoy_positions_m[0],
                )

    def test_10_transition_count_ids_and_continuity_are_exact(self) -> None:
        self.assertEqual(
            sum(episode.episode_length for episode in self.episodes), 72
        )
        episode_ids = [episode.episode_id for episode in self.episodes]
        self.assertEqual(len(episode_ids), len(set(episode_ids)))
        for episode in self.episodes:
            self.assertEqual(episode.episode_length, 3)
            np.testing.assert_array_equal(
                episode.next_observations[:-1], episode.observations[1:]
            )
            np.testing.assert_array_equal(
                episode.next_uav_positions_m[:-1],
                episode.uav_positions_m[1:],
            )
            np.testing.assert_array_equal(
                episode.next_remaining_energy[:-1],
                episode.remaining_energy[1:],
            )
            np.testing.assert_array_equal(
                episode.next_rates_bps[:-1], episode.rates_bps[1:]
            )

    def test_11_runner_marks_collector_truncation_without_termination(self) -> None:
        for episode in self.episodes:
            self.assertEqual(episode.terminal_reason, "collector_limit")
            self.assertFalse(bool(episode.terminated[-1]))
            self.assertTrue(bool(episode.truncated[-1]))
            self.assertFalse(np.any(episode.terminated[:-1]))
            self.assertFalse(np.any(episode.truncated[:-1]))

    def test_12_all_numeric_arrays_and_actions_are_finite(self) -> None:
        for episode in self.episodes:
            for name, array in episode.arrays().items():
                with self.subTest(episode=episode.episode_id, field=name):
                    self.assertTrue(np.all(np.isfinite(array)))
            norms = np.linalg.norm(episode.actions, axis=1)
            self.assertTrue(np.all(norms <= 1.0 + 1e-12))

    def test_13_unsupported_variables_have_explicit_empty_semantics(self) -> None:
        for episode in self.episodes:
            self.assertFalse(episode.queue_values_available)
            self.assertEqual(
                episode.queue_values.shape, (episode.episode_length, 0)
            )
            self.assertFalse(episode.transmitted_payload_available)
            self.assertEqual(
                episode.transmitted_payload_bits.shape,
                (episode.episode_length, 0),
            )
            self.assertFalse(episode.desired_goal_available)
            self.assertEqual(episode.desired_goal.shape, (0,))
            self.assertTrue(episode.achieved_goal_available)
            self.assertEqual(episode.achieved_goal.shape, (3,))

    def test_14_episode_split_is_disjoint_and_group_leakage_free(self) -> None:
        assignments = deterministic_episode_split(
            self.episodes, seed=90210
        )
        split_sets = {
            split: {
                episode_id
                for episode_id, assigned in assignments.items()
                if assigned == split
            }
            for split in ("train", "validation", "test")
        }
        self.assertTrue(all(split_sets.values()))
        self.assertTrue(split_sets["train"].isdisjoint(split_sets["validation"]))
        self.assertTrue(split_sets["train"].isdisjoint(split_sets["test"]))
        self.assertTrue(
            split_sets["validation"].isdisjoint(split_sets["test"])
        )
        group_to_split: dict[tuple[str, int], str] = {}
        for episode in self.episodes:
            key = (episode.scenario_id, episode.seed)
            assigned = assignments[episode.episode_id]
            self.assertEqual(group_to_split.setdefault(key, assigned), assigned)
        report = validate_dataset(self.episodes, assignments)
        self.assertTrue(report["passed"])
        for split in ("train", "validation", "test"):
            composition = report["split_composition"][split]
            self.assertEqual(
                set(composition["num_buoys_composition"]), {"2", "4"}
            )
            self.assertEqual(
                set(composition["policy_composition"]), set(POLICY_NAMES)
            )

    def test_15_save_reload_is_exact_and_pickle_free(self) -> None:
        assignments = deterministic_episode_split(self.episodes, seed=14)
        with tempfile.TemporaryDirectory(
            prefix="t", dir=REPOSITORY_ROOT / "round_02"
        ) as temporary:
            output = Path(temporary)
            manifest = save_dataset(
                output,
                self.episodes,
                split_assignments=assignments,
                metadata={"training_performed": False},
            )
            loaded = load_dataset(output)
            assert_roundtrip_exact(self.episodes, loaded)
            validation = validate_saved_dataset(output)
            self.assertTrue(validation["passed"])
            self.assertTrue(validation["roundtrip_exact"])
            self.assertEqual(manifest["episode_count"], len(self.episodes))
            self.assertEqual(manifest["transition_count"], 72)
            self.assertTrue((output / "dataset_schema.json").is_file())
            with (output / "dataset_schema.json").open(
                "r", encoding="utf-8"
            ) as handle:
                schema = json.load(handle)
            self.assertFalse(
                schema["unavailable_semantics"]["queue_values"]["available"]
            )
            alignment = schema["transition_time_alignment"]
            self.assertIn("info_before", alignment["raw_state_before"])
            self.assertIn("post-action", alignment["step_link_capacity_source"])
            self.assertIn(
                "not actual delivered payload",
                alignment["step_link_capacity_source"],
            )
            for entry in manifest["episodes"]:
                with np.load(
                    output / entry["npz_file"], allow_pickle=False
                ) as archive:
                    self.assertNotIn("object", {str(archive[key].dtype) for key in archive})

    def test_16_k_step_pairs_have_correct_endpoints(self) -> None:
        episode = self.episodes[0]
        pairs_k1 = enumerate_k_step_pairs(episode, 1)
        pairs_k3 = enumerate_k_step_pairs(episode, 3)
        pairs_k5 = enumerate_k_step_pairs(episode, 5)
        self.assertEqual(len(pairs_k1), 3)
        self.assertEqual(len(pairs_k3), 1)
        self.assertEqual(len(pairs_k5), 0)
        source, target = get_k_step_state_pair(episode, pairs_k3[0])
        np.testing.assert_array_equal(source, episode.observations[0])
        np.testing.assert_array_equal(target, episode.next_observations[2])
        self.assertEqual(pairs_k3[0].episode_id, episode.episode_id)

    def test_17_k_step_enumerator_rejects_terminal_crossing_and_padding(self) -> None:
        malformed = SimpleNamespace(
            episode_id="boundary_probe",
            episode_length=5,
            valid_transition_mask=np.ones(5, dtype=np.bool_),
            terminated=np.array([False, True, False, False, False]),
            truncated=np.zeros(5, dtype=np.bool_),
            observations=np.arange(10, dtype=np.float32).reshape(5, 2),
            next_observations=np.arange(10, 20, dtype=np.float32).reshape(5, 2),
        )
        starts = {
            pair.start_timestep for pair in enumerate_k_step_pairs(malformed, 3)
        }
        self.assertNotIn(0, starts)
        self.assertNotIn(1, starts)

        padded = SimpleNamespace(
            episode_id="padding_probe",
            episode_length=4,
            valid_transition_mask=np.array([True, False, True, True]),
            terminated=np.zeros(4, dtype=np.bool_),
            truncated=np.array([False, False, False, True]),
            observations=np.zeros((4, 2), dtype=np.float32),
            next_observations=np.zeros((4, 2), dtype=np.float32),
        )
        padded_starts = {
            pair.start_timestep for pair in enumerate_k_step_pairs(padded, 2)
        }
        self.assertNotIn(0, padded_starts)
        self.assertNotIn(1, padded_starts)

    def test_18_k_step_audit_covers_policy_scale_and_split(self) -> None:
        assignments = deterministic_episode_split(self.episodes, seed=5)
        rows = audit_k_steps(
            self.episodes,
            split_assignments=assignments,
            k_values=(1, 5, 10, 20, 30),
        )
        self.assertEqual({row["k"] for row in rows}, {1, 5, 10, 20, 30})
        self.assertEqual(
            {row["policy_id"] for row in rows}, set(POLICY_NAMES)
        )
        self.assertEqual({row["num_buoys"] for row in rows}, {2, 4})
        self.assertEqual(
            {row["split"] for row in rows},
            {"train", "validation", "test"},
        )
        self.assertTrue(
            all(row["terminal_boundary_violation_count"] == 0 for row in rows)
        )
        self.assertTrue(
            all(row["invalid_padding_count"] == 0 for row in rows)
        )

    def test_19_metrics_use_capacity_proxy_and_mark_tasks_unavailable(self) -> None:
        rows = [
            compute_episode_metrics(
                episode, delta_t=self.config.delta_t
            )
            for episode in self.episodes
        ]
        summaries = summarize_episode_metrics(rows)
        self.assertEqual(len(summaries), 4)
        for row in rows:
            self.assertGreater(row["cumulative_link_capacity_bits"], 0.0)
            self.assertFalse(row["cumulative_collected_data_available"])
            self.assertIsNone(row["cumulative_collected_data_bits"])
            self.assertFalse(row["final_remaining_queue_available"])
            self.assertIsNone(row["final_remaining_queue"])
            self.assertEqual(row["nan_inf_count"], 0)

    def test_20_bounded_experiment_writes_all_required_artifacts_and_figures(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="t", dir=REPOSITORY_ROOT / "round_02"
        ) as temporary:
            root = Path(temporary)
            output = root / "o"
            figures = root / "f"
            result = run_baseline_experiment(
                env_config=MaritimeEnvConfig(
                    episode_horizon=2, randomize_buoys=True
                ),
                policy_names=POLICY_NAMES,
                scenario_id="round02_artifact_test",
                episodes_per_policy=3,
                seeds=(3, 5, 7),
                buoy_counts=(2, 3),
                max_steps=2,
                output_dir=output,
                figures_dir=figures,
                run_id="artifact_test",
                mode="smoke",
            )
            required = {
                "baseline_episode_metrics.csv",
                "baseline_summary.csv",
                "dataset_manifest.json",
                "dataset_schema.json",
                "k_step_audit.csv",
                "run_config.json",
            }
            self.assertTrue(all((output / name).is_file() for name in required))
            self.assertEqual(
                {path.name for path in result.figure_paths.values()},
                set(FIGURE_FILENAMES.values()),
            )
            self.assertTrue(
                all(path.is_file() and path.stat().st_size > 0 for path in result.figure_paths.values())
            )
            with (output / "run_config.json").open(
                "r", encoding="utf-8"
            ) as handle:
                run_config = json.load(handle)
            self.assertFalse(run_config["training_performed"])
            self.assertEqual(run_config["episode_count"], 24)
            self.assertEqual(run_config["transition_count"], 48)
            with (output / "baseline_episode_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metric_rows), 24)
            self.assertTrue(
                all(row["final_remaining_queue"] == "" for row in metric_rows)
            )

    def test_21_no_save_mode_still_audits_and_plots(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="t", dir=REPOSITORY_ROOT / "round_02"
        ) as temporary:
            root = Path(temporary)
            result = run_baseline_experiment(
                env_config=MaritimeEnvConfig(episode_horizon=1),
                policy_names=(MAX_QUEUE,),
                episodes_per_policy=1,
                seeds=(2,),
                buoy_counts=(2,),
                max_steps=1,
                save_trajectories=False,
                output_dir=root / "o",
                figures_dir=root / "f",
                run_id="no_save",
                mode="bounded",
            )
            self.assertTrue(result.split_assignments)
            self.assertFalse((root / "o" / "dataset_manifest.json").exists())
            self.assertTrue((root / "o" / "k_step_audit.csv").is_file())
            self.assertEqual(len(result.figure_paths), 6)

    def test_22_round02_modules_do_not_import_learning_frameworks(self) -> None:
        module_names = (
            "baselines.py",
            "trajectory.py",
            "dataset_io.py",
            "k_step_audit.py",
            "metrics.py",
            "plotting.py",
            "runner.py",
        )
        forbidden_roots = {
            "torch",
            "tensorflow",
            "pytorch_lightning",
            "stable_baselines3",
        }
        for module_name in module_names:
            source = (SOURCE_ROOT / "wt" / "maritime" / module_name).read_text(
                encoding="utf-8"
            )
            tree = ast.parse(source)
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            with self.subTest(module=module_name):
                self.assertTrue(imported.isdisjoint(forbidden_roots))


if __name__ == "__main__":
    unittest.main()
