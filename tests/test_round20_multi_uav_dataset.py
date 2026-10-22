"""Focused contracts for the deterministic non-learning Round-20 pipeline."""
from __future__ import annotations

import compileall
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.round19.multi_env import SharedSpectrumMultiUAVEnv, SharedSpectrumOptions, assignment_link_metrics, gain_matrix
from wt.round20.audit import R19_LOCK_SHA256, permute_buoys, permute_uavs, preflight, split_audit, task_waypoint_audits
from wt.round20.dataset import BEHAVIORS, build_task_waypoint_samples, plain, run_episode
from wt.round20.scenarios import development_scenarios, untouched_evaluation_scenarios


class Round20MultiUAVDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = development_scenarios()
        cls.shared_scenario = next(item for item in cls.scenarios if item.spectrum_tier == "low" and item.num_buoys == 3)
        cls.episode = run_episode(cls.shared_scenario, BEHAVIORS[0])
        cls.episode["quality_tier"] = "high"
        cls.samples, cls.normalization = build_task_waypoint_samples([cls.episode])
        cls.graph = cls.episode["transitions"][1]["graph_state_t"]

    def test_01_round19_lock_unchanged(self):
        self.assertEqual(hashlib.sha256((ROOT / "round_19/ROUND_LOCK.json").read_bytes()).hexdigest().upper(), R19_LOCK_SHA256)

    def test_02_no_optimizer_constructed(self):
        self.assertNotIn("optimizer", "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src/wt/round20").glob("*.py")))

    def test_03_no_backward_constructed(self):
        self.assertNotIn("backward", "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src/wt/round20").glob("*.py")))

    def test_04_no_neural_checkpoint(self):
        self.assertFalse(list((ROOT / "src/wt/round20").glob("*.pt")))

    def test_05_environment_physics_unchanged(self):
        config = self.shared_scenario.config()
        self.assertEqual((config.total_bandwidth_hz, config.buoy_tx_power_w, config.noise_power_w, config.reference_channel_gain, config.pathloss_exponent), (1.0e6, .1, 1.0e-9, 1.0e-3, 2.2))

    def test_06_shared_spectrum_formula_available(self):
        config = self.shared_scenario.config()
        gains, _ = gain_matrix(np.asarray(self.shared_scenario.uav_positions_m), np.asarray(self.shared_scenario.buoy_positions_m), config)
        desired, interference, sinr, _, rate = assignment_link_metrics(np.array((0, 1)), gains, config, .25)
        self.assertTrue(np.allclose(sinr, desired / (config.noise_power_w + interference)) and np.all(rate >= 0.0))

    def test_07_matching_source_unchanged(self):
        lock = json.loads((ROOT / "round_19/ROUND_LOCK.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256((ROOT / "src/wt/round19/multi_env.py").read_bytes()).hexdigest().upper(), lock["source_tree"]["src/wt/round19/multi_env.py"])

    def test_08_safety_unchanged(self):
        self.assertEqual(self.shared_scenario.config().multi_uav_min_separation_m, 15.0)

    def test_09_behavior_deterministic_by_seed(self):
        first = run_episode(self.shared_scenario, BEHAVIORS[4])
        second = run_episode(self.shared_scenario, BEHAVIORS[4])
        self.assertEqual(plain(first["transitions"]), plain(second["transitions"]))

    def test_10_noisy_behavior_legal(self):
        noisy = run_episode(self.shared_scenario, BEHAVIORS[4])
        self.assertTrue(all(np.all(np.linalg.norm(np.asarray(item["joint_action_applied_t"]), axis=1) <= 1.0 + 1e-9) and item["min_uav_distance_m"] >= 15.0 - 1e-8 and isinstance(item["safety_projection_fallback"], bool) for item in noisy["transitions"]))

    def test_11_noisy_h1_preserves_h1_source(self):
        self.assertEqual(BEHAVIORS[4].base_horizon, 1)

    def test_12_policy_switch_has_no_future_input(self):
        source = (ROOT / "src/wt/round20/dataset.py").read_text(encoding="utf-8").split("def _switch_base", 1)[1].split("def behavior_action", 1)[0]
        self.assertNotIn("future", source)

    def test_13_random_joint_action_legal(self):
        random_episode = run_episode(self.shared_scenario, BEHAVIORS[-1])
        self.assertTrue(all(np.all(np.linalg.norm(np.asarray(item["joint_action_t"]), axis=1) <= 1.0 + 1e-9) for item in random_episode["transitions"]))

    def test_14_scenario_level_split(self):
        self.assertEqual({item.split for item in self.scenarios}, {"train", "validation", "development_test"})

    def test_15_no_scenario_across_splits(self):
        self.assertTrue(split_audit(self.scenarios)["scenario_split_isolation"])

    def test_16_shared_queue_conservation(self):
        self.assertTrue(all(np.max(np.abs(np.asarray(item["queue_conservation_residual_bits"]))) <= 1e-6 for item in self.episode["transitions"]))

    def test_17_no_duplicate_payload(self):
        self.assertTrue(all(np.all(np.asarray(item["payload_by_buoy_bits"]) <= np.asarray(item["queue_before_bits"]) + 1e-6) for item in self.episode["transitions"]))

    def test_18_uav_node_serialization(self):
        self.assertEqual(np.asarray(self.graph["uav_nodes"]).shape, (2, 8))

    def test_19_buoy_node_serialization(self):
        self.assertEqual(np.asarray(self.graph["buoy_nodes"]).shape, (5, 7))

    def test_20_ub_edge_serialization(self):
        self.assertEqual(np.asarray(self.graph["ub_edges"]).shape, (2, 5, 8))

    def test_21_uu_edge_serialization(self):
        self.assertEqual(np.asarray(self.graph["uu_edges"]).shape, (2, 2, 9))

    def test_22_masks_correct(self):
        self.assertEqual((int(np.sum(self.graph["uav_mask"])), int(np.sum(self.graph["buoy_mask"]))), (2, 3))

    def test_23_variable_k(self):
        k5 = next(item for item in self.scenarios if item.num_buoys == 5 and item.spectrum_tier == "low")
        env = SharedSpectrumMultiUAVEnv(k5.config(), buoy_positions_m=np.asarray(k5.buoy_positions_m), options=k5.options())
        _, info = env.reset(seed=k5.seed)
        from wt.round20.dataset import build_graph_state
        self.assertEqual(int(np.sum(build_graph_state(info, env.config)["buoy_mask"])), 5)

    def test_24_uav_permutation_equivariance(self):
        swapped = permute_uavs(self.graph, (1, 0))
        self.assertTrue(
            np.array_equal(np.asarray(swapped["uav_nodes"])[0], np.asarray(self.graph["uav_nodes"])[1])
            and np.array_equal(np.asarray(permute_uavs(swapped, (1, 0))["ub_edges"]), np.asarray(self.graph["ub_edges"]))
        )

    def test_25_buoy_permutation_equivariance(self):
        swapped = permute_buoys(self.graph, (2, 1, 0))
        self.assertTrue(
            np.array_equal(np.asarray(swapped["buoy_nodes"])[0], np.asarray(self.graph["buoy_nodes"])[2])
            and np.array_equal(np.asarray(permute_buoys(swapped, (2, 1, 0))["ub_edges"]), np.asarray(self.graph["ub_edges"]))
        )

    def test_26_h5_service_labels_correct(self):
        sample = next(item for item in self.samples if item["H_task"] == 5 and item["t"] == 0)
        expected = np.sum(np.asarray([item["service_by_uav_buoy_bits"] for item in self.episode["transitions"][:5]]), axis=0)
        self.assertTrue(np.array_equal(np.asarray(sample["per_uav_gross_service_progress_bits"]), expected))

    def test_27_h10_service_labels_correct(self):
        sample = next(item for item in self.samples if item["H_task"] == 10 and item["t"] == 0)
        expected = np.sum(np.asarray([item["service_by_uav_buoy_bits"] for item in self.episode["transitions"][:10]]), axis=0)
        self.assertTrue(np.array_equal(np.asarray(sample["per_uav_gross_service_progress_bits"]), expected))

    def test_28_h5_progress_labels_correct(self):
        sample = next(item for item in self.samples if item["H_task"] == 5 and item["t"] == 0)
        self.assertTrue(np.allclose(np.asarray(sample["per_uav_gross_service_progress_normalized"]), np.asarray(sample["per_uav_gross_service_progress_bits"]) / self.shared_scenario.config().queue_capacity_bits))

    def test_29_h10_progress_labels_correct(self):
        sample = next(item for item in self.samples if item["H_task"] == 10 and item["t"] == 0)
        self.assertTrue(np.allclose(np.asarray(sample["per_uav_gross_service_progress_normalized"]), np.asarray(sample["per_uav_gross_service_progress_bits"]) / self.shared_scenario.config().queue_capacity_bits))

    def test_30_shared_queue_service_conservation(self):
        audit = task_waypoint_audits([self.episode], self.samples)
        self.assertTrue(audit["all_conservation_passed"])

    def test_31_team_return_formula(self):
        sample = next(item for item in self.samples if item["H_task"] == 5 and item["t"] == 0)
        expected = sum((.99 ** index) * item["reward_t"] for index, item in enumerate(self.episode["transitions"][:5]))
        self.assertAlmostEqual(sample["team_h_step_return_raw"], expected)

    def test_32_targets_do_not_cross_episode(self):
        self.assertTrue(all(item["t"] + item["H_task"] <= self.episode["steps"] for item in self.samples))

    def test_33_no_future_fields_in_graph_input(self):
        self.assertTrue(all("future" not in key for item in self.samples for key in item["current_graph_state"]))

    def test_34_quality_metadata_not_graph_input(self):
        self.assertNotIn("quality_tier", self.graph)

    def test_35_behavior_metadata_not_graph_input(self):
        self.assertNotIn("behavior_generator", self.graph)

    def test_36_no_nan_inf(self):
        self.assertTrue(np.all(np.isfinite(np.asarray(self.graph["ub_edges"]))) and np.all(np.isfinite(np.asarray(self.graph["uu_edges"]))))

    def test_37_save_reload_checksum(self):
        encoded = json.dumps(plain(self.episode["transitions"][0]), sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), hashlib.sha256(json.dumps(json.loads(encoded.decode("utf-8")), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest())

    def test_38_historical_protected_access_zero(self):
        manifest = json.loads((ROOT / "round_20/round_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["old_development_test_access_count"], manifest["fresh_development_access_count"], manifest["untouched_holdout_new_access_count"]), (0, 0, 0))

    def test_39_new_untouched_policy_rollouts_zero(self):
        self.assertEqual(sum(item["policy_rollout_count"] for item in untouched_evaluation_scenarios()), 0)

    def test_40_hgnn_false(self):
        self.assertFalse(json.loads((ROOT / "round_20/round_manifest.json").read_text(encoding="utf-8"))["hgnn_training_performed"])

    def test_41_ppo_marl_false(self):
        manifest = json.loads((ROOT / "round_20/round_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["ppo_training_performed"] or manifest["multi_agent_rl_performed"])

    def test_42_round19_preflight(self):
        self.assertTrue(preflight(ROOT)["round19_verified"])

    def test_43_schema_future_variable_m(self):
        from wt.round20.dataset import graph_schema
        self.assertTrue(graph_schema()["future_variable_m_compatible"])

    def test_44_compileall(self):
        self.assertTrue(compileall.compile_dir(str(ROOT / "src/wt/round20"), quiet=1))


if __name__ == "__main__":
    unittest.main()
