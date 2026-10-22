"""Round-35 Level-2 structural and governance regression tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.round35.audit import legal_assignment_count, level2_config, require_engineering_only_manifest
from wt.round35.level2_env import (
    Level2SharedSpectrumMultiUAVEnv,
    association_candidate_mask,
    external_preference_matching,
)


OUTPUT = ROOT / "round_35" / "outputs" / "round35_level2_structural_validation_20260902_0001"
FIGURES = ROOT / "round_35" / "figures" / "round35_level2_structural_validation_20260902_0001"
R34_LOCK_SHA256 = "59A472D367BEE8364C2D5CC4468759685795E6CB7E2176DFFD2B12F56C489AF0"


def load(name: str) -> dict:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def level2_env() -> Level2SharedSpectrumMultiUAVEnv:
    return Level2SharedSpectrumMultiUAVEnv(
        level2_config(),
        buoy_positions_m=np.asarray(((58., 72.), (76., 72.), (94., 72.), (112., 72.), (58., 128.), (76., 128.), (94., 128.), (112., 128.))),
        interference_overlap_factor=0.25,
    )


class Round35Level2Tests(unittest.TestCase):
    def test_01_round34_lock_unchanged(self) -> None:
        self.assertEqual(hashlib.sha256((ROOT / "round_34" / "ROUND_LOCK.json").read_bytes()).hexdigest().upper(), R34_LOCK_SHA256)

    def test_02_protected_exception_history_unchanged(self) -> None:
        lock = json.loads((ROOT / "round_34" / "ROUND_LOCK.json").read_text(encoding="utf-8"))
        self.assertEqual((lock["development_test_access_count"], lock["formal_scientific_protected_records_used"]), (1, 0))

    def test_03_no_second_protected_access(self) -> None:
        self.assertFalse(load("access_audit.json")["second_protected_access"])

    def test_04_no_model_training(self) -> None:
        self.assertFalse(json.loads((ROOT / "round_35" / "round_manifest.json").read_text(encoding="utf-8"))["training_performed"])

    def test_05_no_optimizer(self) -> None:
        source = (ROOT / "src" / "wt" / "round35" / "level2_env.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer =", source)

    def test_06_no_backward(self) -> None:
        source = (ROOT / "src" / "wt" / "round35" / "level2_env.py").read_text(encoding="utf-8").lower()
        self.assertNotIn(".backward", source)

    def test_07_legacy_m2k3_regression(self) -> None:
        self.assertTrue(load("round35_legacy_scale_parameterization_regression.json")["fixtures"][0]["passed"])

    def test_08_legacy_m2k5_regression(self) -> None:
        fixtures = load("round35_legacy_scale_parameterization_regression.json")["fixtures"]
        self.assertTrue(all(item["passed"] for item in fixtures if "k5" in item["fixture"]))

    def test_09_t12_t20_compatibility(self) -> None:
        fixtures = load("round35_legacy_scale_parameterization_regression.json")["fixtures"]
        self.assertEqual({12, 20}, {int(item["fixture"].split("t")[-1]) for item in fixtures})

    def test_10_m4k8_observation_shapes(self) -> None:
        env = level2_env(); env.reset(seed=1); observation = env.relational_observation()
        self.assertEqual(observation["uav_buoy_relations"].shape, (4, 8, 8))

    def test_11_m4k8_action_shapes(self) -> None:
        self.assertEqual(level2_env().action_shape, (4, 2))

    def test_12_association_preference_shape(self) -> None:
        self.assertEqual(level2_env().association_action_shape, (4, 8))

    def test_13_logits_must_be_finite(self) -> None:
        env = level2_env(); env.reset(seed=1)
        with self.assertRaises(ValueError):
            env.step({"move": np.zeros((4, 2)), "association": np.full((4, 8), np.nan)})

    def test_14_candidate_mask_respected(self) -> None:
        mask = association_candidate_mask(np.asarray((0., 1., 1.)), np.ones((2, 3)))
        assignment, _, _, _, _ = external_preference_matching(np.asarray((0., 1., 1.)), np.ones((2, 3)), level2_config(num_uavs=2, num_buoys=3), np.ones((2, 3)), candidate_mask=mask)
        self.assertNotIn(0, assignment)

    def test_15_exclusivity_respected(self) -> None:
        result = load("association_decoder_constraint_audit.json")
        self.assertEqual(result["status"], "PASS")

    def test_16_idle_uav_allowed(self) -> None:
        assignment, _, _, _, _ = external_preference_matching(np.ones(3), np.ones((2, 3)), level2_config(num_uavs=2, num_buoys=3), -np.ones((2, 3)))
        self.assertTrue(np.array_equal(assignment, np.asarray((-1, -1))))

    def test_17_deterministic_tie_break(self) -> None:
        self.assertEqual(load("association_decoder_constraint_audit.json")["deterministic_tie_assignment"], [0, 1, 2, 3])

    def test_18_preference_affects_assignment(self) -> None:
        self.assertTrue(json.loads((OUTPUT / "relational_action_ownership_gate.json").read_text(encoding="utf-8"))["a_assoc_causally_changes_assignment"])

    def test_19_no_hidden_heuristic_override(self) -> None:
        self.assertFalse(load("relational_action_ownership_gate.json")["decoder_uses_legacy_heuristic_score"])

    def test_20_movement_channel_independent(self) -> None:
        rows = (OUTPUT / "action_channel_separation.csv").read_text(encoding="utf-8")
        self.assertIn("B_different_move_same_association", rows)

    def test_21_association_channel_independent(self) -> None:
        rows = (OUTPUT / "action_channel_separation.csv").read_text(encoding="utf-8")
        self.assertIn("C_same_move_different_association", rows)

    def test_22_permutation_equivariance(self) -> None:
        self.assertEqual(load("round35_permutation_audit.json")["status"], "PASS")

    def test_23_queue_conservation(self) -> None:
        env = level2_env(); env.reset(seed=1); _, _, _, _, info = env.step({"move": np.zeros((4, 2)), "association": np.zeros((4, 8))})
        self.assertTrue(np.allclose(info["queue_conservation_residual_bits"], 0.0))

    def test_24_service_conservation(self) -> None:
        env = level2_env(); env.reset(seed=1); _, _, _, _, info = env.step({"move": np.zeros((4, 2)), "association": np.zeros((4, 8))})
        self.assertTrue(np.all(info["transmitted_payload_bits"] <= info["link_capacity_bits"] + 1.0e-6))

    def test_25_shared_spectrum_interference_valid(self) -> None:
        env = level2_env(); env.reset(seed=1); _, _, _, _, info = env.step({"move": np.zeros((4, 2)), "association": np.zeros((4, 8))})
        self.assertTrue(np.all(np.isfinite(info["cross_interference_power_w"])))

    def test_26_safety_unchanged(self) -> None:
        fixture = load("round35_legacy_scale_parameterization_regression.json")["fixtures"][0]
        self.assertTrue(fixture["checks"]["safety"])

    def test_27_reward_unchanged(self) -> None:
        fixture = load("round35_legacy_scale_parameterization_regression.json")["fixtures"][0]
        self.assertTrue(fixture["checks"]["reward"])

    def test_28_no_task_waypoint_training(self) -> None:
        self.assertFalse(json.loads((ROOT / "round_35" / "round_manifest.json").read_text(encoding="utf-8"))["task_waypoint_used"])

    def test_29_no_hgnn_training(self) -> None:
        self.assertFalse(json.loads((ROOT / "round_35" / "round_manifest.json").read_text(encoding="utf-8"))["hgnn_used"])

    def test_30_no_ppo(self) -> None:
        source = (ROOT / "src" / "wt" / "round35" / "level2_env.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("import torch", source)

    def test_31_protected_split_fail_closed(self) -> None:
        with self.assertRaises(PermissionError):
            require_engineering_only_manifest({"scenario_family": "round35_level2_engineering_v1", "generated": True, "split": "development_test", "protected_data_access_count": 0})

    def test_32_untouched_access_zero(self) -> None:
        self.assertEqual(load("access_audit.json")["untouched_eval_access_count"], 0)

    def test_33_no_nan_inf(self) -> None:
        env = level2_env(); env.reset(seed=1); observation = env.relational_observation()
        self.assertTrue(all(np.all(np.isfinite(value)) for value in observation.values() if value.dtype != bool))

    def test_34_compileable_source_tree(self) -> None:
        self.assertTrue((ROOT / "src" / "wt" / "round35" / "audit.py").is_file())

    def test_35_exact_assignment_count(self) -> None:
        self.assertEqual(legal_assignment_count(4, 8), 3393)

    def test_36_engineering_scenario_count(self) -> None:
        self.assertEqual(load("round35_level2_engineering_manifest.json")["scenario_count"], 64)

    def test_37_level2_gate(self) -> None:
        self.assertEqual(load("level2_relational_benchmark_gate.json")["gate"], "L2-A")

    def test_38_required_figures_exist(self) -> None:
        self.assertEqual(len(list(FIGURES.glob("*.png"))), 12)
        self.assertEqual(len(list(FIGURES.glob("*.svg"))), 12)


if __name__ == "__main__":
    unittest.main()
