"""Focused contract tests for the non-learning Round-14 movement mapping."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from wt.round14.guidance import (
    MAPPING_NAMES,
    MappingResult,
    current_observable_mapping_inputs,
    guidance_action,
    map_task_waypoint_to_motion,
    return_magnitude,
    split_task_waypoint,
)


ROOT = Path(__file__).resolve().parents[1]


class Round14GuidanceTests(unittest.TestCase):
    """Small pure-function tests: no dataset, checkpoint, optimizer, or rollout."""

    def setUp(self) -> None:
        self.mask = np.array([True, True, False])
        self.uav = np.array([0.0, 0.0])
        self.buoys = np.array([[2.0, 0.0], [0.0, 4.0], [7.0, 7.0]])
        self.rates = np.array([1.0, 3.0, 999.0])
        self.phi = np.array([0.75, 0.25, 99.0, 2.0, 1.0, 99.0, 0.0])

    def _mapping(self, name: str, **kwargs):
        return map_task_waypoint_to_motion(name, self.phi, self.mask, self.uav, self.buoys, self.rates, **kwargs)

    def test_001_schema_has_three_candidates(self) -> None:
        self.assertEqual(MAPPING_NAMES, ("A_service", "B_service_progress", "C_service_progress_rate"))

    def test_002_phi_split_service(self) -> None:
        service, _, _ = split_task_waypoint(self.phi, 3)
        np.testing.assert_allclose(service, [0.75, 0.25, 99.0])

    def test_003_phi_split_progress(self) -> None:
        _, progress, _ = split_task_waypoint(self.phi, 3)
        np.testing.assert_allclose(progress, [2.0, 1.0, 99.0])

    def test_004_phi_split_return(self) -> None:
        self.assertEqual(split_task_waypoint(self.phi, 3)[2], 0.0)

    def test_005_a_favours_service_direction(self) -> None:
        result = self._mapping("A_service")
        self.assertGreater(result.direction_xy[0], result.direction_xy[1])

    def test_006_b_uses_positive_progress(self) -> None:
        result = self._mapping("B_service_progress")
        self.assertGreater(result.direction_xy[0], result.direction_xy[1])

    def test_007_c_uses_current_rate(self) -> None:
        result = self._mapping("C_service_progress_rate")
        self.assertGreater(result.direction_xy[1], 0.0)

    def test_008_invalid_padding_weight_is_zero_a(self) -> None:
        self.assertEqual(self._mapping("A_service").node_weights[2], 0.0)

    def test_009_invalid_padding_weight_is_zero_b(self) -> None:
        self.assertEqual(self._mapping("B_service_progress").node_weights[2], 0.0)

    def test_010_invalid_padding_weight_is_zero_c(self) -> None:
        self.assertEqual(self._mapping("C_service_progress_rate").node_weights[2], 0.0)

    def test_011_weights_sum_to_one_a(self) -> None:
        self.assertAlmostEqual(float(self._mapping("A_service").node_weights.sum()), 1.0)

    def test_012_weights_sum_to_one_b(self) -> None:
        self.assertAlmostEqual(float(self._mapping("B_service_progress").node_weights.sum()), 1.0)

    def test_013_weights_sum_to_one_c(self) -> None:
        self.assertAlmostEqual(float(self._mapping("C_service_progress_rate").node_weights.sum()), 1.0)

    def test_014_direction_is_unit_when_nonzero(self) -> None:
        self.assertAlmostEqual(float(np.linalg.norm(self._mapping("A_service").direction_xy)), 1.0)

    def test_015_step_fraction_is_retained(self) -> None:
        self.assertEqual(self._mapping("A_service", step_fraction=0.4).step_fraction, 0.4)

    def test_016_action_uses_default_fraction(self) -> None:
        self.assertAlmostEqual(float(np.linalg.norm(guidance_action(self._mapping("A_service", step_fraction=0.4)))), 0.4)

    def test_017_action_uses_explicit_magnitude(self) -> None:
        self.assertAlmostEqual(float(np.linalg.norm(guidance_action(self._mapping("A_service"), magnitude=0.3))), 0.3)

    def test_018_action_is_radially_bounded(self) -> None:
        self.assertLessEqual(float(np.linalg.norm(guidance_action(self._mapping("A_service")))), 1.0)

    def test_019_return_magnitude_midpoint(self) -> None:
        self.assertAlmostEqual(return_magnitude(self.phi, 3), 0.5)

    def test_020_return_magnitude_increases(self) -> None:
        high = self.phi.copy(); high[-1] = 3.0
        self.assertGreater(return_magnitude(high, 3), return_magnitude(self.phi, 3))

    def test_021_return_magnitude_is_bounded(self) -> None:
        for value in (-100.0, 0.0, 100.0):
            phi = self.phi.copy(); phi[-1] = value
            self.assertGreaterEqual(return_magnitude(phi, 3), 0.0)
            self.assertLessEqual(return_magnitude(phi, 3), 1.0)

    def test_022_zero_direction_has_deterministic_fallback(self) -> None:
        buoys = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
        phi = np.array([0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0])
        result = map_task_waypoint_to_motion("A_service", phi, self.mask, self.uav, buoys, self.rates)
        self.assertEqual(result.fallback, "zero_direction")

    def test_023_zero_direction_emits_zero_action(self) -> None:
        result = MappingResult("A_service", np.zeros(2), 1.0, np.array([1.0, 0.0, 0.0]), "zero_direction")
        np.testing.assert_array_equal(guidance_action(result), [0.0, 0.0])

    def test_024_rejects_unknown_mapping(self) -> None:
        with self.assertRaises(ValueError):
            self._mapping("unknown")

    def test_025_rejects_invalid_mask(self) -> None:
        with self.assertRaises(ValueError):
            map_task_waypoint_to_motion("A_service", self.phi, [False, False, False], self.uav, self.buoys, self.rates)

    def test_026_rejects_negative_rate(self) -> None:
        with self.assertRaises(ValueError):
            map_task_waypoint_to_motion("A_service", self.phi, self.mask, self.uav, self.buoys, [-1.0, 1.0, 0.0])

    def test_027_rejects_out_of_range_step_fraction(self) -> None:
        with self.assertRaises(ValueError):
            self._mapping("A_service", step_fraction=1.1)

    def test_028_rejects_wrong_phi_width(self) -> None:
        with self.assertRaises(ValueError):
            split_task_waypoint([1.0, 2.0], 3)

    def test_029_current_info_only_contract(self) -> None:
        info = {"valid_node_mask": self.mask, "uav_position_m": self.uav, "buoy_positions_m": self.buoys, "rates_bps": self.rates, "future_arrivals": [99]}
        result = current_observable_mapping_inputs(info, self.phi)
        self.assertEqual(set(result), {"phi_task", "valid_node_mask", "uav_position_m", "buoy_positions_m", "current_rates_bps"})


class Round14ControlFileTests(unittest.TestCase):
    def test_030_round_manifest_has_required_false_flags(self) -> None:
        manifest = json.loads((ROOT / "round_14/round_manifest.json").read_text(encoding="utf-8"))
        for key in ("training_performed", "neural_policy_training_performed", "service_action_extension_performed", "hgnn_training_performed", "ppo_training_performed", "untouched_holdout_accessed"):
            self.assertFalse(manifest[key])

    def test_031_protocol_forbids_holdout(self) -> None:
        protocol = json.loads((ROOT / "round_14/configs/task_to_motion_mapping_protocol.json").read_text(encoding="utf-8"))
        self.assertIn("untouched_holdout", protocol["forbidden_splits"])

    def test_032_protocol_has_no_training(self) -> None:
        protocol = json.loads((ROOT / "round_14/configs/task_to_motion_mapping_protocol.json").read_text(encoding="utf-8"))
        self.assertFalse(protocol["training_performed"])

    def test_033_round14_remains_locked_when_a_later_round_is_active(self) -> None:
        lock = json.loads((ROOT / "round_14/ROUND_LOCK.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["status"], "locked")

    def test_034_round14_working_rules_prohibit_environment_change(self) -> None:
        rules = (ROOT / "round_14/AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("Do not modify the environment", rules)


if __name__ == "__main__":
    unittest.main()
