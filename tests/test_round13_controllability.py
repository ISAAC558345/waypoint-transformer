"""Audit-only contracts for Round-13 Task-Waypoint controllability work."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from wt.round13.audit import (
    _action_set,
    _check_round12_lock,
    _new_policy,
    _rollout_from_snapshot,
    _snapshot,
    _state_hash,
    _validation_scenarios,
)
from wt.round13.contracts import (
    current_state_service_heuristic,
    discrete_service_fractions,
    valid_masked_priority_fractions,
)


ROOT = Path(__file__).resolve().parents[1]


class Round13ControllabilityContracts(unittest.TestCase):
    def test_01_masked_simplex_keeps_padding_zero(self) -> None:
        result = valid_masked_priority_fractions(np.array([2.0, 6.0, 7.0]), np.array([True, True, False]))
        self.assertTrue(np.allclose(result, [0.25, 0.75, 0.0]))
        self.assertAlmostEqual(float(result.sum()), 1.0)

    def test_02_zero_priority_has_valid_uniform_fallback(self) -> None:
        result = valid_masked_priority_fractions(np.zeros(3), np.array([True, False, True]))
        self.assertTrue(np.allclose(result, [0.5, 0.0, 0.5]))

    def test_03_discrete_decoder_requires_a_valid_index(self) -> None:
        self.assertTrue(np.array_equal(discrete_service_fractions(1, np.array([True, True, False])), [0.0, 1.0, 0.0]))
        with self.assertRaises(ValueError):
            discrete_service_fractions(2, np.array([True, True, False]))

    def test_04_current_state_heuristic_is_observable_and_queue_aware(self) -> None:
        self.assertEqual(current_state_service_heuristic(np.array([3.0, 8.0]), np.array([4.0, 1.0]), np.array([True, True])), 0)
        self.assertIsNone(current_state_service_heuristic(np.zeros(2), np.ones(2), np.ones(2, dtype=bool)))

    def test_05_round12_lock_predictor_and_policy_files_are_unchanged(self) -> None:
        audit = _check_round12_lock(ROOT)
        self.assertEqual(audit["round12_lock"], "PASS")
        self.assertEqual(audit["policy_checkpoint_count"], 24)
        self.assertFalse(audit["untouched_holdout_accessed"])

    def test_06_protocol_prohibits_training_and_holdout_access(self) -> None:
        protocol = json.loads((ROOT / "round_13/configs/round13_controllability_protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["formal_scenario_split"], "validation")
        self.assertFalse(protocol["training_performed"])
        self.assertFalse(protocol["untouched_holdout_accessed"])
        self.assertEqual(protocol["continuation_policy"], "queue_rate_energy")

    def test_07_current_environment_action_is_two_dimensional_movement_only(self) -> None:
        scenario = _validation_scenarios(ROOT)[0]
        env, observation, info = _snapshot(scenario, 0, 10)
        self.assertEqual(tuple(env.action_space.shape), (2,))
        self.assertTrue(np.all(env.action_space.low == -1.0))
        self.assertTrue(np.all(env.action_space.high == 1.0))
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertIn("transmitted_payload_bits", info)

    def test_08_all_preregistered_perturbations_are_legal(self) -> None:
        scenario = _validation_scenarios(ROOT)[0]
        env, observation, info = _snapshot(scenario, 0, 10)
        actions = _action_set(observation, info)
        self.assertEqual(len(actions), 14)
        self.assertTrue(all(env.action_space.contains(action) for action in actions.values()))

    def test_09_snapshot_is_restored_and_continuation_is_deterministic(self) -> None:
        scenario = _validation_scenarios(ROOT)[0]
        env, observation, info = _snapshot(scenario, 0, 10)
        before = _state_hash(observation, info)
        action = _action_set(observation, info)["+x"]
        left = _rollout_from_snapshot(env, observation, info, action, 10)
        right = _rollout_from_snapshot(env, observation, info, action, 10)
        self.assertEqual(before, _state_hash(observation, info))
        self.assertEqual(left["H_step_return"], right["H_step_return"])
        self.assertTrue(np.array_equal(left["gross_service_progress_bits"], right["gross_service_progress_bits"]))
        self.assertTrue(left["all_actions_legal"])

    def test_10_service_progress_is_actual_payload_and_finite(self) -> None:
        scenario = _validation_scenarios(ROOT)[0]
        env, observation, info = _snapshot(scenario, 0, 10)
        action = _action_set(observation, info)["queue_rate_energy"]
        rollout = _rollout_from_snapshot(env, observation, info, action, 10)
        self.assertGreaterEqual(rollout["total_service_bits"], 0.0)
        self.assertAlmostEqual(rollout["total_service_bits"], float(np.sum(rollout["gross_service_progress_bits"])))
        self.assertTrue(np.isfinite(rollout["H_step_return"]))

    def test_11_round13_module_has_no_learning_or_other_systems(self) -> None:
        source = (ROOT / "src/wt/round13/audit.py").read_text(encoding="utf-8").lower()
        for forbidden in ("import torch", "torch.", ".backward(", "torch.optim", "from hgnn", "from ppo", "multi-uav"):
            self.assertNotIn(forbidden, source)
        self.assertNotIn(".train(", source)

    def test_12_queue_and_capacity_bound_the_prospective_payload(self) -> None:
        queue = np.array([5.0, 2.0]); capacity = np.array([3.0, 10.0])
        payload = np.minimum(queue, capacity)
        self.assertTrue(np.all(payload <= queue))
        self.assertTrue(np.all(payload <= capacity))


if __name__ == "__main__":
    unittest.main()
