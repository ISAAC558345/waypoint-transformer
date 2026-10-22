"""Focused non-training contracts for the Round-15 planning audit."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from wt.round15 import ACTION_CANDIDATE_NAMES, finite_lattice_plan, myopic_one_step_greedy
from wt.round15.audit import _angle_degrees, _bootstrap, _policy_descriptor, _queue_tie_proxy, dispatch_policy_action, preflight
from wt.round15.planner import _causal_score, action_candidates
from wt.round15.scenarios import materialize_scenario, planning_scenarios, scenario_manifest_rows


ROOT = Path(__file__).resolve().parents[1]


class Round15PlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = planning_scenarios()[0]
        self.env, self.observation, self.info = materialize_scenario(self.scenario)

    def test_001_twelve_new_scenarios_are_registered(self) -> None:
        self.assertEqual(len(planning_scenarios()), 12)

    def test_002_scenarios_cover_three_and_five_buoys(self) -> None:
        self.assertEqual({item.num_buoys for item in planning_scenarios()}, {3, 5})

    def test_003_scenarios_cover_difficulty(self) -> None:
        self.assertEqual({item.difficulty for item in planning_scenarios()}, {"easy", "medium", "hard"})

    def test_004_scenarios_cover_pressure(self) -> None:
        self.assertEqual({item.queue_pressure for item in planning_scenarios()}, {"low", "medium", "high"})

    def test_005_scenarios_cover_geometry(self) -> None:
        self.assertEqual({item.geometry for item in planning_scenarios()}, {"compact", "spread"})

    def test_006_scenario_reset_is_deterministic(self) -> None:
        _, _, first = materialize_scenario(self.scenario)
        _, _, second = materialize_scenario(self.scenario)
        np.testing.assert_array_equal(first["buoy_positions_m"], second["buoy_positions_m"])

    def test_007_scenario_manifest_has_no_historical_data(self) -> None:
        self.assertTrue(all(not row["uses_historical_dataset"] for row in scenario_manifest_rows()))

    def test_008_candidate_set_is_frozen(self) -> None:
        self.assertEqual(len(ACTION_CANDIDATE_NAMES), 13)

    def test_009_candidate_set_is_legal(self) -> None:
        self.assertTrue(all(self.env.action_space.contains(action) for action in action_candidates(self.env, self.observation, self.info).values()))

    def test_010_myopic_action_is_legal(self) -> None:
        self.assertTrue(self.env.action_space.contains(myopic_one_step_greedy(self.env, self.observation, self.info)))

    def test_011_causal_h1_is_marked_deployable_diagnostic(self) -> None:
        decision = finite_lattice_plan(self.env, self.observation, self.info, horizon=1, mode="causal")
        self.assertTrue(decision.deployable_diagnostic)

    def test_012_causal_h1_is_not_oracle(self) -> None:
        decision = finite_lattice_plan(self.env, self.observation, self.info, horizon=1, mode="causal")
        self.assertFalse(decision.oracle_exogenous_knowledge)

    def test_013_oracle_is_explicitly_non_deployable(self) -> None:
        decision = finite_lattice_plan(self.env, self.observation, self.info, horizon=1, mode="oracle")
        self.assertTrue(decision.oracle_exogenous_knowledge)
        self.assertFalse(decision.deployable_diagnostic)

    def test_014_planner_is_deterministic_from_same_snapshot(self) -> None:
        one = finite_lattice_plan(self.env, self.observation, self.info, horizon=5, mode="causal")
        two = finite_lattice_plan(self.env, self.observation, self.info, horizon=5, mode="causal")
        np.testing.assert_array_equal(one.action, two.action)
        self.assertEqual(one.candidate_scores, two.candidate_scores)

    def test_015_planner_rejects_unknown_horizon(self) -> None:
        with self.assertRaises(ValueError):
            finite_lattice_plan(self.env, self.observation, self.info, horizon=2, mode="causal")

    def test_016_planner_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            finite_lattice_plan(self.env, self.observation, self.info, horizon=1, mode="future")

    def test_017_causal_score_does_not_deepcopy_environment(self) -> None:
        self.assertNotIn("deepcopy", inspect.getsource(_causal_score))

    def test_018_dispatch_returns_baseline_without_decision(self) -> None:
        action, decision = dispatch_policy_action(self.env, self.observation, self.info, "nearest_buoy")
        self.assertTrue(self.env.action_space.contains(action))
        self.assertIsNone(decision)

    def test_019_policy_descriptor_is_unambiguous(self) -> None:
        self.assertEqual(_policy_descriptor("causal_H20"), ("causal", 20))

    def test_020_queue_tie_proxy_definition(self) -> None:
        self.assertTrue(_queue_tie_proxy(np.array([100.0, 95.0, 10.0])))
        self.assertFalse(_queue_tie_proxy(np.array([100.0, 80.0, 10.0])))

    def test_021_action_angle_handles_zero(self) -> None:
        self.assertEqual(_angle_degrees(np.zeros(2), np.zeros(2)), 0.0)

    def test_022_bootstrap_is_deterministic(self) -> None:
        self.assertEqual(_bootstrap([1.0, 2.0, 3.0], replicates=40), _bootstrap([1.0, 2.0, 3.0], replicates=40))

    def test_023_protocol_disables_training(self) -> None:
        protocol = json.loads((ROOT / "round_15/configs/planning_protocol.json").read_text(encoding="utf-8"))
        self.assertFalse(protocol["training_performed"])

    def test_024_manifest_discloses_one_metadata_only_access(self) -> None:
        manifest = json.loads((ROOT / "round_15/round_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["untouched_holdout_accessed"])
        self.assertEqual(manifest["untouched_holdout_access_count"], 1)

    def test_025_round15_source_has_no_neural_training_import(self) -> None:
        source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src/wt/round15").glob("*.py"))
        self.assertNotIn("torch", source)
        self.assertNotIn(".backward(", source)

    def test_026_preflight_preserves_round14_lock(self) -> None:
        lock = json.loads((ROOT / "round_15/ROUND_LOCK.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["status"], "locked")
        self.assertTrue(lock["acceptance_complete"])

    def test_027_round15_remains_locked_after_a_later_round_starts(self) -> None:
        self.assertEqual(json.loads((ROOT / "round_15/ROUND_LOCK.json").read_text(encoding="utf-8"))["status"], "locked")


if __name__ == "__main__":
    unittest.main()
