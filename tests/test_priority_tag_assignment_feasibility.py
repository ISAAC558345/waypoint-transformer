"""Boundary checks for the registered assignment cap. They do not run the eight scenarios."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.attribution import attribute_served
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.candidates import select_registered
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.criteria import opportunity_evidence, service_levels
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.identity import contract_problems, history_problems, interface_identity
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.protocol import (
    OPPORTUNITY_ABSENT,
    OPPORTUNITY_LOCAL,
    OPPORTUNITY_REPLICATED,
    OPPORTUNITY_THIN,
)


class AssignmentFeasibilityTests(unittest.TestCase):
    def test_cap_keeps_baseline_and_does_not_pad(self) -> None:
        pool = [[0], [1], [0], [2], [3], [4], [5]]
        selected = select_registered(pool, 6, 5)
        self.assertEqual(selected["selected"], [(0,), (1,), (2,), (3,), (5,)])
        self.assertTrue(selected["truncated"])
        self.assertEqual(selected["duplicate_count"], 1)
        short = select_registered([[0], [1]], 1, 5)
        self.assertEqual(short["selected"], [(0,), (1,)])
        self.assertFalse(short["truncated"])

    def test_service_levels_follow_the_registered_order(self) -> None:
        self.assertEqual(service_levels(1.1, -1.0, 0, 0, 0.0, 0.0)["s2"], True)
        self.assertEqual(service_levels(1.0, -1.0, 0, 0, 0.0, 0.0)["s0"], False)
        self.assertEqual(service_levels(2.0, -1.1, 0, 0, 0.0, 0.0)["s0"], False)
        self.assertEqual(service_levels(2.0, 0.0, 1, 0, 0.0, 0.0)["s1"], False)
        self.assertEqual(service_levels(2.0, 0.0, 0, 0, -0.1, 0.0)["s2"], False)
        self.assertEqual(service_levels(2.0, 0.0, 0, 0, 0.0, 0.1)["s2"], False)

    def test_opportunity_labels_do_not_upgrade_partial_coverage(self) -> None:
        replicated = [
            {"scenario_id": 1, "decision_time": 0, "interpretable": True, "s0": True, "s2": True},
            {"scenario_id": 2, "decision_time": 0, "interpretable": True, "s0": True, "s2": True},
        ]
        self.assertEqual(opportunity_evidence(replicated, expected_scenarios=2, expected_snapshots=2), OPPORTUNITY_REPLICATED)
        local = [{"scenario_id": 1, "decision_time": 0, "interpretable": True, "s0": True, "s2": False}]
        self.assertEqual(opportunity_evidence(local, expected_scenarios=2, expected_snapshots=2), OPPORTUNITY_LOCAL)
        absent = [
            {"scenario_id": 1, "decision_time": 0, "interpretable": True, "s0": False, "s2": False},
            {"scenario_id": 2, "decision_time": 15, "interpretable": True, "s0": False, "s2": False},
        ]
        self.assertEqual(opportunity_evidence(absent, expected_scenarios=2, expected_snapshots=2), OPPORTUNITY_ABSENT)
        self.assertEqual(opportunity_evidence([], expected_scenarios=2, expected_snapshots=2), OPPORTUNITY_THIN)

    def test_known_service_excludes_later_buckets_and_does_not_need_snap(self) -> None:
        buckets = [[
            {"admission_step": 0, "priority_tag": 0, "bits_remaining": 100.0, "insert_index": 1},
            {"admission_step": 4, "priority_tag": 1, "bits_remaining": 50.0, "insert_index": 2},
        ]]
        taken = attribute_served(buckets, [120.0], 6, {1}, lambda _bucket, _time: 0.5)
        self.assertEqual(taken["known_ordinary"], 100.0)
        self.assertEqual(taken["future_tagged"], 20.0)
        self.assertEqual(taken["known_tagged"], 0.0)
        self.assertEqual(buckets[0][0]["bits_remaining"], 100.0)

    def test_identity_stays_on_the_experiment_pool(self) -> None:
        self.assertEqual(contract_problems(), [])
        self.assertEqual(history_problems(), [])
        identity = interface_identity()
        self.assertTrue(identity["not_a_formal_scheme_c_evaluation"])
        self.assertFalse(identity["formal_scheme_c"]["called_by_this_diagnosis"])
        self.assertEqual(identity["candidate_generation"]["k"], 5)
        self.assertEqual(identity["formal_scheme_c"]["k"], 6)

    def test_new_sources_omit_forbidden_tokens(self) -> None:
        token = "deci" + "de("
        held = ("322" + "048", "322" + "064")
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_assignment_feasibility"
        session = (root / "session.py").read_text(encoding="utf-8")
        self.assertIn("self.assignment_rng.integers(0, len(pool))", session)
        for path in list(root.glob("*.py")) + [Path(__file__)]:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(token, text)
            for item in held:
                self.assertNotIn(item, text)
            self.assertNotIn("optimizer." + "step", text)
            self.assertNotIn("back" + "ward(", text)


if __name__ == "__main__":
    unittest.main()
