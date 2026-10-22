"""Phase-5ZM gate and rank-mean invariance checks."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.phase5zm_multiseed_model_policy.consensus import rank_mean_select
from wt.lowlevel_v2.phase5zm_multiseed_model_policy.protocol import (
    FRESH_CONFIRMATION_AUTHORIZED,
    MODEL_R1_AGGREGATION_APPROVED,
    PLANNER_READY,
    assert_consumed,
    choose_gate,
)


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "wt" / "lowlevel_v2" / "phase5zm_multiseed_model_policy"


class Phase5ZMTests(unittest.TestCase):
    def test_only_human_spec_keeps_ms_a_with_ms_d(self) -> None:
        chosen = choose_gate(
            permutation_ok=True,
            j15_retained=True,
            illegal=False,
            isolation_ok=True,
            large_conflict=False,
            live_matches_saved=True,
            human_spec_approved=False,
        )
        self.assertEqual(chosen["gate"], "MS-A")
        self.assertEqual(chosen["blockers"], ["MS-D"])
        self.assertTrue(chosen["aggregation_technically_verified"])
        self.assertFalse(chosen["fresh_confirmation_authorized"])
        self.assertFalse(chosen["planner_ready"])

    def test_degradation_outranks_instability(self) -> None:
        chosen = choose_gate(
            permutation_ok=False,
            j15_retained=False,
            illegal=True,
            isolation_ok=False,
            large_conflict=True,
            live_matches_saved=False,
            human_spec_approved=False,
        )
        self.assertEqual(chosen["gate"], "MS-C")
        self.assertEqual(chosen["blockers"], ["MS-C", "MS-B", "MS-D"])

    def test_invariance_failure_is_ms_b(self) -> None:
        chosen = choose_gate(
            permutation_ok=False,
            j15_retained=True,
            illegal=False,
            isolation_ok=True,
            large_conflict=False,
            live_matches_saved=True,
            human_spec_approved=False,
        )
        self.assertEqual(chosen["gate"], "MS-B")
        self.assertIn("MS-D", chosen["blockers"])

    def test_flags_and_forbidden_window(self) -> None:
        self.assertFalse(PLANNER_READY)
        self.assertFalse(FRESH_CONFIRMATION_AUTHORIZED)
        self.assertFalse(MODEL_R1_AGGREGATION_APPROVED)
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)

    def test_package_does_not_call_deployment_decision(self) -> None:
        for path in PACKAGE.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))

    def test_selection_is_seed_and_candidate_invariant(self) -> None:
        values = np.asarray(
            [
                [1.0, 9.0, 3.0, 2.0],
                [8.0, 1.0, 3.0, 2.0],
                [7.0, 1.0, 4.0, 2.0],
                [6.0, 2.0, 3.0, 1.0],
                [5.0, 1.0, 4.0, 2.0],
            ],
            dtype=np.float64,
        )
        assignments = np.asarray([[0, 1], [1, 0], [2, -1], [-1, 2]], dtype=np.int64)
        base = rank_mean_select(values, assignments)
        self.assertNotEqual(int(base["index"]), int(np.argmax(values[0])))
        for order in (np.array([4, 3, 2, 1, 0]), np.array([1, 3, 0, 4, 2])):
            moved = rank_mean_select(values[order], assignments)
            self.assertEqual(moved["key"], base["key"])
            self.assertEqual(moved["strength"], base["strength"])
            self.assertEqual(int(moved["rank_sum"][moved["index"]]), int(base["rank_sum"][base["index"]]))
        for order in (np.array([3, 1, 2, 0]), np.array([1, 0, 3, 2])):
            moved = rank_mean_select(values[:, order], assignments[order])
            self.assertEqual(moved["key"], base["key"])
            restored = np.empty(values.shape[1], dtype=np.int64)
            restored[order] = moved["rank_sum"]
            self.assertTrue(np.array_equal(restored, base["rank_sum"]))

    def test_exact_tie_uses_assignment_order_not_input_order(self) -> None:
        values = np.ones((5, 2), dtype=np.float64)
        first = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
        second = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        self.assertEqual(rank_mean_select(values, first)["key"], (0, 1))
        self.assertEqual(rank_mean_select(values, second)["key"], (0, 1))
        self.assertEqual(rank_mean_select(values, second)["stage"], "canonical_order")


if __name__ == "__main__":
    unittest.main()
