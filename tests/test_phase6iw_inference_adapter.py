"""Interface checks that do not open reserved scenarios or the official planner."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, TRAIN_POOL, VALIDATION_POOL
from wt.benchmark_v2.phase6iw_inference_adapter.interface import (
    SchemeCAdapter,
    assignment_is_legal,
    g1_rate_scores,
    stable_choice,
)
from wt.benchmark_v2.phase6iw_inference_adapter.protocol import assign_gate


class Phase6IWInferenceAdapterTests(unittest.TestCase):
    def test_pools_and_gate(self) -> None:
        self.assertTrue(set(TRAIN_POOL).isdisjoint(set(VALIDATION_POOL)))
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))
        self.assertEqual(assign_gate(False), "HIGH_LEVEL_INFERENCE_INTERFACE_READY")
        self.assertEqual(assign_gate(True), "HIGH_LEVEL_INFERENCE_INTERFACE_BLOCKED")

    def test_idle_is_legal_and_a_repeated_buoy_is_not(self) -> None:
        self.assertTrue(assignment_is_legal(np.array([0, 1, 2, 3, 4, -1])))
        self.assertFalse(assignment_is_legal(np.array([0, 0, 2, 3, 4, 5])))

    def test_rate_choice_uses_the_supplied_pool_and_breaks_ties_stably(self) -> None:
        rate = np.array([[5.0, 1.0], [1.0, 4.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        assignments = np.array([[0, 1, -1, -1, -1, -1], [1, 0, -1, -1, -1, -1], [0, 1, -1, -1, -1, -1]])
        scores = g1_rate_scores(rate, assignments)
        index, chosen = stable_choice(scores, assignments)
        self.assertEqual(index, 0)
        self.assertEqual(chosen, [0, 1, -1, -1, -1, -1])
        self.assertLess(float(scores[1]), float(scores[0]))

    def test_forbidden_inputs_are_rejected_before_a_forward(self) -> None:
        adapter = SchemeCAdapter.__new__(SchemeCAdapter)
        with self.assertRaises(RuntimeError):
            adapter._reject_forbidden({"j15_bits": np.zeros(1)})
