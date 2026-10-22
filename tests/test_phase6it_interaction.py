"""Interaction gate and the additive score identity."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, TRAIN_POOL, VALIDATION_POOL
from wt.benchmark_v2.phase6it_interaction.audit import context_ranges, decompositions, pair_scores
from wt.benchmark_v2.phase6it_interaction.protocol import assign_gate


class Phase6ITInteractionTests(unittest.TestCase):
    def test_pools_exclude_reserved_scenarios(self) -> None:
        self.assertTrue(set(TRAIN_POOL).isdisjoint(RESERVED_TEST))
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))

    def test_disjoint_pair_sum_has_no_interaction_and_no_context_change(self) -> None:
        matrix = np.arange(72, dtype=np.float64).reshape(6, 12)
        base = np.array([6, 7, 8, 9, 10, 11], dtype=np.int64)
        rows = [base]
        for item in decompositions(base):
            rows.extend((item["joint"], item["first"], item["second"]))
        assignments = np.vstack(rows)
        scores = pair_scores(matrix, assignments)
        table = {tuple(row): scores[index] for index, row in enumerate(assignments)}
        base_score = table[tuple(base)]
        for item in decompositions(base):
            interaction = table[tuple(item["joint"])] - base_score
            interaction -= table[tuple(item["first"])] - base_score
            interaction -= table[tuple(item["second"])] - base_score
            if item["kind"] == "disjoint":
                self.assertLess(abs(interaction), 1.0e-8)
        backgrounds = np.vstack(
            (
                np.array([[6, 7, 8, 9, 10, 11], [7, 6, 8, 9, 10, 11], [6, 7, 9, 8, 10, 11], [7, 6, 9, 8, 10, 11]]),
            )
        )
        ranges = context_ranges(backgrounds, pair_scores(matrix, backgrounds))
        self.assertTrue(ranges)
        self.assertTrue(all(value < 1.0e-8 for value in ranges))

    def test_gate_does_not_treat_a_tracking_head_as_proof_of_the_limit(self) -> None:
        self.assertEqual(assign_gate(True, 200, True, 16, False), "ASSIGNMENT_INTERACTION_AUDIT_BLOCKED")
        self.assertEqual(assign_gate(False, 10, True, 16, False), "ASSIGNMENT_INTERACTION_LIMIT_INCONCLUSIVE")
        self.assertEqual(assign_gate(False, 200, True, 4, False), "ASSIGNMENT_INTERACTION_LIMIT_INCONCLUSIVE")
        self.assertEqual(assign_gate(False, 200, True, 16, True), "ASSIGNMENT_INTERACTION_LIMIT_INCONCLUSIVE")
        self.assertEqual(assign_gate(False, 200, True, 16, False), "ASSIGNMENT_INTERACTION_LIMIT_SUPPORTED")
