"""Pool-alignment rules. Reserved scenarios and the planner stay closed."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6ix_candidate_alignment.audit import _compare_rows, formal_order, scheme_c_order
from wt.benchmark_v2.phase6ix_candidate_alignment.protocol import FORMAL_K, FROZEN_K, GENERATOR, assign_gate


class Phase6IXCandidateAlignmentTests(unittest.TestCase):
    def test_gate_requires_identical_rows(self) -> None:
        self.assertEqual(assign_gate(comparable=False, identical=False), "CANDIDATE_POOL_ALIGNMENT_INCONCLUSIVE")
        self.assertEqual(assign_gate(comparable=True, identical=False), "CANDIDATE_POOL_ALIGNMENT_BLOCKED")
        self.assertEqual(assign_gate(comparable=True, identical=True), "CANDIDATE_POOL_ALIGNMENT_SUPPORTED")
        self.assertEqual(GENERATOR, "G1_RATE")
        self.assertEqual(FORMAL_K, 5)
        self.assertEqual(FROZEN_K, 6)

    def test_scheme_c_order_sorts_and_formal_order_keeps_enumeration(self) -> None:
        pool = [np.array([1, -1]), np.array([-1, -1]), np.array([0, 1])]
        self.assertTrue(np.array_equal(formal_order(pool), np.array([[1, -1], [-1, -1], [0, 1]])))
        self.assertTrue(np.array_equal(scheme_c_order(pool), np.array([[-1, -1], [0, 1], [1, -1]])))

    def test_content_difference_is_not_hidden_by_equal_counts(self) -> None:
        left = np.array([[0, 1], [1, 0]])
        right = np.array([[0, 1], [2, 3]])
        compared = _compare_rows(left, right)
        self.assertEqual(compared["same_count"], 1)
        self.assertEqual(compared["same_set"], 0)
        self.assertEqual(compared["same_order"], 0)
        self.assertEqual(compared["only_reference"], 1)
