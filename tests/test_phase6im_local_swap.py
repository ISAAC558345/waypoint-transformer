"""Local swap legality stays inside one buoy set."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6ij_executor_audit.audit import buoy_set
from wt.benchmark_v2.phase6ik_execution_closure.audit import vector_hamming
from wt.benchmark_v2.phase6im_local_swap.audit import assign_gate, single_swaps, transposition_distance, two_swap_assignments


class Phase6IMLocalSwapTests(unittest.TestCase):
    def test_open_pool_excludes_reserved_scenarios(self) -> None:
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))

    def test_single_swap_preserves_the_buoy_set(self) -> None:
        base = np.asarray([9, 6, 10, 8, 11, 7], dtype=np.int64)
        found = single_swaps(base)
        self.assertEqual(len(found), 15)
        for left, right, row in found:
            self.assertLess(left, right)
            self.assertTrue(is_legal(row, 6, 12))
            self.assertEqual(buoy_set(row), buoy_set(base))
            self.assertEqual(vector_hamming(row, base), 2)

    def test_two_swaps_exclude_the_base_and_the_single_swaps(self) -> None:
        base = np.asarray([9, 6, 10, 8, 11, 7], dtype=np.int64)
        singles = {tuple(int(item) for item in row) for _left, _right, row in single_swaps(base)}
        doubles = two_swap_assignments(base)
        keys = {tuple(int(item) for item in row) for row in doubles}
        self.assertNotIn(tuple(int(item) for item in base), keys)
        self.assertTrue(keys.isdisjoint(singles))
        self.assertTrue(all(is_legal(row, 6, 12) for row in doubles))

    def test_transposition_distance_counts_one_exchange(self) -> None:
        scheme = np.asarray([9, 6, 10, 8, 11, 7], dtype=np.int64)
        oracle = np.asarray([9, 10, 6, 8, 11, 7], dtype=np.int64)
        self.assertEqual(transposition_distance(scheme, oracle), 1)
        self.assertIsNone(transposition_distance(scheme, np.asarray([8, 6, 10, 9, 11, 5], dtype=np.int64)))

    def test_gate_requires_both_effect_and_explanation(self) -> None:
        self.assertEqual(assign_gate(True, 16, 16), "LOCAL_PAIR_AUDIT_BLOCKED")
        self.assertEqual(assign_gate(False, 16, 12), "LOCAL_PAIR_EFFECT_SUPPORTED")
        self.assertEqual(assign_gate(False, 16, 2), "LOCAL_PAIR_EFFECT_INCONCLUSIVE")
        self.assertEqual(assign_gate(False, 0, 0), "LOCAL_PAIR_EFFECT_INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
