"""Closure checks keep vector Hamming distinct from the buoy set."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6ij_executor_audit.audit import buoy_set
from wt.benchmark_v2.phase6ik_execution_closure.audit import assign_gate, vector_hamming


class Phase6IKClosureTests(unittest.TestCase):
    def test_open_pool_excludes_reserved_scenarios(self) -> None:
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))

    def test_self_hamming_is_not_the_buoy_set_comparison(self) -> None:
        oracle = np.asarray([9, 10, 6, 8, 11, 7], dtype=np.int64)
        swapped = np.asarray([9, 6, 10, 8, 11, 7], dtype=np.int64)
        self.assertEqual(vector_hamming(oracle, oracle), 0)
        self.assertEqual(vector_hamming(oracle, swapped), 2)
        self.assertEqual(buoy_set(oracle), buoy_set(swapped))

    def test_gate_separates_inconsistency_from_reproduction(self) -> None:
        self.assertEqual(assign_gate(False, True, False), "EXECUTION_AUDIT_INCONSISTENT")
        self.assertEqual(assign_gate(True, True, True), "EXECUTION_AUDIT_PARTIALLY_REPRODUCIBLE")
        self.assertEqual(assign_gate(True, True, False), "EXECUTION_AUDIT_REPRODUCIBLE")


if __name__ == "__main__":
    unittest.main()
