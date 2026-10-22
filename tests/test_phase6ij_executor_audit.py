"""Pairing rules for the frozen-executor audit. Rollouts are not run here."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6ij_executor_audit.audit import assign_gate, buoy_set, pair_ok


class Phase6IJExecutorTests(unittest.TestCase):
    def test_open_pool_excludes_reserved_scenarios(self) -> None:
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))

    def test_same_buoy_set_requires_a_different_pairing(self) -> None:
        oracle = np.asarray([9, 10, 6, 8, 11, 7], dtype=np.int64)
        swapped = np.asarray([9, 6, 10, 8, 11, 7], dtype=np.int64)
        other = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
        self.assertEqual(buoy_set(oracle), buoy_set(swapped))
        self.assertTrue(pair_ok(oracle, swapped))
        self.assertFalse(pair_ok(oracle, oracle))
        self.assertFalse(pair_ok(oracle, other))

    def test_gate_requires_a_stable_positive_execution_gap(self) -> None:
        blocked = {"failure": 1, "paired": 1, "delta_causal": 1.0e6}
        self.assertEqual(assign_gate([blocked] * 12, False), "EXECUTION_AUDIT_BLOCKED")
        weak = {"failure": 0, "paired": 1, "delta_causal": 0.0}
        self.assertEqual(assign_gate([weak] * 12, False), "EXECUTION_GAIN_INCONCLUSIVE")
        strong = {"failure": 0, "paired": 1, "delta_causal": 1.0e6}
        self.assertEqual(assign_gate([strong] * 12, False), "EXECUTION_GAIN_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
