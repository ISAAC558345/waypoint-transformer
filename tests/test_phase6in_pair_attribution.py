"""Attribution gates stay closed when decision features do not agree."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6in_pair_attribution.audit import _key, assign_gate


class Phase6INAttributionTests(unittest.TestCase):
    def test_open_pool_excludes_reserved_scenarios(self) -> None:
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))

    def test_record_key_includes_time_and_seed(self) -> None:
        left = _key(322051, [9, 10, 6, 8, 11, 7])
        right = _key(322051, np.asarray([9, 10, 6, 8, 11, 7]))
        self.assertEqual(left, (322051, 15, (9, 10, 6, 8, 11, 7), 0))
        self.assertEqual(left, right)

    def test_gate_requires_a_consistent_decision_feature(self) -> None:
        self.assertEqual(assign_gate(True, 3), "PAIR_SWAP_ATTRIBUTION_BLOCKED")
        self.assertEqual(assign_gate(False, 1), "PAIR_SWAP_ATTRIBUTION_SUPPORTED")
        self.assertEqual(assign_gate(False, 0), "PAIR_SWAP_ATTRIBUTION_INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
