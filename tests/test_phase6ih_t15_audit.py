"""Alignment rules for the t=15 audit. Scoring is not run here."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, TRAIN_POOL, VALIDATION_POOL
from wt.benchmark_v2.phase6ih_t15_audit.audit import lexicographic_rows, utility_rank_matches


class Phase6IHAuditTests(unittest.TestCase):
    def test_open_pools_exclude_reserved_scenarios(self) -> None:
        self.assertTrue(set(TRAIN_POOL).isdisjoint(RESERVED_TEST))
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))

    def test_lexicographic_order_rejects_a_swap(self) -> None:
        ordered = np.asarray([[-1, 0, 1, 2, 3, 4], [-1, 0, 1, 2, 3, 5]], dtype=np.int64)
        swapped = ordered[::-1].copy()
        self.assertTrue(lexicographic_rows(ordered))
        self.assertFalse(lexicographic_rows(swapped))

    def test_one_bit_group_matches_the_stored_rank_rule(self) -> None:
        labels = np.asarray([10.0, 9.5, 8.0], dtype=np.float64)
        ranks = np.asarray([1, 1, 2], dtype=np.int64)
        self.assertTrue(utility_rank_matches(labels, ranks, eps=1.0))
        self.assertFalse(utility_rank_matches(labels, np.asarray([1, 2, 3]), eps=1.0))


if __name__ == "__main__":
    unittest.main()
