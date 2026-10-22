"""Grouping rules for the validation audit. Model scoring is not run here."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6ie_validation_audit.audit import _score_view, aggregate
from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import DecisionDataset
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST


class Phase6IEAuditTests(unittest.TestCase):
    def test_validation_index_stays_inside_the_open_pool(self) -> None:
        seen = {int(scenario) for scenario, _time in DecisionDataset("validation").index}
        self.assertTrue(seen.isdisjoint(RESERVED_TEST))
        self.assertEqual(min(seen), 322048)
        self.assertEqual(max(seen), 322063)

    def test_stable_tie_break_and_exact_pair_shield(self) -> None:
        labels = np.asarray([5.0, 5.0, 1.0], dtype=np.float64)
        ranks = np.asarray([1, 1, 2], dtype=np.int64)
        view = _score_view(np.asarray([1.0, 1.0, 0.0]), labels, ranks)
        self.assertEqual(view["chosen"], 0)
        self.assertEqual(view["score_tie_count"], 2)
        self.assertEqual(view["equal_pairs"], 1)
        self.assertEqual(view["strict_pairs_inside_best_group"], 0)
        self.assertEqual(view["top1_matches_regret_rule"], 1)

    def test_payload_groups_follow_the_sign_of_the_delta(self) -> None:
        rows = [
            {"payload_delta": 5.0, "frozen_top1": 1.0, "epoch6_top1": 0.0},
            {"payload_delta": 0.0, "frozen_top1": 0.0, "epoch6_top1": 1.0},
            {"payload_delta": -2.0, "frozen_top1": 1.0, "epoch6_top1": 1.0},
        ]
        packed = aggregate(rows)
        self.assertEqual(packed["improved"]["n"], 1)
        self.assertEqual(packed["tied"]["n"], 1)
        self.assertEqual(packed["declined"]["n"], 1)
        self.assertEqual(packed["frozen_hit_epoch6_miss"]["n"], 1)
        self.assertEqual(packed["same_top1_payload_down"]["mean"], -2.0)


if __name__ == "__main__":
    unittest.main()
