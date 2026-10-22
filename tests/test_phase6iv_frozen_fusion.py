"""Rank-fusion checks that do not open reserved scenarios."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, TRAIN_POOL, VALIDATION_POOL
from wt.benchmark_v2.phase6iv_frozen_fusion.audit import average_rank, fuse_ranks, fusion_order
from wt.benchmark_v2.phase6iv_frozen_fusion.protocol import RULES, assign_gate, rule_supported


class Phase6IVFrozenFusionTests(unittest.TestCase):
    def test_pools_and_gate_are_fixed(self) -> None:
        self.assertTrue(set(TRAIN_POOL).isdisjoint(set(VALIDATION_POOL)))
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))
        self.assertEqual(RULES, ("B", "C", "D", "E"))
        passed = {
            "top1_above_scheme_c": True,
            "regret_not_worse": True,
            "mean_ci_above_scheme_c": True,
            "scenarios_consistent": True,
            "gain_not_concentrated": True,
            "supported": True,
        }
        self.assertTrue(rule_supported(passed))
        self.assertEqual(assign_gate(False, True, [passed] * 4), "FROZEN_FUSION_GAIN_SUPPORTED")
        failed = dict(passed)
        failed["supported"] = False
        failed["top1_above_scheme_c"] = False
        self.assertEqual(assign_gate(False, True, [failed] * 4), "FROZEN_FUSION_GAIN_INCONCLUSIVE")
        self.assertEqual(assign_gate(True, True, [passed] * 4), "FROZEN_FUSION_AUDIT_BLOCKED")

    def test_ties_share_a_rank_and_scheme_c_breaks_fusion_ties(self) -> None:
        scores = np.array([5.0, 5.0, 1.0])
        ranks = average_rank(scores)
        self.assertEqual(float(ranks[0]), float(ranks[1]))
        self.assertLess(float(ranks[0]), float(ranks[2]))
        scheme = average_rank(np.array([3.0, 2.0, 1.0]))
        other = average_rank(np.array([2.0, 3.0, 1.0]))
        fused = fuse_ranks([other], scheme)
        self.assertAlmostEqual(float(fused[0]), float(fused[1]))
        order = fusion_order(fused, scheme)
        self.assertEqual(int(order[0]), 0)
