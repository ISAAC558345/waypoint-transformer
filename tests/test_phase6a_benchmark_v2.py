"""Phase-6A rules. They do not open the reserved pool."""
from __future__ import annotations

import unittest

from wt.benchmark_v2.phase6a.protocol import RESERVED_POOL, assert_allowed, choose_gate


class Phase6ATests(unittest.TestCase):
    def test_reserved_and_historical_ids_are_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_allowed(320100)
        with self.assertRaises(RuntimeError):
            assert_allowed(310118)
        assert_allowed(320000)
        self.assertEqual(RESERVED_POOL[0], 320100)

    def test_missing_bit_queue_is_bv2_e_and_does_not_revise_phase5(self) -> None:
        chosen = choose_gate(
            bit_queue_exists=False,
            e1_matches_environment_transition=False,
            e1_e2_conserved=False,
            e0_saturated=True,
            endpoint_separates=True,
            switch_negatively_related=True,
            continuity_pattern=True,
        )
        self.assertEqual(chosen["gate"], "BV2-E")
        self.assertEqual(chosen["phase5zp_gate"], "CONF-C")
        self.assertEqual(chosen["phase5zq_gate"], "ATTR-C")
        self.assertFalse(chosen["phase5_revised"])
        self.assertFalse(chosen["planner_ready"])
        self.assertFalse(chosen["lambda_selected"])

    def test_valid_bit_metric_can_reach_bv2_c(self) -> None:
        chosen = choose_gate(
            bit_queue_exists=True,
            e1_matches_environment_transition=True,
            e1_e2_conserved=True,
            e0_saturated=True,
            endpoint_separates=True,
            switch_negatively_related=True,
            continuity_pattern=True,
        )
        self.assertEqual(chosen["gate"], "BV2-C")


if __name__ == "__main__":
    unittest.main()
