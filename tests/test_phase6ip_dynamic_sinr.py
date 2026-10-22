"""Dynamic SINR observability stays offline and does not treat an empty link as zero."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6ip_dynamic_sinr.audit import assign_gate, straight_line_channel
from wt.l3b.config import L3BConfig


class Phase6IPDynamicSinrTests(unittest.TestCase):
    def test_open_pool_excludes_reserved_scenarios(self) -> None:
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))

    def test_gate_prefers_a_verified_decision_time_explanation(self) -> None:
        self.assertEqual(assign_gate(True, True, True, True), "OBSERVABILITY_INCONCLUSIVE")
        self.assertEqual(assign_gate(False, True, False, False), "CURRENT_STATE_FEATURES_SUFFICIENT")
        self.assertEqual(assign_gate(False, False, True, False), "CURRENT_STATE_FEATURES_SUFFICIENT")
        self.assertEqual(assign_gate(False, False, False, True), "DYNAMIC_RESPONSE_REQUIRED")
        self.assertEqual(assign_gate(False, False, False, False), "OBSERVABILITY_INCONCLUSIVE")

    def test_straight_line_marks_a_uav_that_never_serves(self) -> None:
        config = L3BConfig()
        arrays = {
            "uav_position_m": np.asarray([[0.0, 0.0], [0.0, 80.0]], dtype=np.float64),
            "buoy_position_m": np.asarray([[0.0, 0.0], [50.0, 0.0]], dtype=np.float64),
            "q_bits": np.asarray([1.0e6, 0.0], dtype=np.float64),
        }
        channel = straight_line_channel(arrays, np.asarray([0, 1], dtype=np.int64), config)
        self.assertGreater(float(channel["delivered_bits"][0, 0]), 0.0)
        self.assertEqual(int(channel["service_steps"][0, 1]), 0)
        self.assertTrue(np.isnan(channel["mean_sinr"][0, 1]))
        self.assertGreaterEqual(int(channel["entry"][0, 0]), 0)

    def test_far_uav_enters_on_the_straight_line_step(self) -> None:
        config = L3BConfig()
        arrays = {
            "uav_position_m": np.asarray([[0.0, 80.0]], dtype=np.float64),
            "buoy_position_m": np.asarray([[0.0, 0.0]], dtype=np.float64),
            "q_bits": np.asarray([1.0e6], dtype=np.float64),
        }
        channel = straight_line_channel(arrays, np.asarray([0], dtype=np.int64), config)
        self.assertEqual(int(channel["entry"][0, 0]), 2)
        self.assertTrue(np.isfinite(channel["mean_sinr"][0, 0]))


if __name__ == "__main__":
    unittest.main()
