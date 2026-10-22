"""Phase-9.4.2 frozen-model confirmation and order-protocol checks."""
from __future__ import annotations

import unittest

import numpy as np

from wt.htw_phase941.protocol import CONFIRMATION_SEEDS, USED_L3B_SCENARIO_SEEDS
from wt.htw_phase942.decode_order import freeze_orders, inverse_assignment, permute_uav_batch
from wt.htw_phase942.pressures import label_three, regime_of
from wt.htw_phase942.protocol import BLOCKED_SCENARIO_SEEDS, ORDER_SEED, SCAN_START


class Phase942Tests(unittest.TestCase):
    def test_01_new_seeds_blocked_from_0_39(self) -> None:
        self.assertTrue(set(range(40)).issubset(set(BLOCKED_SCENARIO_SEEDS)))
        self.assertEqual(set(CONFIRMATION_SEEDS) | set(USED_L3B_SCENARIO_SEEDS), set(BLOCKED_SCENARIO_SEEDS))
        self.assertGreaterEqual(SCAN_START, 40)

    def test_02_orders_frozen_and_o0_identity(self) -> None:
        orders = freeze_orders(ORDER_SEED)
        self.assertEqual(orders["O0"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(orders["O1"], [5, 4, 3, 2, 1, 0])
        self.assertEqual(len(orders), 6)
        again = freeze_orders(ORDER_SEED)
        self.assertEqual(orders, again)

    def test_03_inverse_permute_roundtrip(self) -> None:
        order = np.array([5, 4, 3, 2, 1, 0], dtype=np.int64)
        orig = np.array([0, 1, 2, 3, 4, -1], dtype=np.int64)
        permuted = orig[order]
        back = inverse_assignment(permuted, order, 6)
        self.assertTrue(np.array_equal(back, orig))

    def test_04_regimes_2x2(self) -> None:
        self.assertEqual(regime_of("low", "low"), "R1")
        self.assertEqual(regime_of("high", "low"), "R2")
        self.assertEqual(regime_of("low", "high"), "R3")
        self.assertEqual(regime_of("high", "high"), "R4")
        self.assertIsNone(regime_of("medium", "high"))
        self.assertEqual(label_three(0.0, 1.0, 2.0), "low")

    def test_05_permute_batch_uav_dim(self) -> None:
        batch = {
            "uav": np.arange(6 * 8, dtype=np.float32).reshape(1, 6, 8),
            "uav_mask": np.ones((1, 6), dtype=bool),
            "edge": np.arange(6 * 12 * 3, dtype=np.float32).reshape(1, 6, 12, 3),
            "edge_uu": np.arange(6 * 6 * 2, dtype=np.float32).reshape(1, 6, 6, 2),
            "buoy": np.zeros((1, 12, 7), dtype=np.float32),
        }
        order = np.array([1, 0, 2, 3, 4, 5])
        out = permute_uav_batch(batch, order)
        self.assertTrue(np.allclose(out["uav"][0, 0], batch["uav"][0, 1]))


if __name__ == "__main__":
    unittest.main()
