"""Phase-10.2 diagnosis tests. No training, no confirmation."""
from __future__ import annotations

import unittest

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.htw_phase9.const import HOLD
from wt.l3b.env import L3BEnv

from wt.htw_phase102.analyze import payload_caliber
from wt.htw_phase102.protocol import DESIGN_VAL_SEEDS, SNAPSHOT_RULE
from wt.htw_phase102.replay import classify_rho
from wt.htw_phase102.snapshot import capture_env, restore_env


class Phase102Tests(unittest.TestCase):
    def test_01_payload_caliber_definitions(self) -> None:
        cal = payload_caliber([8.5e6, 8.5e6], [7.25e6, 7.25e6])
        self.assertAlmostEqual(cal["mean_b1_over_mean_b0"], 7.25 / 8.5)
        self.assertAlmostEqual(cal["gap_frac"], 1.0 - 7.25 / 8.5)
        self.assertEqual(cal["n"], 2)

    def test_02_idle_excluded_from_active_in_range(self) -> None:
        self.assertTrue(all(8 <= s <= 15 for s in DESIGN_VAL_SEEDS))
        self.assertNotIn(1321, DESIGN_VAL_SEEDS)
        self.assertIn("Do not filter by later C1/C2 payoff", SNAPSHOT_RULE)

    def test_03_completion_vs_target_change(self) -> None:
        done = classify_rho(0.9, 0.0, True, False, False, False, False, False)
        switch = classify_rho(0.4, 0.0, False, True, False, False, False, False)
        self.assertIn("NORMAL_COMPLETION", done)
        self.assertIn("TARGET_CHANGE", switch)
        self.assertNotIn("NORMAL_COMPLETION", switch)

    def test_04_snapshot_restore(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        env.progress[1] = 0.3
        env.payload_bits = 2.0e6
        snap = capture_env(env)
        env.progress[1] = 0.0
        env.payload_bits = 0.0
        restore_env(env, snap)
        self.assertAlmostEqual(float(env.progress[1]), 0.3)
        self.assertAlmostEqual(float(env.payload_bits), 2.0e6)

    def test_05_zero_move_is_not_association_change(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=8)
        assignment = np.array([0, 1, 2, -1, 4, 5], dtype=np.int64)
        option = TaskOption(assignment, 0, HOLD, np.ones(6, dtype=bool), np.ones(12, dtype=bool), assignment.copy())
        move = np.zeros((6, 2), dtype=np.float64)
        after = env.step(move, option.association)
        np.testing.assert_array_equal(after["assignment"], assignment)


if __name__ == "__main__":
    unittest.main()
