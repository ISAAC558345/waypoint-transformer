"""Reward and split checks that do not start training."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL, TRAIN_POOL
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import low_level_reward
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL


class Phase6IRAlignmentTests(unittest.TestCase):
    def test_reward_uses_only_the_current_step_and_splits_stay_disjoint(self) -> None:
        self.assertEqual(float(ALPHA_RESIDUAL), 0.25)
        zeros = np.zeros((6, 2), dtype=np.float64)
        reward, comm = low_level_reward(1_500_000.0, zeros, zeros, zeros, 1_000_000.0)
        self.assertAlmostEqual(comm, 1.5, places=6)
        self.assertAlmostEqual(reward, 1.5, places=6)
        window = (400_000.0, 500_000.0, 600_000.0)
        total = 0.0
        for served in window:
            step_reward, step_comm = low_level_reward(served, zeros, zeros, zeros, 1_000_000.0)
            self.assertAlmostEqual(step_comm * 1_000_000.0, served, places=3)
            total += step_reward
        self.assertAlmostEqual(total, 1.5, places=6)
        self.assertTrue(set(TRAIN_POOL).isdisjoint(DEV_POOL))
        self.assertTrue(set(TRAIN_POOL).isdisjoint(RESERVED_TEST))
        self.assertTrue(set(DEV_POOL).isdisjoint(RESERVED_TEST))
        self.assertEqual(list(range(0, 48, 15)), [0, 15, 30, 45])


if __name__ == "__main__":
    unittest.main()
