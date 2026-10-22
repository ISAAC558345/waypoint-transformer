"""Level-2-A causal environment checks. No training."""
from __future__ import annotations

import unittest

import numpy as np

from wt.l2a.config import L2AConfig
from wt.l2a.env import L2AEnv, legal_assignment_count, project_exclusive
from wt.l2a.policies import myopic_assignment, oracle_option_assignment


class L2ACausalTests(unittest.TestCase):
    def test_01_scale_and_duration(self) -> None:
        env = L2AEnv()
        self.assertEqual(env.config.num_uavs, 5)
        self.assertEqual(env.config.num_buoys, 12)
        self.assertGreaterEqual(env.config.service_slots, 3)
        self.assertGreater(legal_assignment_count(5, 12), 31)

    def test_02_exclusive_association(self) -> None:
        env = L2AEnv()
        env.reset(seed=0)
        requested = np.zeros(5, dtype=np.int64)
        assigned = project_exclusive(requested, env.uav_positions, env.buoy_positions)
        self.assertEqual(int(np.sum(assigned == 0)), 1)

    def test_03_progress_resets_on_switch(self) -> None:
        env = L2AEnv()
        env.reset(seed=0)
        stay = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        env.step(np.zeros((5, 2)), stay)
        env.progress[:] = 0.5
        env.served_k[:] = stay
        env.step(np.zeros((5, 2)), np.array([1, 1, 2, 3, 4], dtype=np.int64))
        self.assertLess(float(env.progress[0]) + 1.0e-9, 0.5)

    def test_04_oracle_and_myopic_are_exclusive(self) -> None:
        env = L2AEnv()
        env.reset(seed=0)
        for fn in (myopic_assignment, oracle_option_assignment):
            assignment = fn(env)
            chosen = [int(item) for item in assignment if int(item) >= 0]
            self.assertEqual(len(chosen), len(set(chosen)))


if __name__ == "__main__":
    unittest.main()
