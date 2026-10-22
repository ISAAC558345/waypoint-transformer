"""Distance potential. Inside the service radius the potential is zero and bits are not in it."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6io_obs_reward.shaping import distance_potential, shaping_reward
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL, PPO


class Phase6IOShapingTests(unittest.TestCase):
    def test_inside_radius_has_zero_potential_and_terminal_is_zero(self) -> None:
        self.assertEqual(float(ALPHA_RESIDUAL), 0.25)
        env = L3BBitQueueEnvV2()
        env.reset(seed=321096)
        env.uav_positions[:] = np.array([[20.0 + i * 20.0, 20.0] for i in range(6)])
        buoy = env.buoy_positions[0].copy()
        env.uav_positions[0] = buoy + np.array([4.0, 0.0])
        association = np.full(6, -1, dtype=np.int64)
        association[0] = 0
        self.assertEqual(distance_potential(env, association), 0.0)
        env.uav_positions[0] = buoy + np.array([16.0 + 22.0, 0.0])
        phi = distance_potential(env, association)
        self.assertAlmostEqual(phi, -1.0, places=6)
        self.assertAlmostEqual(shaping_reward(phi, 0.0, True), -phi, places=6)
        self.assertAlmostEqual(shaping_reward(-1.0, -0.5, False, gamma=float(PPO.gamma)), float(PPO.gamma) * -0.5 - (-1.0), places=6)
        self.assertGreater(shaping_reward(-1.0, 0.0, False), 0.0)


if __name__ == "__main__":
    unittest.main()
