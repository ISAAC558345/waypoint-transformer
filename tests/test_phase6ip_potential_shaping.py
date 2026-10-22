"""Direction of the distance potential. Bits, idle motion, and termination are covered."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import low_level_reward
from wt.benchmark_v2.phase6ip_potential_shaping.shaping import direction_report, directions_pass, distance_potential, shaping_reward
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL, PPO


class Phase6IPPotentialTests(unittest.TestCase):
    def test_direction_cases(self) -> None:
        self.assertEqual(float(ALPHA_RESIDUAL), 0.25)
        self.assertEqual(float(PPO.gamma), 0.99)
        report = direction_report()
        self.assertTrue(directions_pass(report))
        self.assertEqual(report["units"], "m")
        self.assertAlmostEqual(report["phi_one_step"], -1.0, places=6)
        self.assertGreater(report["toward"], 0.0)
        self.assertLess(report["away"], 0.0)
        self.assertAlmostEqual(report["inside"], 0.0, places=8)
        self.assertAlmostEqual(report["idle_motion"], 0.0, places=8)
        self.assertAlmostEqual(report["idle_shift_minus_stay"], 0.0, places=8)
        self.assertAlmostEqual(report["terminal"], 1.0, places=6)
        self.assertAlmostEqual(report["terminal_inside"], 0.0, places=8)
        self.assertTrue(report["phi_unchanged_by_payload"])
        self.assertAlmostEqual(report["distance_norm_m"], report["service_radius_m"] + report["distance_norm_m"] - report["service_radius_m"], places=6)

    def test_toward_away_inside_idle_and_terminal_on_one_env(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=321000)
        radius = float(env.config.service_radius_m)
        step = float(env.config.step_length_m)
        self.assertGreater(radius, 0.0)
        self.assertEqual(step, float(direction_report()["distance_norm_m"]))
        buoy = env.buoy_positions[0].copy()
        association = np.full(env.config.num_uavs, -1, dtype=np.int64)
        association[0] = 0
        env.uav_positions[0] = buoy + np.array([radius + step, 0.0])
        phi = distance_potential(env, association)
        env.uav_positions[0] = buoy + np.array([radius, 0.0])
        self.assertGreater(shaping_reward(phi, distance_potential(env, association), False), 0.0)
        env.uav_positions[0] = buoy + np.array([radius + 2.0 * step, 0.0])
        self.assertLess(shaping_reward(phi, distance_potential(env, association), False), 0.0)
        env.uav_positions[0] = buoy + np.array([0.5 * radius, 0.0])
        self.assertEqual(distance_potential(env, association), 0.0)
        self.assertEqual(shaping_reward(0.0, distance_potential(env, association), False), 0.0)
        idle = np.full(env.config.num_uavs, -1, dtype=np.int64)
        env.uav_positions[1] = buoy + np.array([radius + 5.0 * step, 0.0])
        self.assertEqual(distance_potential(env, idle), 0.0)
        self.assertEqual(shaping_reward(0.0, distance_potential(env, idle), False), 0.0)
        self.assertAlmostEqual(shaping_reward(-1.0, -3.0, True), 1.0, places=6)
        self.assertAlmostEqual(shaping_reward(-1.0, -3.0, True, gamma=float(PPO.gamma)), -(-1.0), places=6)
        self.assertNotAlmostEqual(shaping_reward(-1.0, -3.0, False), shaping_reward(-1.0, -3.0, True), places=5)

    def test_shaping_does_not_replace_the_bit_reward(self) -> None:
        zeros = np.zeros((6, 2), dtype=np.float64)
        task, comm = low_level_reward(2_000_000.0, zeros, zeros, zeros, 1_000_000.0)
        self.assertAlmostEqual(comm, 2.0, places=6)
        self.assertAlmostEqual(task, 2.0, places=6)
        bonus = shaping_reward(-1.0, 0.0, False)
        self.assertGreater(bonus, 0.0)
        self.assertAlmostEqual((task + bonus) - task, bonus, places=6)
        env = L3BBitQueueEnvV2()
        env.reset(seed=321000)
        association = np.full(env.config.num_uavs, -1, dtype=np.int64)
        association[0] = 0
        env.uav_positions[0] = env.buoy_positions[0] + np.array([4.0, 0.0])
        before = distance_potential(env, association)
        env.payload_bits = 5.0e7
        env.q_bits[:] = 123.0
        self.assertEqual(distance_potential(env, association), before)


if __name__ == "__main__":
    unittest.main()
