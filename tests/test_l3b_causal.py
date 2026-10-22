"""Level-3-B SINR coupling checks. No training."""
from __future__ import annotations

import unittest

import numpy as np

from wt.l3b.env import L3BEnv, sinr_rates_for_links
from wt.l3b.policies import exclusive_matching, myopic_assignment


class L3BCausalTests(unittest.TestCase):
    def test_01_scale_and_no_coop(self) -> None:
        env = L3BEnv()
        self.assertEqual(env.config.num_uavs, 6)
        self.assertEqual(env.config.num_buoys, 12)
        self.assertEqual(env.config.option_hold, 15)
        info = env.reset(seed=0)
        self.assertNotIn("coop_required", info)
        self.assertIn("rate_if_assigned", info)
        self.assertIn("interference_uav_uav", info)
        self.assertIn("queue_pressure", info)

    def test_02_second_link_lowers_sinr(self) -> None:
        env = L3BEnv()
        env.reset(seed=0)
        env.uav_positions[0] = env.buoy_positions[0]
        env.uav_positions[1] = env.buoy_positions[1]
        env.assignment[:] = -1
        env.assignment[0] = 0
        env.queue[:] = 0.0
        env.queue[0] = 4.0
        env.queue[1] = 4.0
        solo = sinr_rates_for_links(env, [(0, 0)])[0][0]
        both = sinr_rates_for_links(env, [(0, 0), (1, 1)])[0][0]
        self.assertGreater(float(solo), float(both) + 1.0e-9)

    def test_03_conflict_still_wipes(self) -> None:
        env = L3BEnv()
        env.reset(seed=0)
        env.uav_positions[:] = env.buoy_positions[0]
        env.queue[:] = 0.0
        env.queue[0] = 4.0
        info = env.step(np.zeros((6, 2)), np.zeros(6, dtype=np.int64))
        self.assertEqual(float(np.sum(info["step_service_blocks"])), 0.0)
        self.assertGreater(int(info["n_conflict_pairs"]), 0)

    def test_04_graph_updates_with_assignment(self) -> None:
        env = L3BEnv()
        env.reset(seed=0)
        env.uav_positions[0] = env.buoy_positions[0]
        env.uav_positions[1] = env.buoy_positions[1]
        z = np.full(6, -1, dtype=np.int64)
        z[0] = 0
        first = env.step(np.zeros((6, 2)), z)
        z[1] = 1
        second = env.step(np.zeros((6, 2)), z)
        self.assertGreater(float(np.sum(second["interference_uav_uav"])), float(np.sum(first["interference_uav_uav"])))
        self.assertEqual(tuple(first["queue_pressure"].shape), (12,))

    def test_05_exclusive_is_conflict_free(self) -> None:
        env = L3BEnv()
        env.reset(seed=0)
        chosen = exclusive_matching(env)
        used = [int(item) for item in chosen if int(item) >= 0]
        self.assertEqual(len(used), len(set(used)))
        mine = myopic_assignment(env)
        self.assertTrue(bool(np.max(np.bincount(np.clip(mine, 0, None), minlength=1)) >= 1))


if __name__ == "__main__":
    unittest.main()
