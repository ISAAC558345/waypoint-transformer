"""Level-3-A competition causal checks. No training."""
from __future__ import annotations

import unittest

import numpy as np

from wt.l3a.env import L3AEnv, conflict_mask, legal_assignment_count
from wt.l3a.policies import exclusive_myopic_assignment, myopic_assignment, oracle_option_assignment


class L3ACausalTests(unittest.TestCase):
    def test_01_scale(self) -> None:
        env = L3AEnv()
        self.assertEqual(env.config.num_uavs, 6)
        self.assertEqual(env.config.num_buoys, 12)
        self.assertEqual(env.config.option_hold, 15)
        self.assertGreater(legal_assignment_count(6, 12), legal_assignment_count(5, 12))

    def test_02_conflict_wipes_all_not_closer_wins(self) -> None:
        env = L3AEnv()
        env.reset(seed=0)
        env.uav_positions[:] = env.buoy_positions[0]
        env.queue[:] = 0.0
        env.queue[0] = 4.0
        requested = np.zeros(6, dtype=np.int64)
        info = env.step(np.zeros((6, 2)), requested)
        self.assertGreater(int(info["n_conflict_pairs"]), 0)
        self.assertEqual(float(np.sum(info["step_service_blocks"])), 0.0)
        self.assertTrue(bool(np.all(info["conflict_uav_mask"])))

    def test_03_exclusive_request_can_serve(self) -> None:
        env = L3AEnv()
        env.reset(seed=0)
        env.uav_positions[0] = env.buoy_positions[0]
        env.queue[:] = 0.0
        env.queue[0] = 4.0
        requested = np.full(6, -1, dtype=np.int64)
        requested[0] = 0
        info = env.step(np.zeros((6, 2)), requested)
        self.assertTrue(bool(info["exclusive_ok"]))
        self.assertGreater(float(env.progress[0]), 0.0)

    def test_04_myopic_can_conflict_oracle_exclusive(self) -> None:
        env = L3AEnv()
        env.reset(seed=0)
        mine = myopic_assignment(env)
        self.assertTrue(bool(np.any(conflict_mask(mine, env.config.num_buoys))))
        option = oracle_option_assignment(env)
        chosen = [int(item) for item in option if int(item) >= 0]
        self.assertEqual(len(chosen), len(set(chosen)))
        exclusive = exclusive_myopic_assignment(env)
        ex_chosen = [int(item) for item in exclusive if int(item) >= 0]
        self.assertEqual(len(ex_chosen), len(set(ex_chosen)))

    def test_05_no_sinr_coop_fields(self) -> None:
        info = L3AEnv().reset(seed=0)
        self.assertNotIn("sinr", info)
        self.assertNotIn("coop_required", info)


if __name__ == "__main__":
    unittest.main()
