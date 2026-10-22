"""Phase-10.0 interface tests. No scientific confirmation rerun."""
from __future__ import annotations

import unittest

import numpy as np

from wt.htw_phase100.compatibility import audit
from wt.htw_phase100.contract import TaskOption
from wt.htw_phase100.protocol import DESIGN_SEEDS, HOLD
from wt.htw_phase100.rule_executor import RuleExecutorAdapter
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward


class Phase100Tests(unittest.TestCase):
    def test_01_adapter_matches_fly_toward_and_refuses_remap(self) -> None:
        env = L3BEnv()
        env.reset(seed=0)
        assignment = np.array([0, 1, 2, -1, 4, 5], dtype=np.int64)
        option = TaskOption(
            targets=assignment.copy(),
            decision_time=0,
            remaining_hold=HOLD,
            uav_mask=np.ones(6, dtype=bool),
            buoy_mask=np.ones(12, dtype=bool),
            association=assignment.copy(),
        )
        move = RuleExecutorAdapter().act(env, option)
        np.testing.assert_array_equal(move, fly_toward(env, assignment))
        self.assertEqual(float(np.linalg.norm(move[3])), 0.0)
        bad = option.copy()
        bad.association = assignment.copy()
        bad.association[0] = 7
        with self.assertRaises(RuntimeError):
            RuleExecutorAdapter().act(env, bad)

    def test_02_hold_schedule_matches_t48(self) -> None:
        env = L3BEnv()
        self.assertEqual(int(env.config.episode_horizon), 48)
        self.assertEqual(int(HOLD), 15)
        times = []
        pending = 0
        for t in range(48):
            if pending <= 0:
                times.append(t)
                pending = int(HOLD)
            pending -= 1
        self.assertEqual(times, [0, 15, 30, 45])

    def test_03_design_seeds_not_confirmation(self) -> None:
        self.assertEqual(DESIGN_SEEDS, (0, 1, 2, 3))
        self.assertTrue(all(s < 24 for s in DESIGN_SEEDS))

    def test_04_compatibility_does_not_claim_direct_load(self) -> None:
        result = audit()
        self.assertEqual(result["verdict"], "needs_independent_adapter")
        self.assertFalse(result["trained_this_phase"])
        self.assertIn("num_uavs", {row["item"] for row in result["issues"]})

    def test_05_idle_not_in_range(self) -> None:
        env = L3BEnv()
        env.reset(seed=0)
        self.assertFalse(env._in_range(0, -1))


if __name__ == "__main__":
    unittest.main()
