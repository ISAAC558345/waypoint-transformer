"""Phase-6.2 controller reward and interface checks. No full training grid."""
from __future__ import annotations

import unittest

import numpy as np

from wt.htw_phase62.rewards import low_level_reward, mean_target_distance_slots
from wt.htw_phase62.vector import GoalOptionVector
from wt.l2a.env import L2AEnv
from wt.round30.ppo import PPOConfig


class HTWPhase62Tests(unittest.TestCase):
    def test_01_reward_variants(self) -> None:
        self.assertAlmostEqual(low_level_reward(1.0, 0.5, 2.0, 0.0, 0.0), 1.0)
        self.assertAlmostEqual(low_level_reward(1.0, 0.5, 2.0, 0.5, 0.0), 1.25)
        self.assertAlmostEqual(low_level_reward(1.0, 0.5, 2.0, 0.5, 0.05), 1.15)

    def test_02_distance_is_to_option_buoy(self) -> None:
        env = L2AEnv()
        env.reset(seed=0)
        assignment = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        dist = mean_target_distance_slots(env, assignment)
        self.assertGreater(dist, 0.0)
        idle = mean_target_distance_slots(env, np.full(5, -1, dtype=np.int64))
        self.assertAlmostEqual(idle, 0.0)

    def test_03_vector_keeps_env_comm_separate(self) -> None:
        vector = GoalOptionVector((0,), 1, 3, lambda_c=1.0, lambda_d=0.05)
        action = np.zeros((1, 5, 2), dtype=np.float32)
        _features, rewards, _dones, _done_rows = vector.step(action)
        self.assertEqual(rewards.shape, (1,))
        self.assertTrue(np.isfinite(float(rewards[0])))
        self.assertEqual(inspect_env_step_name(), "association")

    def test_04_budget_matches_phase61(self) -> None:
        from wt.htw_phase61.run_phase61 import MINI_UPDATES, SMOKE_CONFIG, SEED
        from wt.htw_phase62 import run_phase62

        self.assertEqual(run_phase62.MINI_UPDATES, MINI_UPDATES)
        self.assertEqual(run_phase62.SEED, SEED)
        self.assertEqual(run_phase62.SMOKE_CONFIG.rollout_length, SMOKE_CONFIG.rollout_length)
        self.assertIsInstance(SMOKE_CONFIG, PPOConfig)


def inspect_env_step_name() -> str:
    import inspect

    return inspect.signature(L2AEnv.step).parameters["association"].name


if __name__ == "__main__":
    unittest.main()
