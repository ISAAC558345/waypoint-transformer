"""Phase-2 hierarchical controller checks. Does not edit env reward or PPO core."""
from __future__ import annotations

import inspect
import unittest

import numpy as np

from wt.htw_phase2.tracking import auxiliary_reward, tracking_distance
from wt.htw_phase2.vector import HierarchicalTrackingVector
from wt.htwc_phase1.oracle import oracle_conditioner
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round30.ppo import PPOAgent
from wt.round30.scenarios import training_scenarios


class HTWPhase2OracleControllerTests(unittest.TestCase):
    def test_01_auxiliary_reward_does_not_change_env_formula(self) -> None:
        self.assertAlmostEqual(auxiliary_reward(0.4, 0.2, 1.0), 0.2)
        source = inspect.getsource(PPOAgent.update)
        self.assertIn("clip", source)
        self.assertNotIn("tracking", source)

    def test_02_oracle_features_are_conditioned(self) -> None:
        scenario = training_scenarios()[0]
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        features = oracle_conditioner(info, env, horizon=5)
        self.assertEqual(features["uav"].shape[-1], 16)

    def test_03_tracking_distance_zero_at_waypoint(self) -> None:
        waypoint = np.array([[0.1, -0.2], [0.3, 0.4], [0.0, 0.0], [0.0, 0.0]])
        mask = np.array([True, True, False, False])
        self.assertAlmostEqual(tracking_distance(waypoint, waypoint, mask), 0.0)

    def test_04_vector_uses_aux_not_env_reward_when_lambda_positive(self) -> None:
        from wt.htw_phase2.experiment import oracle_low_level_conditioner

        scenario = training_scenarios()[0]
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        action = np.zeros((2, 2), dtype=np.float64)
        _, comm, _, _, _ = env.step(action, matching_mode="legacy")
        vector = HierarchicalTrackingVector((scenario,), 1, 0, oracle_low_level_conditioner, tracking_lambda=10.0)
        _, rewards, _, _ = vector.step(np.zeros((1, 2, 2), dtype=np.float64))
        self.assertLess(float(rewards[0]), float(comm) - 1.0e-6)
        self.assertTrue(np.isfinite(rewards[0]))


if __name__ == "__main__":
    unittest.main()
