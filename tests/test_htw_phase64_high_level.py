"""Phase-6.4 hold/reward/obs checks. Does not train the full grid."""
from __future__ import annotations

import unittest

import numpy as np

from wt.htw_phase61.features import BUOY_DIM
from wt.htw_phase61.models import L2AOptionActorCritic
from wt.htw_phase61.run_phase61 import SMOKE_CONFIG
from wt.htw_phase64.features import ENHANCED_BUOY_DIM, feature_dict_high
from wt.htw_phase64.models import HighLevelOptionModel
from wt.htw_phase64.vector import HeldHighVector, high_level_reward
from wt.l2a.env import L2AEnv
from wt.round30.ppo import PPOAgent, seed_everything


class HTWPhase64Tests(unittest.TestCase):
    def test_01_reward_modes(self) -> None:
        self.assertAlmostEqual(high_level_reward("episode", 1.0, 0.5, 2.0, 9.0, False), 0.0)
        self.assertAlmostEqual(high_level_reward("episode", 1.0, 0.5, 2.0, 9.0, True), 9.0)
        self.assertAlmostEqual(high_level_reward("delta", 1.0, 0.5, 2.0, 9.0, False), 1.0)
        self.assertAlmostEqual(high_level_reward("delta_service", 1.0, 0.5, 2.0, 9.0, False), 1.25)
        self.assertAlmostEqual(high_level_reward("delta_service_delay", 1.0, 0.5, 2.0, 9.0, False), 1.15)

    def test_02_enhanced_buoy_width(self) -> None:
        env = L2AEnv()
        info = env.reset(seed=0)
        base = feature_dict_high(info, env, enhanced=False)
        extra = feature_dict_high(info, env, last_arrival=np.ones(12), enhanced=True)
        self.assertEqual(base["buoy"].shape[-1], BUOY_DIM)
        self.assertEqual(extra["buoy"].shape[-1], ENHANCED_BUOY_DIM)
        self.assertEqual(HighLevelOptionModel(ENHANCED_BUOY_DIM).shape.buoy_dim, ENHANCED_BUOY_DIM)

    def test_03_hold_consumes_multiple_env_steps(self) -> None:
        seed_everything(0)
        low = PPOAgent(L2AOptionActorCritic(), SMOKE_CONFIG)
        for parameter in low.model.parameters():
            parameter.requires_grad = False
        vector = HeldHighVector((0,), 1, 4, low, hold=5, reward_mode="delta", enhanced=False)
        z = np.zeros((1, 5), dtype=np.int64)
        before = vector.environment_steps
        vector.step(z)
        self.assertEqual(vector.environment_steps - before, 5)


if __name__ == "__main__":
    unittest.main()
