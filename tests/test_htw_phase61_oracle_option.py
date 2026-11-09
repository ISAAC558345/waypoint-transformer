"""Phase-6.1 interface checks. Does not train the formal budget."""
from __future__ import annotations

import inspect
import unittest

import numpy as np

from wt.legacy.htw_historical61.features import OPTION_DIM, OPTION_UAV_DIM, UAV_DIM, encode_option, feature_dict_l2a
from wt.legacy.htw_historical61.models import L2AOptionActorCritic, L2AStateActorCritic
from wt.legacy.htw_historical61.vector import L2AVector
from wt.legacy.queue_task_causal.env import L2AEnv
from wt.legacy.queue_task_causal.policies import oracle_option_assignment
from wt.legacy.deepsets_ppo import ppo as ppo_module
from wt.algorithms.ppo_common import PPOConfig, seed_everything
from wt.legacy.deepsets_ppo.ppo import PPOAgent
from wt.legacy.deepsets_ppo.runner import collect


class HTWPhase61Tests(unittest.TestCase):
    def test_01_option_one_hot_and_idle(self) -> None:
        encoded = encode_option(np.array([-1, 3, 11, 0, 6], dtype=np.int64))
        self.assertEqual(encoded.shape, (5, OPTION_DIM))
        self.assertAlmostEqual(float(encoded[0, 12]), 1.0)
        self.assertAlmostEqual(float(encoded[1, 3]), 1.0)
        self.assertAlmostEqual(float(np.sum(encoded)), 5.0)

    def test_02_feature_width(self) -> None:
        env = L2AEnv()
        info = env.reset(seed=0)
        features = feature_dict_l2a(info, env)
        self.assertEqual(features["uav"].shape, (5, UAV_DIM))
        self.assertEqual(features["buoy"].shape, (12, 7))
        self.assertTrue(bool(np.all(features["uav_mask"])))
        option = L2AOptionActorCritic()
        state = L2AStateActorCritic()
        self.assertEqual(option.actor.shape.uav_dim, OPTION_UAV_DIM)
        self.assertEqual(option.actor.shape.max_uavs, 5)
        self.assertEqual(state.actor.shape.uav_dim, UAV_DIM)

    def test_03_oracle_association_stays_exclusive(self) -> None:
        env = L2AEnv()
        env.reset(seed=1)
        chosen = [int(item) for item in oracle_option_assignment(env) if int(item) >= 0]
        self.assertEqual(len(chosen), len(set(chosen)))

    def test_04_one_ppo_update_is_finite(self) -> None:
        seed_everything(0)
        agent = PPOAgent(L2AOptionActorCritic(), PPOConfig(rollout_length=8, num_envs=2, num_minibatches=2, update_epochs=1))
        vector = L2AVector((0, 1), 2, 11, use_option=True)
        storage, _events, health = collect(agent, vector)
        update = agent.update(storage)
        self.assertTrue(all(np.isfinite(value) for value in health.values()))
        self.assertTrue(all(np.isfinite(value) for value in update.values() if isinstance(value, float)))

    def test_05_ppo_core_symbols_unchanged(self) -> None:
        source = inspect.getsource(ppo_module.compute_gae)
        self.assertIn("gae_lambda", source)
        self.assertIn("clip_eps", inspect.getsource(ppo_module.PPOAgent.update))
        self.assertEqual(inspect.signature(L2AEnv.step).parameters["association"].name, "association")


if __name__ == "__main__":
    unittest.main()