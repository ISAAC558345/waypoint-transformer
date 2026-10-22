"""Phase-6.3 high-level option interface checks."""
from __future__ import annotations

import inspect
import unittest

import numpy as np

from wt.htw_phase61.features import IDLE_INDEX, OPTION_DIM
from wt.htw_phase61.models import L2AOptionActorCritic
from wt.htw_phase61.run_phase61 import SMOKE_CONFIG
from wt.htw_phase63.high_ppo import DiscretePPOAgent, HighLevelOptionModel, assignment_to_indices, indices_to_assignment
from wt.htw_phase63.vector import HierarchicalVector, collect_high
from wt.l2a.env import L2AEnv, project_exclusive
from wt.round30.ppo import PPOAgent, compute_gae, seed_everything


class HTWPhase63Tests(unittest.TestCase):
    def test_01_idle_and_exclusive_roundtrip(self) -> None:
        assignment = np.array([-1, 3, 11, 0, 6], dtype=np.int64)
        indices = assignment_to_indices(assignment)
        self.assertEqual(int(indices[0]), IDLE_INDEX)
        self.assertEqual(indices_to_assignment(indices).tolist(), assignment.tolist())
        env = L2AEnv()
        env.reset(seed=0)
        requested = np.zeros(5, dtype=np.int64)
        projected = project_exclusive(requested, env.uav_positions, env.buoy_positions)
        self.assertEqual(int(np.sum(projected == 0)), 1)

    def test_02_high_level_logits_width(self) -> None:
        model = HighLevelOptionModel()
        self.assertEqual(model.actor_head[-1].out_features, OPTION_DIM)

    def test_03_one_high_level_update(self) -> None:
        seed_everything(0)
        low = PPOAgent(L2AOptionActorCritic(), SMOKE_CONFIG)
        config = type(SMOKE_CONFIG)(rollout_length=8, num_envs=2, num_minibatches=2, update_epochs=1)
        high = DiscretePPOAgent(HighLevelOptionModel(), config)
        vector = HierarchicalVector((0, 1), 2, 9, low, lambda_c=0.5, lambda_d=0.05, high_mode="learned")
        storage, _events, health = collect_high(high, vector)
        update = high.update(storage)
        self.assertTrue(all(np.isfinite(value) for value in health.values()))
        self.assertTrue(all(np.isfinite(value) for value in update.values()))

    def test_04_ppo_core_imported_not_rewritten(self) -> None:
        source = inspect.getsource(compute_gae)
        self.assertIn("gae_lambda", source)
        self.assertIn("clip_eps", inspect.getsource(PPOAgent.update))


if __name__ == "__main__":
    unittest.main()
