"""Phase-7 HGNN high-level option checks."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from wt.htw_phase61.features import BUOY_DIM
from wt.htw_phase61.models import L2AOptionActorCritic
from wt.htw_phase61.run_phase61 import SMOKE_CONFIG
from wt.htw_phase7.graph import EDGE_DIM, build_edge_features
from wt.htw_phase7.models import DeepSetsOptionModel, HeteroOptionHGNN, MLPOptionModel
from wt.htw_phase7.vector import GraphHeldVector
from wt.l2a.env import L2AEnv
from wt.round30.ppo import PPOAgent, PPOConfig, seed_everything
from wt.htw_phase7.agent import GraphDiscretePPOAgent
from wt.htw_phase7.vector import collect_graph_held


class HTWPhase7Tests(unittest.TestCase):
    def test_01_edge_shape(self) -> None:
        env = L2AEnv()
        info = env.reset(seed=0)
        edge = build_edge_features(info, env)
        self.assertEqual(edge.shape, (5, 12, EDGE_DIM))
        self.assertTrue(np.all(np.isfinite(edge)))

    def test_02_encoders_forward(self) -> None:
        batch = 2
        uav = torch.zeros(batch, 5, 8)
        buoy = torch.zeros(batch, 12, BUOY_DIM)
        edge = torch.zeros(batch, 5, 12, EDGE_DIM)
        um = torch.ones(batch, 5, dtype=torch.bool)
        bm = torch.ones(batch, 12, dtype=torch.bool)
        glob = torch.zeros(batch, 3)
        for model in (MLPOptionModel(BUOY_DIM), DeepSetsOptionModel(BUOY_DIM), HeteroOptionHGNN(BUOY_DIM)):
            logits, value = model(uav, buoy, um, bm, glob, edge)
            self.assertEqual(tuple(logits.shape), (batch, 5, 13))
            self.assertEqual(tuple(value.shape), (batch,))

    def test_03_one_hgnn_update(self) -> None:
        seed_everything(0)
        low = PPOAgent(L2AOptionActorCritic(), SMOKE_CONFIG)
        for parameter in low.model.parameters():
            parameter.requires_grad = False
        config = PPOConfig(rollout_length=8, num_envs=2, num_minibatches=2, update_epochs=1)
        high = GraphDiscretePPOAgent(HeteroOptionHGNN(BUOY_DIM), config)
        vector = GraphHeldVector((0, 1), 2, 5, low, hold=15, reward_mode="delta", enhanced=False)
        storage, _events, health = collect_graph_held(high, vector)
        self.assertIn("edge", storage)
        update = high.update(storage)
        self.assertTrue(all(np.isfinite(value) for value in health.values()))
        self.assertTrue(all(np.isfinite(value) for value in update.values()))


if __name__ == "__main__":
    unittest.main()
