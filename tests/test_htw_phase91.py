"""Phase-9.1 no-idle encoder checks. No full 300-update training."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from wt.htw_phase64.run_phase64 import high_config
from wt.htw_phase9.agent import GraphDiscretePPOAgent
from wt.htw_phase9.const import indices_to_assignment
from wt.htw_phase9.features import build_graph, feature_dict_l3b, stack_feature_dicts
from wt.htw_phase91.models import OPTION_DIM, HeteroOptionHGNN, MLPOptionModel
from wt.l3b.env import L3BEnv


class Phase91Tests(unittest.TestCase):
    def test_01_no_idle_option_dim(self) -> None:
        self.assertEqual(OPTION_DIM, 12)

    def test_02_assignment_never_idle(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=0)
        agent = GraphDiscretePPOAgent(MLPOptionModel(), high_config())
        feats = feature_dict_l3b(info, env)
        batch = stack_feature_dicts([feats])
        edge, edge_uu = build_graph(info, env)
        batch["edge"] = edge[None, ...]
        batch["edge_uu"] = edge_uu[None, ...]
        z = agent.act(batch, deterministic=True)["z"][0]
        self.assertEqual(int(z.shape[-1]), 6)
        self.assertTrue(bool(np.all((z >= 0) & (z < 12))))
        assignment = indices_to_assignment(z, 6)
        self.assertTrue(bool(np.all(assignment >= 0)))

    def test_03_hgnn_logits_are_buoy_only(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=0)
        feats = feature_dict_l3b(info, env)
        batch = stack_feature_dicts([feats])
        edge, edge_uu = build_graph(info, env)
        model = HeteroOptionHGNN()
        logits, value = model(
            torch.as_tensor(batch["uav"]),
            torch.as_tensor(batch["buoy"]),
            torch.as_tensor(batch["uav_mask"]).bool(),
            torch.as_tensor(batch["buoy_mask"]).bool(),
            torch.as_tensor(batch["global"]),
            torch.as_tensor(edge[None, ...]),
            torch.as_tensor(edge_uu[None, ...]),
        )
        self.assertEqual(tuple(logits.shape), (1, 6, 12))
        self.assertEqual(tuple(value.shape), (1,))


if __name__ == "__main__":
    unittest.main()
