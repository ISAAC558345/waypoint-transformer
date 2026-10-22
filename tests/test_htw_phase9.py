"""Phase-9 encoder checks. Does not train the full budget in CI."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from wt.htw_phase9.agent import GraphDiscretePPOAgent
from wt.htw_phase9.const import MAX_UAVS, OPTION_DIM, indices_to_assignment
from wt.htw_phase9.features import build_graph, feature_dict_l3b, stack_feature_dicts
from wt.htw_phase9.models import DeepSetsOptionModel, HeteroOptionHGNN, MLPOptionModel
from wt.htw_phase64.run_phase64 import high_config
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward


class Phase9Tests(unittest.TestCase):
    def test_01_feature_and_graph_shapes(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=0)
        feats = feature_dict_l3b(info, env)
        self.assertEqual(feats["uav"].shape, (6, 8))
        self.assertEqual(float(feats["uav"][0, 7]), 0.0)
        edge, edge_uu = build_graph(info, env)
        self.assertEqual(edge.shape[-1], 3)
        self.assertEqual(edge_uu.shape[-1], 2)

    def test_02_no_exclusive_projection(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=0)
        model = MLPOptionModel()
        agent = GraphDiscretePPOAgent(model, high_config())
        feats = feature_dict_l3b(info, env)
        batch = stack_feature_dicts([feats])
        edge, edge_uu = build_graph(info, env)
        batch["edge"] = edge[None, ...]
        batch["edge_uu"] = edge_uu[None, ...]
        z = agent.act(batch, deterministic=True)["z"][0]
        assignment = indices_to_assignment(z, 6)
        self.assertEqual(assignment.shape[0], 6)
        piled = np.zeros(6, dtype=np.int64)
        after = env.step(fly_toward(env, piled), piled)
        self.assertGreater(int(after["n_conflict_pairs"]), 0)

    def test_03_models_forward(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=0)
        feats = feature_dict_l3b(info, env)
        batch = stack_feature_dicts([feats])
        edge, edge_uu = build_graph(info, env)
        tensors = {key: torch.as_tensor(batch[key]) for key in batch}
        tensors["uav_mask"] = tensors["uav_mask"].bool()
        tensors["buoy_mask"] = tensors["buoy_mask"].bool()
        edge_t = torch.as_tensor(edge[None, ...])
        uu_t = torch.as_tensor(edge_uu[None, ...])
        for model in (MLPOptionModel(), DeepSetsOptionModel(), HeteroOptionHGNN()):
            logits, value = model(tensors["uav"], tensors["buoy"], tensors["uav_mask"], tensors["buoy_mask"], tensors["global"], edge_t, uu_t)
            self.assertEqual(tuple(logits.shape), (1, MAX_UAVS, OPTION_DIM))
            self.assertEqual(tuple(value.shape), (1,))


if __name__ == "__main__":
    unittest.main()
