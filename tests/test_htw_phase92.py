"""Phase-9.2 matching-head checks. No PPO."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from wt.htw_phase92.matching import hungarian_maximize, match_uav_buoy
from wt.htw_phase92.models import HGNNAssignmentHead, MLPOptionClassifier
from wt.htw_phase9.features import build_graph, feature_dict_l3b, stack_feature_dicts
from wt.l3b.env import L3BEnv


class Phase92Tests(unittest.TestCase):
    def test_01_hungarian_exclusive(self) -> None:
        scores = np.array(((1.0, 3.0), (4.0, 1.0)), dtype=np.float64)
        cols = hungarian_maximize(scores)
        self.assertEqual(int(cols[0]), 1)
        self.assertEqual(int(cols[1]), 0)

    def test_02_match_capacity_one_and_idle(self) -> None:
        buoy_scores = np.array(((10.0, 9.0, 0.0), (9.5, 8.0, 0.0), (1.0, 1.0, 0.0)), dtype=np.float64)
        idle = np.array((0.0, 0.0, 5.0), dtype=np.float64)
        mask_k = np.array((True, True, True))
        mask_m = np.array((True, True, True))
        assignment = match_uav_buoy(buoy_scores, idle, mask_k, mask_m)
        chosen = [int(item) for item in assignment if int(item) >= 0]
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertEqual(int(assignment[2]), -1)

    def test_03_hgnn_outputs_edge_scores(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=0)
        feats = feature_dict_l3b(info, env)
        batch = stack_feature_dicts([feats])
        edge, edge_uu = build_graph(info, env)
        model = HGNNAssignmentHead()
        scores, idle = model(
            torch.as_tensor(batch["uav"]),
            torch.as_tensor(batch["buoy"]),
            torch.as_tensor(batch["uav_mask"]).bool(),
            torch.as_tensor(batch["buoy_mask"]).bool(),
            torch.as_tensor(batch["global"]),
            torch.as_tensor(edge[None, ...]),
            torch.as_tensor(edge_uu[None, ...]),
        )
        self.assertEqual(tuple(scores.shape), (1, 6, 12))
        self.assertEqual(tuple(idle.shape), (1, 6))
        self.assertFalse(hasattr(model, "actor"))
        mlp = MLPOptionClassifier()
        self.assertTrue(hasattr(mlp, "actor"))


if __name__ == "__main__":
    unittest.main()
