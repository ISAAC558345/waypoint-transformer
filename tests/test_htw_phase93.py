"""Phase-9.3 relational scoring and ranking loss checks."""
from __future__ import annotations

import unittest

import torch

from wt.htw_phase93.losses import ranking_loss
from wt.htw_phase93.models import RelationalHGNNAssignmentHead
from wt.htw_phase9.features import build_graph, feature_dict_l3b, stack_feature_dicts
from wt.l3b.env import L3BEnv


class Phase93Tests(unittest.TestCase):
    def test_01_ranking_prefers_positive(self) -> None:
        scores = torch.tensor([[[2.0, 0.0, -1.0]]])
        idle = torch.tensor([[-2.0]])
        assignment = torch.tensor([[0]])
        uav_mask = torch.tensor([[True]])
        buoy_mask = torch.tensor([[True, True, True]])
        loss, acc = ranking_loss(scores, idle, assignment, uav_mask, buoy_mask)
        self.assertGreater(float(acc), 0.99)
        self.assertGreater(float(loss), 0.0)

    def test_02_relational_score_uses_graph_context(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=0)
        feats = feature_dict_l3b(info, env)
        batch = stack_feature_dicts([feats])
        edge, edge_uu = build_graph(info, env)
        model = RelationalHGNNAssignmentHead()
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
        self.assertGreater(int(model.pair_score[0].in_features), 3 * 128)


if __name__ == "__main__":
    unittest.main()
