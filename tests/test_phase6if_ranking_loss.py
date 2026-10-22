"""Loss identities on a tiny candidate set. Full training is not run here."""
from __future__ import annotations

import unittest

import torch

from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import DecisionDataset
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.benchmark_v2.phase6if_ranking_loss.losses import gap_weighted_loss, tolerance_margin_loss, top_regret_loss


class Phase6IFLossTests(unittest.TestCase):
    def test_reserved_scenarios_are_outside_both_splits(self) -> None:
        seen = {int(scenario) for scenario, _time in DecisionDataset("train").index + DecisionDataset("validation").index}
        self.assertTrue(seen.isdisjoint(RESERVED_TEST))

    def test_equal_labels_do_not_create_pairwise_penalties(self) -> None:
        scores = torch.tensor([0.2, -0.4, 0.1], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)
        weighted = gap_weighted_loss(scores, labels)
        margin = tolerance_margin_loss(scores, labels)
        self.assertEqual(float(weighted), 0.0)
        self.assertEqual(float(margin), 0.0)

    def test_small_gap_is_ignored_by_the_tolerance(self) -> None:
        scores = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0.0, 100.0], dtype=torch.float64)
        loss = tolerance_margin_loss(scores, labels)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.all(scores.grad == 0))

    def test_large_gap_pushes_the_better_candidate_up(self) -> None:
        scores = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0.0, 2.0e6], dtype=torch.float64)
        loss = gap_weighted_loss(scores, labels)
        loss.backward()
        self.assertGreater(float(scores.grad[0]), 0.0)
        self.assertLess(float(scores.grad[1]), 0.0)

    def test_target_loss_does_not_require_an_oracle_index(self) -> None:
        scores = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0.0, 1.0e6, 2.0e6], dtype=torch.float64)
        loss = top_regret_loss(scores, labels)
        loss.backward()
        self.assertLess(float(scores.grad[2]), float(scores.grad[1]))
        self.assertLess(float(scores.grad[1]), float(scores.grad[0]))


if __name__ == "__main__":
    unittest.main()
