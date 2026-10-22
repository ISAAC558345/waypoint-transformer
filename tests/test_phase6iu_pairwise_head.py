"""Pairwise interaction head checks that do not open reserved scenarios."""
from __future__ import annotations

import unittest

import torch

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, TRAIN_POOL, VALIDATION_POOL
from wt.benchmark_v2.phase6is_assignment_head.loss import pairwise_rank_loss
from wt.benchmark_v2.phase6iu_pairwise_head.head import PairwiseInteractionHead, symmetric_pair_features
from wt.benchmark_v2.phase6iu_pairwise_head.protocol import LAMBDA_PAIR, SEEDS, assign_gate, seed_supported
from wt.lowlevel_v2.phase5k_joint_decoder.loss import joint_rank_loss


class Phase6IUPairwiseHeadTests(unittest.TestCase):
    def test_pools_and_gate_are_fixed(self) -> None:
        self.assertTrue(set(TRAIN_POOL).isdisjoint(set(VALIDATION_POOL)))
        self.assertTrue(set(TRAIN_POOL).isdisjoint(RESERVED_TEST))
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))
        self.assertEqual(len(SEEDS), 5)
        self.assertEqual(float(LAMBDA_PAIR), 1.0)
        passed = {
            "top1_above_scheme_c": True,
            "regret_not_worse": True,
            "mean_ci_above_scheme_c": True,
            "scenarios_consistent": True,
            "gain_not_concentrated": True,
            "interaction_active": True,
            "interaction_sign": True,
            "quadruple_improves": True,
            "supported": True,
        }
        self.assertTrue(seed_supported(passed))
        self.assertEqual(assign_gate(False, True, [passed] * 5), "PAIRWISE_INTERACTION_GAIN_SUPPORTED")
        failed = dict(passed)
        failed["top1_above_scheme_c"] = False
        failed["supported"] = False
        self.assertEqual(assign_gate(False, True, [failed] * 5), "PAIRWISE_INTERACTION_GAIN_INCONCLUSIVE")
        self.assertEqual(assign_gate(True, True, [passed] * 5), "PAIRWISE_INTERACTION_IMPLEMENTATION_BLOCKED")

    def test_features_are_symmetric_and_use_both_edges(self) -> None:
        left = torch.tensor([[1.0, -2.0, 0.5]])
        right = torch.tensor([[0.25, 2.0, -0.5]])
        other = torch.tensor([[4.0, 0.0, 1.0]])
        forward = symmetric_pair_features(left, right)
        backward = symmetric_pair_features(right, left)
        self.assertLess(float((forward - backward).abs().max()), 1.0e-6)
        self.assertGreater(float((forward - symmetric_pair_features(left, other)).abs().sum()), 0.0)
        self.assertGreater(float((forward - symmetric_pair_features(other, right)).abs().sum()), 0.0)

    def test_identical_rows_permutation_idle_and_chunks_agree(self) -> None:
        torch.manual_seed(0)
        head = PairwiseInteractionHead(chunk=3).eval()
        edge_h = torch.randn(6, 12, 128)
        assignments = torch.tensor(
            [[6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11], [11, 10, 9, 8, 7, 6], [6, 7, 8, 9, 10, -1]],
            dtype=torch.long,
        )
        unary = torch.zeros(4)
        with torch.no_grad():
            scores = head(edge_h, assignments, unary)
            perm = torch.tensor([2, 0, 1, 4, 3, 5])
            permuted = head(edge_h[perm], assignments[:, perm], unary)
            wide = PairwiseInteractionHead(chunk=100).eval()
            wide.load_state_dict(head.state_dict())
            whole = wide(edge_h, assignments, unary)
        self.assertLess(abs(float(scores[0] - scores[1])), 1.0e-6)
        self.assertLess(float((scores - permuted).abs().max()), 1.0e-5)
        self.assertLess(float((scores - whole).abs().max()), 1.0e-5)
        noisy = edge_h.clone()
        noisy[5] = 50.0
        with torch.no_grad():
            idle_changed = head(noisy, assignments, unary)
        self.assertLess(abs(float(scores[3] - idle_changed[3])), 1.0e-5)

    def test_disjoint_pair_residual_is_not_additive(self) -> None:
        torch.manual_seed(2)
        head = PairwiseInteractionHead().eval()
        edge_h = torch.randn(6, 12, 128)
        rows = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [1, 0, 2, 3, 4, 5], [0, 1, 3, 2, 4, 5], [1, 0, 3, 2, 4, 5]],
            dtype=torch.long,
        )
        with torch.no_grad():
            scores = head.pair_score(edge_h, rows)
        residual = float(scores[3] - scores[0] - (scores[1] - scores[0]) - (scores[2] - scores[0]))
        self.assertGreater(abs(residual), 1.0e-4)

    def test_pairwise_loss_and_interaction_gradient(self) -> None:
        torch.manual_seed(3)
        head = PairwiseInteractionHead(chunk=4)
        edge_h = torch.randn(6, 12, 128)
        assignments = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 2, 3, 4, 5], [2, 1, 0, 3, 4, 5], [0, 2, 1, 3, 4, 5]])
        unary = torch.tensor([0.2, -0.1, 0.4, 0.0])
        labels = torch.tensor([3.0, 1.0, 8.0, 1.0])
        scores = head(edge_h, assignments, unary)
        loss = pairwise_rank_loss(scores, labels, chunk=2)
        reference = joint_rank_loss(scores.detach().clone().requires_grad_(True), labels)
        self.assertLess(abs(float(loss.detach()) - float(reference.detach())), 1.0e-6)
        loss.backward()
        grad_norm = torch.linalg.vector_norm(torch.cat([parameter.grad.detach().reshape(-1) for parameter in head.parameters()]))
        self.assertTrue(bool(torch.isfinite(grad_norm)))
        self.assertGreater(float(grad_norm), 0.0)
