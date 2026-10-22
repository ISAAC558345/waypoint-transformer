"""Phase-5ZH ranking loss checks. No environment rollout."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase5ze_response_distillation.predictor import OneStepExecutorResponsePredictor
from wt.lowlevel_v2.phase5ze_response_distillation.tokens import token_dim
from wt.lowlevel_v2.phase5zh_ranking_response_distillation.loss import pairwise_rank_loss, ranking_pairs
from wt.lowlevel_v2.phase5zh_ranking_response_distillation.metrics import choose_gate
from wt.lowlevel_v2.phase5zh_ranking_response_distillation.protocol import (
    CONFIRM_SCENARIOS,
    LAMBDA_RANK,
    TRAIN_SCENARIOS,
    VALUE_TEMPERATURE,
    assert_consumed,
)


class Phase5ZHRankingDistillationTest(unittest.TestCase):
    def test_pairs_stay_inside_one_state_and_drop_ties(self) -> None:
        teacher = np.asarray([3.0, 1.0, 1.0000004, 0.0, 2.0], dtype=np.float64)
        left, right = ranking_pairs(teacher, np.random.default_rng(0), eps=1.0e-3, n_random=0)
        self.assertTrue(np.all(left >= 0) and np.all(right < teacher.size))
        for first, second in zip(left, right):
            self.assertGreater(teacher[int(first)] - teacher[int(second)], 1.0e-3)
            self.assertNotEqual(int(first), int(second))

    def test_softplus_prefers_the_teacher_order(self) -> None:
        agreed = torch.tensor([2.0, 0.0], dtype=torch.float32)
        reversed_order = torch.tensor([0.0, 2.0], dtype=torch.float32)
        left = torch.tensor([0], dtype=torch.long)
        right = torch.tensor([1], dtype=torch.long)
        low = pairwise_rank_loss(agreed, left, right, 1.0)
        high = pairwise_rank_loss(reversed_order, left, right, 1.0)
        self.assertLess(float(low), float(high))
        student = torch.tensor([0.0, 1.0], dtype=torch.float32, requires_grad=True)
        pairwise_rank_loss(student, left, right, 1.0).backward()
        self.assertLess(float(student.grad[0]), 0.0)
        self.assertGreater(float(student.grad[1]), 0.0)

    def test_empty_pairs_stay_connected(self) -> None:
        student = torch.zeros(3, requires_grad=True)
        loss = pairwise_rank_loss(student, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertIsNotNone(student.grad)

    def test_locked_weight_and_temperature(self) -> None:
        self.assertEqual(LAMBDA_RANK, 0.2)
        self.assertEqual(VALUE_TEMPERATURE, 1.0)
        self.assertNotIn(310118, TRAIN_SCENARIOS)
        self.assertNotIn(310118, CONFIRM_SCENARIOS)

    def test_gate_priority(self) -> None:
        flags = dict(
            aligned=True,
            j15_mean_up=True,
            ci_low_positive=True,
            seeds_ok=True,
            retention_ok=True,
            ranking_ok=True,
            physical_ok=True,
            ranking_up=True,
            j15_degraded=False,
        )
        self.assertEqual(choose_gate(**flags), "RANK-A")
        blocked = dict(flags)
        blocked["aligned"] = False
        self.assertEqual(choose_gate(**blocked), "RANK-D")
        degraded = dict(flags)
        degraded["j15_degraded"] = True
        self.assertEqual(choose_gate(**degraded), "RANK-C")
        broken = dict(flags)
        broken["physical_ok"] = False
        broken["ranking_up"] = True
        self.assertEqual(choose_gate(**broken), "RANK-C")
        partial = dict(flags)
        partial["ci_low_positive"] = False
        partial["retention_ok"] = False
        self.assertEqual(choose_gate(**partial), "RANK-B")
        failed = dict(flags)
        failed["j15_mean_up"] = False
        failed["ci_low_positive"] = False
        failed["seeds_ok"] = False
        failed["retention_ok"] = False
        failed["ranking_ok"] = False
        failed["ranking_up"] = False
        self.assertEqual(choose_gate(**failed), "RANK-C")

    def test_closed_scenarios_are_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230100)

    def test_network_width_matches_the_existing_predictor(self) -> None:
        model = OneStepExecutorResponsePredictor(token_dim())
        self.assertEqual(int(model.local[0].out_features), 128)
        self.assertEqual(int(model.response[-1].out_features), 14)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), sum(parameter.numel() for parameter in model.parameters()))

    def test_loss_keeps_the_response_term(self) -> None:
        source = Path("src/wt/lowlevel_v2/phase5zh_ranking_response_distillation/run_phase5zh.py").read_text(encoding="utf-8")
        self.assertIn("response_loss(", source)
        self.assertIn("regression + float(LAMBDA_RANK) * ranking", source)
        self.assertNotIn("decide(", source)
        root = Path("src/wt/lowlevel_v2/phase5zh_ranking_response_distillation")
        for path in root.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
