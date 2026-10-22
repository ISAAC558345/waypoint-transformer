"""Phase-5ZE predictor and gate checks. No environment rollout."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase5ze_response_distillation.metrics import choose_gate, retention_ratio
from wt.lowlevel_v2.phase5ze_response_distillation.predictor import (
    OneStepExecutorResponsePredictor,
    response_loss,
)
from wt.lowlevel_v2.phase5ze_response_distillation.protocol import BINARY_FEATURES, RESPONSE_FEATURES, assert_consumed
from wt.lowlevel_v2.phase5ze_response_distillation.tokens import build_tokens, token_dim


class Phase5ZEResponseDistillationTest(unittest.TestCase):
    def test_predictor_is_uav_equivariant(self) -> None:
        torch.manual_seed(0)
        model = OneStepExecutorResponsePredictor(token_dim())
        model.eval()
        tokens = torch.randn(4, 6, token_dim())
        perm = torch.tensor([2, 0, 4, 1, 5, 3])
        with torch.no_grad():
            original = model(tokens)
            moved = model(tokens[:, perm, :])
        self.assertLess(float(torch.max(torch.abs(moved - original[:, perm, :]))), 1.0e-5)

    def test_hidden_width_is_fixed(self) -> None:
        model = OneStepExecutorResponsePredictor(token_dim())
        self.assertEqual(int(model.local[0].out_features), 128)
        self.assertEqual(int(model.response[0].in_features), 256)
        self.assertEqual(int(model.response[-1].out_features), len(RESPONSE_FEATURES))

    def test_loss_uses_huber_and_bce_without_j15(self) -> None:
        source = Path("src/wt/lowlevel_v2/phase5ze_response_distillation/predictor.py").read_text(encoding="utf-8")
        self.assertNotIn("j_mappo", source)
        self.assertNotIn("j15", source.lower())
        prediction = torch.zeros(2, 6, len(RESPONSE_FEATURES))
        target = torch.zeros_like(prediction)
        binary = torch.zeros(2, 6, len(BINARY_FEATURES))
        loss, huber, bce = response_loss(prediction, target, binary)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(huber))
        self.assertTrue(torch.isfinite(bce))

    def test_token_construction_follows_uav_permutation(self) -> None:
        uavs, buoys, width = 6, 4, 128
        h_u = np.random.default_rng(0).normal(size=(uavs, width))
        h_b = np.random.default_rng(1).normal(size=(buoys, width))
        h_edge = np.random.default_rng(2).normal(size=(uavs, buoys, width))
        rep = np.random.default_rng(3).normal(size=(2, uavs, 12))
        temp = np.random.default_rng(4).normal(size=(2, uavs, 9))
        assignments = np.asarray([[0, -1, 1, 2, -1, 3], [3, 2, -1, -1, 1, 0]], dtype=np.int64)
        actions = np.random.default_rng(5).normal(size=(2, uavs, 2))
        perm = np.asarray([2, 0, 4, 1, 5, 3], dtype=np.int64)
        original = build_tokens(h_u, h_b, h_edge, rep, temp, assignments, actions)
        moved = build_tokens(h_u[perm], h_b, h_edge[perm], rep[:, perm], temp[:, perm], assignments[:, perm], actions[:, perm])
        self.assertLess(float(np.max(np.abs(moved - original[:, perm, :]))), 1.0e-6)

    def test_gate_priority(self) -> None:
        self.assertEqual(choose_gate(r1_reproduced=False, confirmed=True, mean_improved=True), "DISTILL-D")
        self.assertEqual(choose_gate(r1_reproduced=True, confirmed=True, mean_improved=True), "DISTILL-A")
        self.assertEqual(choose_gate(r1_reproduced=True, confirmed=False, mean_improved=True), "DISTILL-B")
        self.assertEqual(choose_gate(r1_reproduced=True, confirmed=False, mean_improved=False), "DISTILL-C")

    def test_retention_skips_zero_true_gain(self) -> None:
        self.assertTrue(np.isnan(retention_ratio(10.0, 0.0, 1.0)))
        self.assertAlmostEqual(retention_ratio(3.0, 4.0, 1.0), 0.75)

    def test_closed_scenarios_are_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230100)

    def test_package_does_not_call_deployment_decide(self) -> None:
        root = Path("src/wt/lowlevel_v2/phase5ze_response_distillation")
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("decide(", text)
        self.assertIn("Adam", text)


if __name__ == "__main__":
    unittest.main()
