"""Phase-5Z stays an audit: the frozen pool is read, not retrained."""
from __future__ import annotations

import unittest
from pathlib import Path

import torch

from wt.lowlevel_v2.phase5t_scorer_failure_audit.represent import assignment_latents
from wt.lowlevel_v2.phase5y_executor_value_distillation.value import ExecutorConditionedJointValue
from wt.lowlevel_v2.phase5z_representation_collision_audit.layers import layer_tensors
from wt.lowlevel_v2.phase5z_representation_collision_audit.metrics import choose_gate
from wt.lowlevel_v2.phase5z_representation_collision_audit.protocol import assert_consumed


class Phase5ZRepresentationCollisionAuditTest(unittest.TestCase):
    def test_prepool_keeps_correspondence_that_mean_pooling_mixes(self) -> None:
        torch.manual_seed(0)
        scorer = ExecutorConditionedJointValue()
        width = 128
        features = {
            "h_u": torch.randn(1, 6, width),
            "h_b": torch.randn(1, 12, width),
            "h_edge": torch.randn(1, 6, 12, width),
            "h_uu": torch.randn(1, 6, 6, width),
            "z_global": torch.randn(width),
        }
        assignments = torch.tensor(
            [
                [2, 5, -1, 4, -1, -1],
                [5, 2, -1, 4, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.long,
        )
        layers = layer_tensors(scorer, features, assignments)
        self.assertTrue(torch.allclose(layers["postpool"], assignment_latents(scorer, features, assignments), atol=1.0e-5))
        self.assertFalse(torch.allclose(layers["prepool"][0], layers["prepool"][1]))
        pooled = layers["prepool"][:, :2].sum(dim=1)
        self.assertTrue(torch.allclose(pooled[0, : 2 * width], pooled[1, : 2 * width], atol=1.0e-5))
        self.assertFalse(torch.allclose(pooled[0, 2 * width :], pooled[1, 2 * width :]))

    def test_gate_priority_is_locked(self) -> None:
        self.assertEqual(choose_gate(head=True, pooling=True, pair=True, phy=True, temp=True, state_insufficient=True), "Z-E")
        self.assertEqual(choose_gate(head=False, pooling=True, pair=True, phy=True, temp=True, state_insufficient=True), "Z-A")
        self.assertEqual(choose_gate(head=False, pooling=False, pair=True, phy=True, temp=True, state_insufficient=True), "Z-B")
        self.assertEqual(choose_gate(head=False, pooling=False, pair=False, phy=True, temp=True, state_insufficient=True), "Z-C")
        self.assertEqual(choose_gate(head=False, pooling=False, pair=False, phy=False, temp=True, state_insufficient=True), "Z-D")
        self.assertEqual(choose_gate(head=False, pooling=False, pair=False, phy=False, temp=False, state_insufficient=True), "Z-F")
        self.assertEqual(choose_gate(head=False, pooling=False, pair=False, phy=False, temp=False, state_insufficient=False), "Z-F")

    def test_holdout_seeds_are_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230100)

    def test_package_does_not_train_or_decide(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "lowlevel_v2" / "phase5z_representation_collision_audit"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("torch.optim", text)
        self.assertNotIn(".backward(", text)
        self.assertNotIn("decide(", text)


if __name__ == "__main__":
    unittest.main()
