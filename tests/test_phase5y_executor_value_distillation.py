"""Phase-5Y checks the value loss and the gate. They do not train."""
from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

import torch

from wt.lowlevel_v2.phase5y_executor_value_distillation.metrics import choose_gate
from wt.lowlevel_v2.phase5y_executor_value_distillation.protocol import assert_consumed
from wt.lowlevel_v2.phase5y_executor_value_distillation.run_phase5y import _train_one
from wt.lowlevel_v2.phase5y_executor_value_distillation.value import ExecutorConditionedJointValue, state_huber


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5y_executor_value_distillation"


class Phase5YTests(unittest.TestCase):
    def test_state_huber_uses_one_block_delta(self) -> None:
        prediction = torch.tensor([2.0, 0.0])
        target = torch.tensor([0.0, 0.0])
        # |2| > 1 gives delta*(|e|-delta/2)=1.5; the zero error contributes 0.
        self.assertAlmostEqual(float(state_huber(prediction, target)), 0.75, places=6)
        assert_consumed(310109)
        assert_consumed(310117)
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230000)

    def test_gate_letters(self) -> None:
        self.assertEqual(choose_gate(confirmed=True, promising=True, regression_without_decision=True, representation_limit=True), "Y-A")
        self.assertEqual(choose_gate(confirmed=False, promising=False, regression_without_decision=False, representation_limit=True), "Y-D")
        self.assertEqual(choose_gate(confirmed=False, promising=False, regression_without_decision=True, representation_limit=False), "Y-C")
        self.assertEqual(choose_gate(confirmed=False, promising=True, regression_without_decision=False, representation_limit=False), "Y-B")
        self.assertEqual(choose_gate(confirmed=False, promising=False, regression_without_decision=False, representation_limit=False), "Y-E")

    def test_training_does_not_read_rollout_statistics(self) -> None:
        source = inspect.getsource(_train_one)
        self.assertNotIn("q_exec", source)
        self.assertNotIn("mean_sinr", source)
        self.assertNotIn("payload_h", source)
        self.assertNotIn("joint_rank_loss", source)
        module = ExecutorConditionedJointValue()
        names = dict(module.named_parameters())
        self.assertTrue(any(name.startswith("interact.") for name in names))
        self.assertIn("mlp.0.weight", names)
        called = []
        for path in PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    called.append(node.func.attr)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    called.append(node.func.id)
        self.assertNotIn("decide", called)
        self.assertIn("Adam", called)
        for relative in (
            "src/wt/htw_phase100/high_level.py",
            "src/wt/lowlevel_v2/residual_policy.py",
            "src/wt/lowlevel_v2/phase5_hgnn_urgency/reward.py",
            "src/wt/lowlevel_v2/phase5_hgnn_urgency/planner.py",
        ):
            self.assertNotIn("phase5y", (ROOT / relative).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
