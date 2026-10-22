"""Phase-5T checks. They do not replay scenarios or update a scorer."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from wt.lowlevel_v2.phase5q_encoder_scorer_audit.scorer import AssignmentConditionedSetScorer
from wt.lowlevel_v2.phase5t_scorer_failure_audit.metrics import choose_gate
from wt.lowlevel_v2.phase5t_scorer_failure_audit.protocol import assert_consumed_seed
from wt.lowlevel_v2.phase5t_scorer_failure_audit.represent import assignment_latents


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5t_scorer_failure_audit"


class Phase5TTests(unittest.TestCase):
    def test_consumed_seeds_only(self) -> None:
        assert_consumed_seed(310100)
        assert_consumed_seed(310117)
        with self.assertRaises(RuntimeError):
            assert_consumed_seed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed_seed(230000)

    def test_gate_letters(self) -> None:
        self.assertEqual(choose_gate(data=True, objective=False, representation=False, optimization=False, temporal=False)["gate"], "T-A")
        self.assertEqual(choose_gate(data=False, objective=True, representation=False, optimization=False, temporal=False)["gate"], "T-B")
        self.assertEqual(choose_gate(data=False, objective=False, representation=True, optimization=False, temporal=False)["gate"], "T-C")
        self.assertEqual(choose_gate(data=False, objective=False, representation=False, optimization=False, temporal=True)["gate"], "T-C")
        self.assertEqual(choose_gate(data=False, objective=False, representation=False, optimization=True, temporal=False)["gate"], "T-D")
        mixed = choose_gate(data=True, objective=True, representation=False, optimization=False, temporal=False)
        self.assertEqual(mixed["gate"], "T-E")
        self.assertEqual(mixed["priority"], ["OBJ", "DATA"])
        empty = choose_gate(data=False, objective=False, representation=False, optimization=False, temporal=False)
        self.assertEqual(empty["gate"], "T-E")
        self.assertFalse(empty["gate_condition_met"])

    def test_latent_matches_the_frozen_mlp(self) -> None:
        scorer = AssignmentConditionedSetScorer()
        scorer.eval()
        features = {
            "h_u": torch.randn(1, 6, scorer.width),
            "h_b": torch.randn(1, 12, scorer.width),
            "h_edge": torch.randn(1, 6, 12, scorer.width),
            "h_uu": torch.randn(1, 6, 6, scorer.width),
            "z_global": torch.randn(1, scorer.width),
        }
        assignments = torch.tensor([[0, 1, 2, -1, -1, -1], [3, 4, 5, -1, -1, -1]], dtype=torch.long)
        with torch.no_grad():
            expected = scorer(features, assignments)
            actual = scorer.mlp(assignment_latents(scorer, features, assignments)).squeeze(-1)
        self.assertTrue(torch.allclose(expected, actual, atol=1.0e-5))

    def test_package_does_not_train_or_call_decide(self) -> None:
        called = []
        for path in PACKAGE.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("Adam", source)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    called.append(node.func.attr)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    called.append(node.func.id)
        self.assertNotIn("decide", called)
        self.assertNotIn("step", called)
        for relative in (
            "src/wt/htw_phase100/high_level.py",
            "src/wt/lowlevel_v2/residual_policy.py",
            "src/wt/lowlevel_v2/phase5_hgnn_urgency/reward.py",
            "src/wt/lowlevel_v2/phase5_hgnn_urgency/planner.py",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("phase5t", text)


if __name__ == "__main__":
    unittest.main()
