"""Phase-5Q checks. They do not roll out scenarios or call decide()."""
from __future__ import annotations

import ast
import unittest
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import (
    choose_gate,
    design_matrix,
    fit_linear,
    pairwise_beats_additive,
    regression_metrics,
    stable_improvement,
    train_fit,
)
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.protocol import assert_phase5q_seed
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.scorer import AssignmentConditionedSetScorer


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5q_encoder_scorer_audit"


class Phase5QTests(unittest.TestCase):
    def test_holdout_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_phase5q_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_phase5q_seed(310102)
        assert_phase5q_seed(310101)

    def test_gate_covers_the_five_outcomes(self) -> None:
        self.assertEqual(
            choose_gate(b_dev_improved=True, c_dev_improved=False, b_train_fit=True, c_train_fit=False, pairwise_better=False),
            "Q-A",
        )
        self.assertEqual(
            choose_gate(b_dev_improved=False, c_dev_improved=True, b_train_fit=False, c_train_fit=True, pairwise_better=True),
            "Q-B",
        )
        self.assertEqual(
            choose_gate(b_dev_improved=True, c_dev_improved=True, b_train_fit=True, c_train_fit=True, pairwise_better=True),
            "Q-C",
        )
        self.assertEqual(
            choose_gate(b_dev_improved=False, c_dev_improved=False, b_train_fit=True, c_train_fit=True, pairwise_better=False),
            "Q-D",
        )
        self.assertEqual(
            choose_gate(b_dev_improved=False, c_dev_improved=False, b_train_fit=False, c_train_fit=False, pairwise_better=False),
            "Q-E",
        )

    def test_stable_improvement_needs_one_payload_block(self) -> None:
        self.assertFalse(train_fit(first_loss=0.6931, final_loss=0.6930, collapsed_fraction=1.0))
        self.assertTrue(train_fit(first_loss=0.6931, final_loss=0.2, collapsed_fraction=0.0))
        self.assertFalse(stable_improvement([1.0e5] * 5, [-0.2] * 5))
        self.assertTrue(stable_improvement([1.0e6] * 5, [-0.2] * 5))
        self.assertFalse(stable_improvement([1.0e6, 1.0e6, -1.0, -1.0, -1.0], [-0.2, -0.2, 0.1, 0.1, 0.1]))

    def test_pairwise_design_explains_a_joint_indicator(self) -> None:
        rows = []
        for chosen in combinations(range(6), 3):
            for perm in permutations((3, 4, 5)):
                row = [-1] * 6
                for uav, buoy in zip(chosen, perm):
                    row[uav] = buoy
                rows.append(row)
        assignments = np.asarray(rows, dtype=np.int64)
        labels = np.where((assignments[:, 0] >= 0) & (assignments[:, 1] >= 0), 1.0e6, 0.0)
        targets = [3, 4, 5]
        additive = design_matrix(assignments, targets, pairwise=False)
        paired = design_matrix(assignments, targets, pairwise=True)
        additive_pred, _, _ = fit_linear(additive, labels, additive)
        paired_pred, _, _ = fit_linear(paired, labels, paired)
        additive_metrics = regression_metrics(additive_pred, labels)
        paired_metrics = regression_metrics(paired_pred, labels)
        self.assertGreater(paired_metrics["r2"], 0.99)
        self.assertLess(additive_metrics["r2"], paired_metrics["r2"] - 0.05)
        self.assertTrue(
            pairwise_beats_additive(
                {"r2": 0.2, "spearman": 0.1, "top1": 0.0, "normalized_regret": 0.8},
                {"r2": 0.8, "spearman": 0.7, "top1": 1.0, "normalized_regret": 0.2},
            )
        )
        self.assertFalse(
            pairwise_beats_additive(
                {"r2": 0.2, "spearman": 0.1, "top1": 0.0, "normalized_regret": 0.2},
                {"r2": 0.9, "spearman": 0.8, "top1": 1.0, "normalized_regret": 0.5},
            )
        )

    def test_scorer_reads_selected_interactions_only(self) -> None:
        torch.manual_seed(0)
        scorer = AssignmentConditionedSetScorer(width=8)
        with torch.no_grad():
            for module in scorer.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.fill_(0.01)
                    module.bias.zero_()
        features = {
            "h_u": torch.randn(1, 6, 8),
            "h_b": torch.randn(1, 12, 8),
            "h_edge": torch.randn(1, 6, 12, 8),
            "h_uu": torch.randn(1, 6, 6, 8),
            "z_global": torch.randn(1, 8),
        }
        assignments = torch.tensor([[3, 4, -1, -1, -1, 5]], dtype=torch.long)
        baseline = scorer(features, assignments)
        untouched = {key: value.clone() for key, value in features.items()}
        untouched["h_edge"][0, 2, 7] += 4.0
        self.assertTrue(torch.allclose(baseline, scorer(untouched, assignments)))
        shifted = {key: value.clone() for key, value in features.items()}
        shifted["h_uu"][0, 0, 1] += 4.0
        gap = torch.max(torch.abs(scorer(shifted, assignments) - baseline)).detach()
        self.assertGreater(float(gap), 1.0e-6)

    def test_package_does_not_call_decide_or_edit_frozen_files(self) -> None:
        called = []
        for path in PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    called.append(node.func.attr)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    called.append(node.func.id)
        self.assertNotIn("decide", called)
        for relative in (
            "src/wt/htw_phase100/high_level.py",
            "src/wt/lowlevel_v2/residual_policy.py",
            "src/wt/lowlevel_v2/phase5_hgnn_urgency/reward.py",
            "src/wt/lowlevel_v2/phase5_hgnn_urgency/planner.py",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("phase5q", text)


if __name__ == "__main__":
    unittest.main()
