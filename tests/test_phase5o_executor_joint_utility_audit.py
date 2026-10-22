"""Phase-5O checks. They do not roll out scenarios or train a model."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.phase5o_executor_joint_utility_audit.metrics import (
    choose_label,
    discrimination,
    quantize_block,
    top1_overlap,
)
from wt.lowlevel_v2.phase5o_executor_joint_utility_audit.protocol import PAYLOAD_BLOCK_BITS, assert_phase5o_seed


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5o_executor_joint_utility_audit"


class Phase5OTests(unittest.TestCase):
    def test_holdout_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_phase5o_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_phase5o_seed(310102)
        assert_phase5o_seed(310100)
        assert_phase5o_seed(310101)

    def test_sub_block_gaps_are_one_level(self) -> None:
        values = np.array([0.0, 10.0, PAYLOAD_BLOCK_BITS, PAYLOAD_BLOCK_BITS + 2.0])
        levels = quantize_block(values)
        self.assertEqual(len(np.unique(levels)), 2)
        stats = discrimination(np.zeros(8))
        self.assertEqual(stats["discriminates"], 0.0)
        varied = discrimination(np.array([0.0, 0.0, PAYLOAD_BLOCK_BITS * 3.0]))
        self.assertEqual(varied["n_unique"], 2.0)
        self.assertEqual(varied["discriminates"], 1.0)

    def test_top1_uses_the_stable_fly_maximizer(self) -> None:
        fly = np.array([0.0, 0.0, 0.0])
        executed = np.array([0.0, PAYLOAD_BLOCK_BITS, 0.0])
        self.assertEqual(top1_overlap(fly, executed), 0.0)
        self.assertEqual(top1_overlap(executed, executed), 1.0)

    def test_label_rule_covers_the_four_conclusions(self) -> None:
        self.assertEqual(
            choose_label(
                n_states=6,
                n_discriminating=1,
                pooled_spearman=0.0,
                mean_normalized_regret=1.0,
                mean_top1=0.0,
                n_outside_beats_pool=0,
            ),
            "LABEL-B",
        )
        self.assertEqual(
            choose_label(
                n_states=6,
                n_discriminating=4,
                pooled_spearman=0.9,
                mean_normalized_regret=0.1,
                mean_top1=1.0,
                n_outside_beats_pool=4,
            ),
            "LABEL-D",
        )
        self.assertEqual(
            choose_label(
                n_states=6,
                n_discriminating=4,
                pooled_spearman=0.1,
                mean_normalized_regret=0.8,
                mean_top1=0.0,
                n_outside_beats_pool=0,
            ),
            "LABEL-A",
        )
        self.assertEqual(
            choose_label(
                n_states=6,
                n_discriminating=5,
                pooled_spearman=0.9,
                mean_normalized_regret=0.1,
                mean_top1=0.8,
                n_outside_beats_pool=1,
            ),
            "LABEL-C",
        )

    def test_package_does_not_call_decide_or_train(self) -> None:
        called = []
        for path in PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    called.append(node.func.attr)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    called.append(node.func.id)
        self.assertNotIn("decide", called)
        self.assertNotIn("backward", called)
        for relative in (
            "src/wt/htw_phase100/high_level.py",
            "src/wt/lowlevel_v2/residual_policy.py",
            "src/wt/lowlevel_v2/phase5_hgnn_urgency/reward.py",
            "src/wt/lowlevel_v2/phase5_hgnn_urgency/planner.py",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("phase5o", text)


if __name__ == "__main__":
    unittest.main()
