"""Phase-5F freeze checks. They do not train the HGNN."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from wt.lowlevel_v2.phase5f_utility_rank.loss import pairwise_rank_loss
from wt.lowlevel_v2.phase5f_utility_rank.protocol import FORBIDDEN_INPUTS, assert_phase5f_seed


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5f_utility_rank"


class Phase5FFreezeTests(unittest.TestCase):
    def test_holdout_and_new_scenario_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_phase5f_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_phase5f_seed(310102)
        assert_phase5f_seed(310100)
        assert_phase5f_seed(310101)

    def test_package_does_not_call_decide_or_mappo(self) -> None:
        called = []
        for path in PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    called.append(node.func.attr)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    called.append(node.func.id)
        self.assertNotIn("decide", called)
        self.assertNotIn("ICARMAPPOLocked", called)
        self.assertNotIn("UrgencyMAPPO", called)

    def test_pairwise_loss_prefers_the_better_rank(self) -> None:
        good = pairwise_rank_loss(torch.tensor([2.0, 0.0]), [1, 2])
        bad = pairwise_rank_loss(torch.tensor([0.0, 2.0]), [1, 2])
        tied = pairwise_rank_loss(torch.tensor([1.0, 1.0]), [1, 1])
        self.assertLess(float(good), float(bad))
        self.assertEqual(float(tied), 0.0)

    def test_label_fields_are_not_model_inputs(self) -> None:
        self.assertIn("payload_gain", FORBIDDEN_INPUTS)
        self.assertIn("utility_rank", FORBIDDEN_INPUTS)

    def test_frozen_files_have_no_phase5f_edit(self) -> None:
        high = (ROOT / "src" / "wt" / "htw_phase100" / "high_level.py").read_text(encoding="utf-8")
        self.assertIn("def decide(", high)
        self.assertNotIn("phase5f", high)
        encoder = (ROOT / "src" / "wt" / "lowlevel_v2" / "phase5_hgnn_urgency" / "encoder.py").read_text(encoding="utf-8")
        self.assertNotIn("phase5f", encoder)
        loss = (ROOT / "src" / "wt" / "htw_phase945" / "utility_loss.py").read_text(encoding="utf-8")
        self.assertIn("TAU", loss)
        self.assertNotIn("phase5f", loss)


if __name__ == "__main__":
    unittest.main()
