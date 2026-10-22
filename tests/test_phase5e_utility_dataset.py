"""Phase-5E freeze checks. They do not build the full dataset."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from wt.lowlevel_v2.phase5e_utility_dataset.protocol import FEATURE_COLUMNS, LABEL_COLUMNS, assert_dataset_seed
from wt.lowlevel_v2.phase5e_utility_dataset.utility import descending_rank, utility_score


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5e_utility_dataset"


class Phase5EFreezeTests(unittest.TestCase):
    def test_holdout_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_dataset_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_dataset_seed(230127)
        assert_dataset_seed(310100)

    def test_package_does_not_train_or_call_decide(self) -> None:
        called = []
        for path in PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    called.append(node.func.attr)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    called.append(node.func.id)
                if isinstance(node, ast.Attribute):
                    called.append(node.attr)
        self.assertNotIn("decide", called)
        self.assertNotIn("backward", called)
        self.assertNotIn("Adam", called)
        self.assertNotIn("remaining_arrivals", called)

    def test_baseline_utility_is_payload(self) -> None:
        self.assertEqual(utility_score(2.0e6, 0.5, 22.0, 3.0, 0.0), 2.0e6)
        self.assertEqual(set(FEATURE_COLUMNS).intersection(LABEL_COLUMNS), set())

    def test_ties_share_rank(self) -> None:
        self.assertEqual(descending_rank([5.0, 5.0, 1.0]), [1, 1, 3])

    def test_frozen_files_have_no_phase5e_edit(self) -> None:
        high = (ROOT / "src" / "wt" / "htw_phase100" / "high_level.py").read_text(encoding="utf-8")
        self.assertIn("def decide(", high)
        self.assertNotIn("phase5e", high)
        actor = (ROOT / "src" / "wt" / "lowlevel_v2" / "residual_policy.py").read_text(encoding="utf-8")
        self.assertNotIn("phase5e", actor)


if __name__ == "__main__":
    unittest.main()
