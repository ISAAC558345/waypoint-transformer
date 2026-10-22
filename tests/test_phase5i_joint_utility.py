"""Phase-5I freeze checks. They do not enumerate assignments."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.phase5i_joint_utility.protocol import assert_phase5i_seed
from wt.lowlevel_v2.phase5i_joint_utility.search import enumerate_exclusive


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5i_joint_utility"


class Phase5IFreezeTests(unittest.TestCase):
    def test_training_scenario_and_holdout_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_phase5i_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_phase5i_seed(310100)
        assert_phase5i_seed(310102)

    def test_exclusive_search_has_no_duplicate_targets(self) -> None:
        assignments = enumerate_exclusive([[0, 1], [1, 2], [0, 2]])
        self.assertGreater(len(assignments), 0)
        for assignment in assignments:
            chosen = [int(item) for item in assignment if int(item) >= 0]
            self.assertEqual(len(chosen), len(set(chosen)))
        self.assertTrue(any(np.array_equal(assignment, np.asarray([0, 1, 2])) for assignment in assignments))

    def test_package_does_not_train_or_call_decide(self) -> None:
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
        self.assertNotIn("Adam", called)

    def test_frozen_files_have_no_phase5i_edit(self) -> None:
        high = (ROOT / "src" / "wt" / "htw_phase100" / "high_level.py").read_text(encoding="utf-8")
        self.assertIn("def decide(", high)
        self.assertNotIn("phase5i", high)
        actor = (ROOT / "src" / "wt" / "lowlevel_v2" / "residual_policy.py").read_text(encoding="utf-8")
        self.assertNotIn("phase5i", actor)


if __name__ == "__main__":
    unittest.main()
