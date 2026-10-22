"""Phase-5J freeze checks. They do not roll out scenarios."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.phase5j_joint_redesign.joint import conflict_rate, joint_utility
from wt.lowlevel_v2.phase5j_joint_redesign.protocol import assert_phase5j_seed


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5j_joint_redesign"


class Phase5JFreezeTests(unittest.TestCase):
    def test_only_existing_phase5e_scenarios_are_accepted(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_phase5j_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_phase5j_seed(310102)
        assert_phase5j_seed(310100)
        assert_phase5j_seed(310101)

    def test_joint_label_equals_joint_payload_when_marginals_overcount(self) -> None:
        self.assertEqual(joint_utility(6.0e6, 1.0e6, lam=1.0), 1.0e6)
        self.assertEqual(conflict_rate(np.asarray([3, 3, 3, 1, 1, -1]), 12), 5.0 / 6.0)
        self.assertEqual(conflict_rate(np.asarray([0, 1, 2, 3, 4, 5]), 12), 0.0)

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

    def test_frozen_files_have_no_phase5j_edit(self) -> None:
        high = (ROOT / "src" / "wt" / "htw_phase100" / "high_level.py").read_text(encoding="utf-8")
        self.assertIn("def decide(", high)
        self.assertNotIn("phase5j", high)
        actor = (ROOT / "src" / "wt" / "lowlevel_v2" / "residual_policy.py").read_text(encoding="utf-8")
        self.assertNotIn("phase5j", actor)


if __name__ == "__main__":
    unittest.main()
