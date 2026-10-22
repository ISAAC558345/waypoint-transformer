"""Phase-5G freeze checks. They do not run the closed loop."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from wt.lowlevel_v2.phase5g_closed_loop.protocol import assert_phase5g_seed


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5g_closed_loop"


class Phase5GFreezeTests(unittest.TestCase):
    def test_training_scenario_and_holdout_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_phase5g_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_phase5g_seed(310100)
        assert_phase5g_seed(310102)
        assert_phase5g_seed(310103)

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
        self.assertNotIn("Adam", called)

    def test_frozen_files_have_no_phase5g_edit(self) -> None:
        high = (ROOT / "src" / "wt" / "htw_phase100" / "high_level.py").read_text(encoding="utf-8")
        self.assertIn("def decide(", high)
        self.assertNotIn("phase5g", high)
        actor = (ROOT / "src" / "wt" / "lowlevel_v2" / "residual_policy.py").read_text(encoding="utf-8")
        self.assertNotIn("phase5g", actor)


if __name__ == "__main__":
    unittest.main()
