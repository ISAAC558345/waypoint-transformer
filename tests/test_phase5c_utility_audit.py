"""Phase-5C freeze checks. They do not run the payload shadow."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from wt.lowlevel_v2.phase5c_utility_audit.protocol import assert_audit_seed


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5c_utility_audit" / "audit.py"


class Phase5CFreezeTests(unittest.TestCase):
    def test_holdout_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_audit_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_audit_seed(230127)
        assert_audit_seed(310100)

    def test_audit_does_not_train_or_call_decide(self) -> None:
        tree = ast.parse(AUDIT.read_text(encoding="utf-8"))
        called = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called.append(node.func.attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.append(node.func.id)
        self.assertNotIn("decide", called)
        self.assertNotIn("backward", called)
        self.assertNotIn("Adam", called)

    def test_planner_file_has_no_phase5c_edit(self) -> None:
        high = (ROOT / "src" / "wt" / "htw_phase100" / "high_level.py").read_text(encoding="utf-8")
        self.assertIn("def decide(", high)
        self.assertNotIn("phase5c", high)
        loss = (ROOT / "src" / "wt" / "htw_phase945" / "utility_loss.py").read_text(encoding="utf-8")
        self.assertIn("l_util + float(LAMBDA_RANK) * l_rank + float(LAMBDA_REF) * l_ref", loss)


if __name__ == "__main__":
    unittest.main()
