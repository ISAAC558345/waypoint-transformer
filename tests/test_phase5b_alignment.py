"""Phase-5B freeze checks. They do not run the oracle shadow search."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from wt.lowlevel_v2.phase5b_alignment.features import hgnn_feature_table
from wt.lowlevel_v2.phase5b_alignment.protocol import assert_diagnosis_seed


ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5b_alignment" / "diagnose.py"


class Phase5BFreezeTests(unittest.TestCase):
    def test_holdout_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_diagnosis_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_diagnosis_seed(230127)
        assert_diagnosis_seed(310100)

    def test_diagnosis_does_not_call_decide_or_train(self) -> None:
        source = DIAG.read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called.append(node.func.attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.append(node.func.id)
        self.assertNotIn("decide", called)
        self.assertNotIn("backward", called)
        self.assertNotIn("Adam", called)
        self.assertIn("decode_assignment", called)

    def test_feature_table_covers_required_rows(self) -> None:
        rows = {row["feature"]: row for row in hgnn_feature_table()}
        for name in ("UAV-UAV edge", "UAV-Buoy edge", "interference feature", "channel quality", "energy cost", "queue state", "urgency"):
            self.assertIn(name, rows)
        self.assertEqual(rows["UAV-UAV edge"]["current"], "yes")
        self.assertEqual(rows["UAV-Buoy edge"]["current"], "yes")
        self.assertIn("0", rows["energy cost"]["current"])

    def test_planner_and_decide_files_are_not_this_phase(self) -> None:
        high = (ROOT / "src" / "wt" / "htw_phase100" / "high_level.py").read_text(encoding="utf-8")
        self.assertIn("def decide(", high)
        self.assertNotIn("phase5b", high)


if __name__ == "__main__":
    unittest.main()
