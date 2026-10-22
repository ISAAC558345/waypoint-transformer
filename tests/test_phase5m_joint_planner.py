"""Phase-5M planner checks. They do not roll out scenarios."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.phase5m_joint_planner.planner import exhaustive_refine, greedy_refine
from wt.lowlevel_v2.phase5m_joint_planner.protocol import assert_phase5m_seed


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5m_joint_planner"


class Phase5MTests(unittest.TestCase):
    def test_holdout_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_phase5m_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_phase5m_seed(310102)
        assert_phase5m_seed(310100)
        assert_phase5m_seed(310101)

    def test_refinement_prefers_the_higher_joint_assignment(self) -> None:
        options = [[0, 1], [0, 1]]

        def evaluate(assignment: np.ndarray) -> float:
            score = 0.0
            if int(assignment[0]) == 1:
                score += 2.0
            if int(assignment[1]) == 0:
                score += 1.0
            return score

        exhaustive, exhaustive_calls = exhaustive_refine(options, evaluate)
        greedy, greedy_calls = greedy_refine(options, evaluate)
        self.assertTrue(np.array_equal(exhaustive, np.asarray([1, 0])))
        self.assertTrue(np.array_equal(greedy, np.asarray([1, 0])))
        self.assertGreater(exhaustive_calls, 1)
        self.assertGreater(greedy_calls, 1)

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
            self.assertNotIn("phase5m", text)


if __name__ == "__main__":
    unittest.main()
