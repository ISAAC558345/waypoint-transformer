"""Phase-5X checks the shadow boundary and the gate. They do not train."""
from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path

import numpy as np

from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.phase5x_executor_lookahead.metrics import choose_gate
from wt.lowlevel_v2.phase5x_executor_lookahead.protocol import assert_consumed
from wt.lowlevel_v2.phase5x_executor_lookahead.shadow import fingerprint, max_abs_diff


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5x_executor_lookahead"


class Phase5XTests(unittest.TestCase):
    def test_clone_does_not_change_the_caller(self) -> None:
        env = L3BEnv()
        env.reset(seed=310100)
        before = fingerprint(env)
        clone = copy.deepcopy(env)
        assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
        assignment[0] = 0
        clone.step(np.zeros((env.config.num_uavs, 2)), assignment)
        self.assertEqual(max_abs_diff(before, fingerprint(env)), 0.0)
        self.assertGreater(int(clone.t), int(env.t))
        assert_consumed(310109)
        assert_consumed(310117)
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230000)

    def test_gate_letters(self) -> None:
        self.assertEqual(choose_gate(short_confirmed=True, long_confirmed=False, promising=False, only_full_execution=False), "X-A")
        self.assertEqual(choose_gate(short_confirmed=False, long_confirmed=True, promising=True, only_full_execution=False), "X-C")
        self.assertEqual(choose_gate(short_confirmed=False, long_confirmed=False, promising=True, only_full_execution=False), "X-B")
        self.assertEqual(choose_gate(short_confirmed=False, long_confirmed=False, promising=False, only_full_execution=True), "X-D")

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
        self.assertNotIn("Adam", called)
        self.assertNotIn("backward", called)
        for relative in (
            "src/wt/htw_phase100/high_level.py",
            "src/wt/lowlevel_v2/residual_policy.py",
            "src/wt/lowlevel_v2/phase5_hgnn_urgency/reward.py",
            "src/wt/lowlevel_v2/phase5_hgnn_urgency/planner.py",
        ):
            self.assertNotIn("phase5x", (ROOT / relative).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
