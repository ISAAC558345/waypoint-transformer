"""Phase-5W checks the service rule and the gate. They do not train."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.phase5w_residual_mechanism_audit.metrics import choose_gate, synergy_ratio, synergy_value
from wt.lowlevel_v2.phase5w_residual_mechanism_audit.protocol import assert_consumed
from wt.lowlevel_v2.phase5w_residual_mechanism_audit.run_phase5w import _hybrids, _legal
from wt.lowlevel_v2.phase5w_residual_mechanism_audit.semantics import service_semantics


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5w_residual_mechanism_audit"


class Phase5WTests(unittest.TestCase):
    def test_progress_resets_on_switch_and_pauses_outside_radius(self) -> None:
        env = L3BEnv()
        env.reset(seed=1)
        env.uav_positions[0] = env.buoy_positions[0]
        assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
        assignment[0] = 0
        movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
        env.step(movement, assignment)
        self.assertGreater(float(env.progress[0]), 0.0)
        held = float(env.progress[0])
        env.uav_positions[0] = env.buoy_positions[0] + np.array([40.0, 0.0])
        env.step(movement, assignment)
        self.assertAlmostEqual(float(env.progress[0]), held)
        assignment[0] = 6
        env.step(movement, assignment)
        self.assertEqual(float(env.progress[0]), 0.0)
        self.assertEqual(int(env.served_k[0]), 6)
        rules = service_semantics()
        self.assertEqual(rules["progress_owner"], "uav")
        self.assertEqual(rules["switch_target"], "reset")
        self.assertEqual(rules["leave_service_radius"], "pause")
        self.assertEqual(rules["completion_rule"], "accumulator")
        self.assertFalse(rules["consecutive_steps_required"])

    def test_gate_and_synergy(self) -> None:
        self.assertEqual(choose_gate(shortcut=True, synergy=False), "W-A")
        self.assertEqual(choose_gate(shortcut=False, synergy=True), "W-B")
        self.assertEqual(choose_gate(shortcut=True, synergy=True), "W-C")
        self.assertEqual(choose_gate(shortcut=False, synergy=False), "W-D")
        self.assertAlmostEqual(synergy_value(5.0e6, [0.0, 0.0, 0.0]), 5.0e6)
        self.assertAlmostEqual(synergy_ratio(5.0e6, [0.0, 0.0]), 1.0, places=5)
        predicted = np.array([2, 5, -1, 4, -1, -1])
        optimal = np.array([-1, 4, 2, 5, -1, -1])
        hybrids = _hybrids(predicted, optimal)
        self.assertLessEqual(len(hybrids), 16)
        self.assertTrue(all(_legal(row) for row, _count in hybrids))
        self.assertFalse(_legal(np.array([1, 1, -1, -1, -1, -1])))
        assert_consumed(310109)
        assert_consumed(310117)
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230000)

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
            self.assertNotIn("phase5w", (ROOT / relative).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
