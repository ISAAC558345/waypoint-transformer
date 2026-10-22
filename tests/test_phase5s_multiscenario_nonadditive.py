"""Phase-5S checks. They do not open confirmation scenarios."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.metrics import choose_gate
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import (
    PAYLOAD_BLOCK_BITS,
    assert_confirm_seed,
    assert_train_seed,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5s_multiscenario_nonadditive"


class Phase5STests(unittest.TestCase):
    def test_seed_roles_are_separated(self) -> None:
        assert_train_seed(310100)
        assert_train_seed(310109)
        assert_confirm_seed(310110)
        assert_confirm_seed(310117)
        with self.assertRaises(RuntimeError):
            assert_train_seed(310110)
        with self.assertRaises(RuntimeError):
            assert_confirm_seed(310109)
        with self.assertRaises(RuntimeError):
            assert_confirm_seed(230000)

    def test_gate_outcomes(self) -> None:
        common = dict(
            n_scenarios=8,
            n_seeds=5,
            regret_phase5q=2.0e6,
            regret_phase5s=1.0e6,
            coverage_regret=4.0e5,
        )
        self.assertEqual(
            choose_gate(
                mean_delta_j=1.0e6,
                delta_j_low=1.0,
                mean_delta_regret=-0.2,
                n_scenarios_up=6,
                n_seeds_up=4,
                **common,
            ),
            "S-A",
        )
        self.assertEqual(
            choose_gate(
                mean_delta_j=1.0e5,
                delta_j_low=-1.0,
                mean_delta_regret=-0.05,
                n_scenarios_up=5,
                n_seeds_up=3,
                **common,
            ),
            "S-C",
        )
        self.assertEqual(
            choose_gate(
                mean_delta_j=0.0,
                delta_j_low=-1.0,
                mean_delta_regret=0.0,
                n_scenarios_up=2,
                n_seeds_up=2,
                regret_phase5q=1.0e5,
                regret_phase5s=1.0e5,
                coverage_regret=PAYLOAD_BLOCK_BITS,
                n_scenarios=8,
                n_seeds=5,
            ),
            "S-B",
        )
        self.assertEqual(
            choose_gate(
                mean_delta_j=-1.0e6,
                delta_j_low=-2.0e6,
                mean_delta_regret=0.2,
                n_scenarios_up=1,
                n_seeds_up=1,
                **common,
            ),
            "S-D",
        )

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
            self.assertNotIn("phase5s", text)


if __name__ == "__main__":
    unittest.main()
