"""Phase-5R checks. They do not roll out the new scenarios."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import choose_gate, cluster_bootstrap
from wt.lowlevel_v2.phase5r_nonadditive_generalization.protocol import PAYLOAD_BLOCK_BITS, assert_phase5r_seed


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5r_nonadditive_generalization"


class Phase5RTests(unittest.TestCase):
    def test_holdout_and_training_scenarios_are_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_phase5r_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_phase5r_seed(310100)
        with self.assertRaises(RuntimeError):
            assert_phase5r_seed(310101)
        assert_phase5r_seed(310102)
        assert_phase5r_seed(310109)

    def test_gate_outcomes(self) -> None:
        self.assertEqual(
            choose_gate(
                mean_delta_j=1.0e6,
                delta_j_low=1.0,
                n_scenarios_up=6,
                n_scenarios=8,
                n_seeds_up=4,
                n_seeds=5,
                mean_delta_regret=-0.2,
                regret_high=-0.01,
                coverage_ok=True,
                pool_hit_fraction=0.2,
                mean_coverage_regret=0.0,
            ),
            "R-A",
        )
        self.assertEqual(
            choose_gate(
                mean_delta_j=1.0e6,
                delta_j_low=-1.0,
                n_scenarios_up=6,
                n_scenarios=8,
                n_seeds_up=4,
                n_seeds=5,
                mean_delta_regret=-0.2,
                regret_high=0.1,
                coverage_ok=True,
                pool_hit_fraction=0.2,
                mean_coverage_regret=0.0,
            ),
            "R-B",
        )
        self.assertEqual(
            choose_gate(
                mean_delta_j=1.0e6,
                delta_j_low=1.0,
                n_scenarios_up=6,
                n_scenarios=8,
                n_seeds_up=4,
                n_seeds=5,
                mean_delta_regret=-0.2,
                regret_high=-0.01,
                coverage_ok=False,
                pool_hit_fraction=0.8,
                mean_coverage_regret=PAYLOAD_BLOCK_BITS,
            ),
            "R-C",
        )
        self.assertEqual(
            choose_gate(
                mean_delta_j=-1.0,
                delta_j_low=-2.0,
                n_scenarios_up=1,
                n_scenarios=8,
                n_seeds_up=1,
                n_seeds=5,
                mean_delta_regret=0.2,
                regret_high=0.4,
                coverage_ok=True,
                pool_hit_fraction=0.1,
                mean_coverage_regret=0.0,
            ),
            "R-D",
        )

    def test_bootstrap_resamples_scenarios(self) -> None:
        pairs = []
        for scenario, value in ((310102, 0.0), (310103, 10.0)):
            for moment in (0, 15, 30):
                pairs.append({"scenario_id": scenario, "time_step": moment, "delta_j_mappo": value})
        first = cluster_bootstrap(pairs, "delta_j_mappo", draws=200, seed=0)
        second = cluster_bootstrap(pairs, "delta_j_mappo", draws=200, seed=0)
        self.assertEqual(first, second)
        self.assertLess(first["low"], first["mean"])
        self.assertGreater(first["high"], first["mean"])

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
            self.assertNotIn("phase5r", text)


if __name__ == "__main__":
    unittest.main()
