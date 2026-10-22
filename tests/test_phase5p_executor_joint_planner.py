"""Phase-5P checks. They do not roll out scenarios."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.phase5p_executor_joint_planner.metrics import choose_gate, signal_counts
from wt.lowlevel_v2.phase5p_executor_joint_planner.protocol import PAYLOAD_BLOCK_BITS, assert_phase5p_seed


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5p_executor_joint_planner"


class Phase5PTests(unittest.TestCase):
    def test_holdout_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_phase5p_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_phase5p_seed(310102)
        assert_phase5p_seed(310100)

    def test_ties_do_not_create_ranking_pairs(self) -> None:
        tied = signal_counts(np.zeros(4))
        self.assertEqual(tied["n_positive_pairs"], 0.0)
        self.assertEqual(tied["n_unique"], 1.0)
        mixed = np.array([0.0, 0.0, PAYLOAD_BLOCK_BITS, PAYLOAD_BLOCK_BITS * 3.0])
        counts = signal_counts(mixed)
        self.assertEqual(counts["n_unique"], 3.0)
        self.assertGreater(counts["n_positive_pairs"], 0.0)
        noise = signal_counts(np.array([0.0, 10.0, PAYLOAD_BLOCK_BITS]))
        self.assertEqual(noise["n_unique"], 2.0)

    def test_gate_rules(self) -> None:
        self.assertEqual(
            choose_gate(
                train_rank_up=True,
                train_payload_up=True,
                dev_rank_up=False,
                dev_payload_up=False,
                dev_regret_down=False,
                n_seeds_improved=0,
                n_seeds=5,
            ),
            "LABELTRAIN-D",
        )
        self.assertEqual(
            choose_gate(
                train_rank_up=True,
                train_payload_up=True,
                dev_rank_up=True,
                dev_payload_up=True,
                dev_regret_down=True,
                n_seeds_improved=4,
                n_seeds=5,
            ),
            "LABELTRAIN-A",
        )
        self.assertEqual(
            choose_gate(
                train_rank_up=True,
                train_payload_up=False,
                dev_rank_up=True,
                dev_payload_up=False,
                dev_regret_down=False,
                n_seeds_improved=1,
                n_seeds=5,
            ),
            "LABELTRAIN-B",
        )
        self.assertEqual(
            choose_gate(
                train_rank_up=False,
                train_payload_up=False,
                dev_rank_up=False,
                dev_payload_up=False,
                dev_regret_down=False,
                n_seeds_improved=1,
                n_seeds=5,
            ),
            "LABELTRAIN-C",
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
            self.assertNotIn("phase5p", text)


if __name__ == "__main__":
    unittest.main()
