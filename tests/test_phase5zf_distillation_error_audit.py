"""Locked audit rules. Thresholds are not retuned from confirmation scores."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.phase5zf_distillation_error_audit.metrics import choose_gate, pairwise_accuracy, recovery_ratio
from wt.lowlevel_v2.phase5zf_distillation_error_audit.protocol import GROUPS, assert_consumed


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5zf_distillation_error_audit"


class Phase5ZFTests(unittest.TestCase):
    def test_groups_cover_each_response_once(self) -> None:
        covered = [index for indexes in GROUPS.values() for index in indexes]
        self.assertEqual(sorted(covered), list(range(14)))
        self.assertEqual(len(covered), len(set(covered)))

    def test_gate_priority(self) -> None:
        self.assertEqual(
            choose_gate(aligned=False, sinr_ok=True, progress_ok=True, interference_ok=True, all_single_below=True, combined_ok=True),
            "ERR-F",
        )
        self.assertEqual(
            choose_gate(aligned=True, sinr_ok=True, progress_ok=True, interference_ok=False, all_single_below=False, combined_ok=True),
            "ERR-A",
        )
        self.assertEqual(
            choose_gate(aligned=True, sinr_ok=False, progress_ok=True, interference_ok=False, all_single_below=False, combined_ok=False),
            "ERR-B",
        )
        self.assertEqual(
            choose_gate(aligned=True, sinr_ok=False, progress_ok=False, interference_ok=True, all_single_below=False, combined_ok=False),
            "ERR-C",
        )
        self.assertEqual(
            choose_gate(aligned=True, sinr_ok=False, progress_ok=False, interference_ok=False, all_single_below=True, combined_ok=True),
            "ERR-D",
        )
        self.assertEqual(
            choose_gate(aligned=True, sinr_ok=False, progress_ok=False, interference_ok=False, all_single_below=False, combined_ok=True),
            "ERR-E",
        )

    def test_pairwise_and_recovery(self) -> None:
        truth = np.asarray([1.0, 2.0, 3.0])
        self.assertEqual(pairwise_accuracy(truth, truth, 1.0e-8), 1.0)
        self.assertEqual(pairwise_accuracy(truth, truth[::-1], 1.0e-8), 0.0)
        self.assertAlmostEqual(recovery_ratio(0.7, 1.0), 0.7)
        self.assertTrue(np.isnan(recovery_ratio(1.0, 0.0)))

    def test_confirm_scope(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230100)

    def test_package_has_no_training_or_deployment_call(self) -> None:
        banned = ("torch.save", "Adam", "decide(")
        for path in PACKAGE.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in banned:
                self.assertNotIn(token, text, msg=f"{path.name} contains {token}")


if __name__ == "__main__":
    unittest.main()
