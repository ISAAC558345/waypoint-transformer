"""Phase-5ZI gate checks. No environment rollout."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.lowlevel_v2.phase5zi_ranking_failure_audit.metrics import choose_gate
from wt.lowlevel_v2.phase5zi_ranking_failure_audit.protocol import assert_consumed


class Phase5ZIRankingFailureAuditTest(unittest.TestCase):
    def test_gate_priority(self) -> None:
        self.assertEqual(
            choose_gate(implementation_error=True, teacher_mismatch=True, easy_region=True, true_h1_stronger=True),
            "ZI-A",
        )
        self.assertEqual(
            choose_gate(implementation_error=False, teacher_mismatch=True, easy_region=True, true_h1_stronger=True),
            "ZI-C",
        )
        self.assertEqual(
            choose_gate(implementation_error=False, teacher_mismatch=False, easy_region=True, true_h1_stronger=True),
            "ZI-B",
        )
        self.assertEqual(
            choose_gate(implementation_error=False, teacher_mismatch=False, easy_region=False, true_h1_stronger=True),
            "ZI-D",
        )
        self.assertEqual(
            choose_gate(implementation_error=False, teacher_mismatch=False, easy_region=False, true_h1_stronger=False),
            "ZI-E",
        )

    def test_closed_scenarios_are_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230100)

    def test_package_does_not_train_or_call_deployment(self) -> None:
        root = Path("src/wt/lowlevel_v2/phase5zi_ranking_failure_audit")
        for path in root.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("decide(", source)
            self.assertNotIn("optimizer.step", source)
            self.assertNotIn("frozen_h1", source)


if __name__ == "__main__":
    unittest.main()
