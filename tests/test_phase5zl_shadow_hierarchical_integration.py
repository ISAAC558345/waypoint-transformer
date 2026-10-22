"""Phase-5ZL gate and isolation checks."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.lowlevel_v2.phase5zl_shadow_hierarchical_integration.protocol import (
    DEPLOYMENT_VERIFIED,
    FORBIDDEN,
    PLANNER_READY,
    assert_consumed,
    choose_gate,
)
from wt.lowlevel_v2.phase5zl_shadow_hierarchical_integration.shadow import select_index


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5zl_shadow_hierarchical_integration"


class Phase5ZLTests(unittest.TestCase):
    def test_unspecified_latency_is_not_int_a(self) -> None:
        chosen = choose_gate(
            reproduced=True,
            isolated=True,
            mapping_ok=True,
            decisions_match=True,
            seeds_agree=True,
            latency_specified=False,
            fallback_approved=False,
        )
        self.assertEqual(chosen["gate"], "INT-D")
        self.assertIn("INT-D", chosen["blockers"])
        self.assertTrue(chosen["contract_technically_clean"])
        self.assertFalse(chosen["planner_ready"])

    def test_seed_disagreement_records_both_blockers(self) -> None:
        chosen = choose_gate(
            reproduced=True,
            isolated=True,
            mapping_ok=True,
            decisions_match=True,
            seeds_agree=False,
            latency_specified=False,
            fallback_approved=False,
        )
        self.assertEqual(chosen["gate"], "INT-C")
        self.assertEqual(chosen["blockers"], ["INT-C", "INT-D"])

    def test_reproduction_failure_outranks_isolation(self) -> None:
        chosen = choose_gate(
            reproduced=False,
            isolated=False,
            mapping_ok=False,
            decisions_match=False,
            seeds_agree=False,
            latency_specified=False,
            fallback_approved=False,
        )
        self.assertEqual(chosen["gate"], "INT-E")
        self.assertEqual(chosen["blockers"], ["INT-E", "INT-B", "INT-C", "INT-D"])

    def test_clean_specified_contract_is_int_a(self) -> None:
        chosen = choose_gate(
            reproduced=True,
            isolated=True,
            mapping_ok=True,
            decisions_match=True,
            seeds_agree=True,
            latency_specified=True,
            fallback_approved=True,
        )
        self.assertEqual(chosen["gate"], "INT-A")
        self.assertEqual(chosen["blockers"], [])

    def test_flags_and_forbidden_window(self) -> None:
        self.assertFalse(PLANNER_READY)
        self.assertFalse(DEPLOYMENT_VERIFIED)
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        self.assertIn(310118, FORBIDDEN)

    def test_package_does_not_call_deployment_decision(self) -> None:
        for path in PACKAGE.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))

    def test_tie_break_is_stable(self) -> None:
        import numpy as np

        self.assertEqual(select_index(np.asarray([1.0, 3.0, 3.0, 2.0])), 1)


if __name__ == "__main__":
    unittest.main()
