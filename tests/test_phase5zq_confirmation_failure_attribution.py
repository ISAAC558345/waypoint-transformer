"""Phase-5ZQ attribution rules. They do not rerun fresh confirmation."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.lowlevel_v2.phase5zq_confirmation_failure_attribution.protocol import (
    PHASE5ZP_GATE,
    choose_gate,
    conservation_error,
)


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "wt" / "lowlevel_v2" / "phase5zq_confirmation_failure_attribution"


class Phase5ZQTests(unittest.TestCase):
    def test_conservation_identity(self) -> None:
        self.assertEqual(conservation_error(15, 102, 3, 78, 36), 0.0)

    def test_accounting_bug_is_attr_e_and_does_not_revise_conf_c(self) -> None:
        chosen = choose_gate(
            accounting_bug=True,
            reproduction_mismatch=False,
            sinr_or_rate_gain=True,
            progress_or_completion_gain=False,
            intermediates_differ=True,
            timing_or_unfinished_differ=True,
            completion_saturated=True,
        )
        self.assertEqual(chosen["gate"], "ATTR-E")
        self.assertEqual(chosen["phase5zp_gate"], PHASE5ZP_GATE)
        self.assertFalse(chosen["phase5zp_gate_revised"])
        self.assertFalse(chosen["independent_algorithm_confirmation"])

    def test_gain_without_completion_plus_saturation_is_mixed(self) -> None:
        chosen = choose_gate(
            accounting_bug=False,
            reproduction_mismatch=False,
            sinr_or_rate_gain=True,
            progress_or_completion_gain=False,
            intermediates_differ=True,
            timing_or_unfinished_differ=True,
            completion_saturated=True,
        )
        self.assertEqual(chosen["gate"], "ATTR-D")
        self.assertFalse(chosen["planner_ready"])
        self.assertFalse(chosen["deployment_verified"])

    def test_saturation_without_rate_gain_is_attr_c(self) -> None:
        chosen = choose_gate(
            accounting_bug=False,
            reproduction_mismatch=False,
            sinr_or_rate_gain=False,
            progress_or_completion_gain=False,
            intermediates_differ=True,
            timing_or_unfinished_differ=True,
            completion_saturated=True,
        )
        self.assertEqual(chosen["gate"], "ATTR-C")

    def test_no_intermediate_difference_is_attr_a(self) -> None:
        chosen = choose_gate(
            accounting_bug=False,
            reproduction_mismatch=False,
            sinr_or_rate_gain=False,
            progress_or_completion_gain=False,
            intermediates_differ=False,
            timing_or_unfinished_differ=False,
            completion_saturated=True,
        )
        self.assertEqual(chosen["gate"], "ATTR-A")

    def test_package_does_not_call_the_official_decision_function(self) -> None:
        for path in PACKAGE.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
