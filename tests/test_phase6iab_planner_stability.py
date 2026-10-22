"""Stability gate checks. They do not run episodes or replace the official policy."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6iaa_formal_planner.protocol import POLICY_FILE
from wt.benchmark_v2.phase6iab_planner_stability.protocol import (
    BIT_TIE,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    SCENARIOS,
    assert_output_path,
    assert_scenario,
    assign_gate,
)


class Phase6IABPlannerStabilityTests(unittest.TestCase):
    def test_gate_uses_the_preregistered_interval(self) -> None:
        self.assertEqual(assign_gate(failures=1, complete_pairs=16, ci_low=1.0, ci_high=2.0), "PLANNER_STABILITY_AUDIT_BLOCKED")
        self.assertEqual(assign_gate(failures=0, complete_pairs=15, ci_low=1.0, ci_high=2.0), "PLANNER_STABILITY_AUDIT_BLOCKED")
        self.assertEqual(assign_gate(failures=0, complete_pairs=16, ci_low=1.0, ci_high=2.0), "PLANNER_STABILITY_GAIN_SUPPORTED")
        self.assertEqual(assign_gate(failures=0, complete_pairs=16, ci_low=-1.0, ci_high=2.0), "PLANNER_STABILITY_GAIN_INCONCLUSIVE")
        self.assertEqual(assign_gate(failures=0, complete_pairs=16, ci_low=-2.0, ci_high=-1.0), "PLANNER_STABILITY_GAIN_NEGATIVE")
        self.assertEqual(BIT_TIE, 1.0)
        self.assertEqual(BOOTSTRAP_DRAWS, 10000)
        self.assertEqual(BOOTSTRAP_SEED, 0)
        self.assertEqual(SCENARIOS, tuple(range(322048, 322064)))

    def test_reserved_scenarios_and_official_policy_are_protected(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_scenario(322064)
        with self.assertRaises(RuntimeError):
            assert_output_path(POLICY_FILE)

    def test_package_does_not_call_decide(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6iab_planner_stability"
        for path in root.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
