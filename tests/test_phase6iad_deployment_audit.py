"""Deployment-audit gates. They do not run an episode or open the holdout."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6iad_deployment_audit.protocol import (
    REGRESSION_SCENARIO,
    REPEAT_SCENARIO,
    assert_output_path,
    assert_scenario,
    assign_gate,
    status_flags,
)
from wt.benchmark_v2.phase6is_assignment_head.protocol import SCHEME_C_CKPT


class Phase6IADDeploymentAuditTests(unittest.TestCase):
    def test_gate_stays_unverified_without_requirements(self) -> None:
        incomplete = assign_gate(observed_faults=0, requirements_specified=False)
        blocked = assign_gate(observed_faults=1, requirements_specified=True)
        ready = assign_gate(observed_faults=0, requirements_specified=True)
        self.assertEqual(incomplete, "DEPLOYMENT_REQUIREMENTS_INCOMPLETE")
        self.assertEqual(blocked, "DEPLOYMENT_AUDIT_BLOCKED")
        self.assertEqual(ready, "DEPLOYMENT_READY")
        held = status_flags(gate=incomplete, planner_ready=True)
        self.assertTrue(held["planner_ready"])
        self.assertFalse(held["deployment_verified"])
        self.assertTrue(status_flags(gate=ready, planner_ready=True)["deployment_verified"])
        self.assertFalse(status_flags(gate=blocked, planner_ready=True)["planner_ready"])

    def test_regression_stays_outside_the_holdout(self) -> None:
        self.assertEqual(REGRESSION_SCENARIO, 322002)
        self.assertEqual(REPEAT_SCENARIO, 322000)
        assert_scenario(REGRESSION_SCENARIO)
        assert_scenario(REPEAT_SCENARIO)
        with self.assertRaises(RuntimeError):
            assert_scenario(322064)
        with self.assertRaises(RuntimeError):
            assert_output_path(SCHEME_C_CKPT)

    def test_package_does_not_call_decide(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6iad_deployment_audit"
        for path in root.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
