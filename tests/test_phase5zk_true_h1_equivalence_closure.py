"""Gate rules for the closure audit."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.l3b.env import L3BEnv, hypothetical_rates
from wt.lowlevel_v2.phase5x_executor_lookahead.shadow import fingerprint, max_abs_diff
from wt.lowlevel_v2.phase5zk_true_h1_equivalence_closure.metrics import choose_gate, combined_limit, within_combined
from wt.lowlevel_v2.phase5zk_true_h1_equivalence_closure.protocol import (
    ATOL,
    DEPLOYMENT_VERIFIED,
    FROZEN_FOR_FRESH_CONFIRMATION,
    PLANNER_READY,
    RTOL,
    assert_consumed,
)


class Phase5ZKTests(unittest.TestCase):
    def test_gate_priority(self) -> None:
        self.assertEqual(choose_gate(baseline_ok=False, behavioral_mismatch=False, numerical_exception=False, speedup_ok=True, latency_ci_positive=True), "ZK-E")
        self.assertEqual(choose_gate(baseline_ok=True, behavioral_mismatch=True, numerical_exception=False, speedup_ok=True, latency_ci_positive=True), "ZK-D")
        self.assertEqual(choose_gate(baseline_ok=True, behavioral_mismatch=False, numerical_exception=True, speedup_ok=True, latency_ci_positive=True), "ZK-B")
        self.assertEqual(choose_gate(baseline_ok=True, behavioral_mismatch=False, numerical_exception=False, speedup_ok=False, latency_ci_positive=True), "ZK-C")
        self.assertEqual(choose_gate(baseline_ok=True, behavioral_mismatch=False, numerical_exception=False, speedup_ok=True, latency_ci_positive=False), "ZK-C")
        self.assertEqual(choose_gate(baseline_ok=True, behavioral_mismatch=False, numerical_exception=False, speedup_ok=True, latency_ci_positive=True), "ZK-A")

    def test_relative_error_alone_does_not_decide(self) -> None:
        reference = 1.0e6
        other = reference + 0.02
        relative = abs(other - reference) / reference
        self.assertGreater(relative, 1.0e-8)
        self.assertLess(relative, float(RTOL))
        self.assertTrue(within_combined(other, reference, ATOL, RTOL))
        self.assertGreater(abs(0.02 - 0.0), combined_limit(0.0, ATOL, RTOL))

    def test_flags_and_closed_scenarios(self) -> None:
        self.assertFalse(PLANNER_READY)
        self.assertFalse(DEPLOYMENT_VERIFIED)
        self.assertFalse(FROZEN_FOR_FRESH_CONFIRMATION)
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)

    def test_package_does_not_call_the_planner(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "lowlevel_v2" / "phase5zk_true_h1_equivalence_closure"
        for path in root.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))

    def test_counterfactual_rates_do_not_write_state(self) -> None:
        env = L3BEnv()
        env.reset(seed=1)
        before = fingerprint(env)
        hypothetical_rates(env)
        after = fingerprint(env)
        self.assertEqual(max_abs_diff(before, after), 0.0)
        self.assertEqual(before["rng"], after["rng"])


if __name__ == "__main__":
    unittest.main()
