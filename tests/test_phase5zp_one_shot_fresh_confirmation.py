"""Phase-5ZP gate checks. They do not open 310118-310125."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.lowlevel_v2.phase5zp_one_shot_fresh_confirmation.protocol import (
    FRESH_POOL,
    choose_gate,
    confirmation_may_start,
    model_s_payload,
)


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "wt" / "lowlevel_v2" / "phase5zp_one_shot_fresh_confirmation"


class Phase5ZPTests(unittest.TestCase):
    def test_conf_a_requires_mean_interval_and_safety(self) -> None:
        chosen = choose_gate(
            protocol_violation=False,
            illegal_assignment=0,
            state_contamination=0,
            nan_final_decision=0,
            inf_final_decision=0,
            unhandled_runtime_exception=0,
            mean_delta=1.0,
            ci_low=0.1,
        )
        self.assertEqual(chosen["gate"], "CONF-A")
        self.assertTrue(chosen["independent_algorithm_confirmation"])
        self.assertFalse(chosen["planner_ready"])
        self.assertFalse(chosen["deployment_verified"])
        self.assertFalse(chosen["latency_gate_active"])

    def test_positive_mean_with_interval_covering_zero_is_conf_b(self) -> None:
        chosen = choose_gate(
            protocol_violation=False,
            illegal_assignment=0,
            state_contamination=0,
            nan_final_decision=0,
            inf_final_decision=0,
            unhandled_runtime_exception=0,
            mean_delta=1.0,
            ci_low=0.0,
        )
        self.assertEqual(chosen["gate"], "CONF-B")
        self.assertFalse(chosen["independent_algorithm_confirmation"])

    def test_non_positive_mean_is_conf_c(self) -> None:
        chosen = choose_gate(
            protocol_violation=False,
            illegal_assignment=0,
            state_contamination=0,
            nan_final_decision=0,
            inf_final_decision=0,
            unhandled_runtime_exception=0,
            mean_delta=0.0,
            ci_low=-1.0,
        )
        self.assertEqual(chosen["gate"], "CONF-C")

    def test_safety_failure_outranks_a_positive_mean(self) -> None:
        chosen = choose_gate(
            protocol_violation=False,
            illegal_assignment=1,
            state_contamination=0,
            nan_final_decision=0,
            inf_final_decision=0,
            unhandled_runtime_exception=0,
            mean_delta=5.0,
            ci_low=1.0,
        )
        self.assertEqual(chosen["gate"], "CONF-D")

    def test_protocol_violation_stops_before_the_pool(self) -> None:
        self.assertFalse(confirmation_may_start(False))
        chosen = choose_gate(
            protocol_violation=True,
            illegal_assignment=1,
            state_contamination=1,
            nan_final_decision=1,
            inf_final_decision=1,
            unhandled_runtime_exception=1,
            mean_delta=5.0,
            ci_low=1.0,
        )
        self.assertEqual(chosen["gate"], "CONF-E")

    def test_model_s_payload_is_the_mean_of_five_seeds(self) -> None:
        self.assertEqual(model_s_payload([1.0, 2.0, 3.0, 4.0, 10.0]), 4.0)

    def test_package_does_not_call_the_official_decision_function(self) -> None:
        for path in PACKAGE.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))

    def test_fresh_pool_is_the_preregistered_eight(self) -> None:
        self.assertEqual(FRESH_POOL, tuple(range(310118, 310126)))


if __name__ == "__main__":
    unittest.main()
