"""Phase-5ZN specification checks. They do not open the fresh pool."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import CONFIRM_SCENARIOS, TRAIN_SCENARIOS
from wt.lowlevel_v2.phase5zn_human_deployment_specification.protocol import (
    FRESH_CONFIRMATION_AUTHORIZED,
    FRESH_POOL,
    MODEL_R1_AGGREGATION_APPROVED,
    PLANNER_READY,
    assert_not_fresh,
    choose_gate,
)


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "wt" / "lowlevel_v2" / "phase5zn_human_deployment_specification"


class Phase5ZNTests(unittest.TestCase):
    def test_waiting_state_is_spec_a(self) -> None:
        chosen = choose_gate(integrity_pass=True, freeze_consistent=True, human_approval_pending=True)
        self.assertEqual(chosen["gate"], "SPEC-A")
        self.assertFalse(chosen["planner_ready"])
        self.assertFalse(chosen["fresh_confirmation_authorized"])

    def test_consumed_pool_is_spec_b(self) -> None:
        chosen = choose_gate(integrity_pass=False, freeze_consistent=True, human_approval_pending=True)
        self.assertEqual(chosen["gate"], "SPEC-B")

    def test_inconsistent_freeze_is_spec_c(self) -> None:
        chosen = choose_gate(integrity_pass=True, freeze_consistent=False, human_approval_pending=True)
        self.assertEqual(chosen["gate"], "SPEC-C")

    def test_completed_approval_is_not_a_deployment_pass(self) -> None:
        with self.assertRaises(RuntimeError):
            choose_gate(integrity_pass=True, freeze_consistent=True, human_approval_pending=False)

    def test_registries_do_not_include_the_fresh_pool(self) -> None:
        self.assertFalse(set(FRESH_POOL) & set(TRAIN_SCENARIOS))
        self.assertFalse(set(FRESH_POOL) & set(CONFIRM_SCENARIOS))
        self.assertFalse(PLANNER_READY)
        self.assertFalse(FRESH_CONFIRMATION_AUTHORIZED)
        self.assertFalse(MODEL_R1_AGGREGATION_APPROVED)
        with self.assertRaises(RuntimeError):
            assert_not_fresh(310118)

    def test_package_does_not_call_deployment_decision(self) -> None:
        for path in PACKAGE.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
