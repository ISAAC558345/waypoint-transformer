"""Phase-5ZO gate checks. They do not open the fresh pool."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import CONFIRM_SCENARIOS, TRAIN_SCENARIOS
from wt.lowlevel_v2.phase5zo_confirmation_freeze.protocol import FRESH_POOL, PLANNER_READY, assert_not_fresh, choose_gate


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "wt" / "lowlevel_v2" / "phase5zo_confirmation_freeze"


class Phase5ZOTests(unittest.TestCase):
    def test_clean_application_is_freeze_a_without_deployment(self) -> None:
        chosen = choose_gate(
            hash_match=True,
            integrity_pass=True,
            contamination_count=0,
            human_spec_applied=True,
            protocol_complete=True,
        )
        self.assertEqual(chosen["gate"], "FREEZE-A")
        self.assertTrue(chosen["frozen_for_fresh_confirmation"])
        self.assertTrue(chosen["fresh_confirmation_authorized"])
        self.assertFalse(chosen["planner_ready"])
        self.assertFalse(chosen["deployment_verified"])

    def test_hash_mismatch_stops_as_freeze_b(self) -> None:
        chosen = choose_gate(
            hash_match=False,
            integrity_pass=False,
            contamination_count=3,
            human_spec_applied=False,
            protocol_complete=False,
        )
        self.assertEqual(chosen["gate"], "FREEZE-B")
        self.assertEqual(chosen["section3_stop_name"], "FREEZE_MISMATCH")
        self.assertFalse(chosen["fresh_confirmation_authorized"])

    def test_pool_failure_is_freeze_c(self) -> None:
        chosen = choose_gate(
            hash_match=True,
            integrity_pass=False,
            contamination_count=0,
            human_spec_applied=True,
            protocol_complete=True,
        )
        self.assertEqual(chosen["gate"], "FREEZE-C")

    def test_contamination_is_freeze_d(self) -> None:
        chosen = choose_gate(
            hash_match=True,
            integrity_pass=True,
            contamination_count=1,
            human_spec_applied=True,
            protocol_complete=True,
        )
        self.assertEqual(chosen["gate"], "FREEZE-D")
        self.assertFalse(chosen["frozen_for_fresh_confirmation"])

    def test_incomplete_protocol_is_freeze_e(self) -> None:
        chosen = choose_gate(
            hash_match=True,
            integrity_pass=True,
            contamination_count=0,
            human_spec_applied=True,
            protocol_complete=False,
        )
        self.assertEqual(chosen["gate"], "FREEZE-E")

    def test_registries_and_package(self) -> None:
        self.assertFalse(set(FRESH_POOL) & set(TRAIN_SCENARIOS))
        self.assertFalse(set(FRESH_POOL) & set(CONFIRM_SCENARIOS))
        self.assertFalse(PLANNER_READY)
        with self.assertRaises(RuntimeError):
            assert_not_fresh(310118)
        for path in PACKAGE.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
