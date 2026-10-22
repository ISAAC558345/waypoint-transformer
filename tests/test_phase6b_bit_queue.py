"""Phase-6B bit-queue accounting. The legacy environment is not edited here."""
from __future__ import annotations

import unittest

from wt.benchmark_v2.phase6b.accounting_checks import run_accounting_tests
from wt.benchmark_v2.phase6b.protocol import assert_sanity_scenario, choose_gate
from wt.l3b.env import L3BEnv


class Phase6BTests(unittest.TestCase):
    def test_reserved_pool_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_sanity_scenario(320100)
        with self.assertRaises(RuntimeError):
            assert_sanity_scenario(310118)
        assert_sanity_scenario(320000)

    def test_accounting_cases_pass(self) -> None:
        result = run_accounting_tests()
        failed = [row["test"] for row in result["rows"] if not row["passed"]]
        self.assertEqual(failed, [])

    def test_legacy_class_is_not_the_v2_class(self) -> None:
        self.assertEqual(L3BEnv.__name__, "L3BEnv")
        source = L3BEnv.step.__doc__
        self.assertIsNone(source)

    def test_failed_conservation_is_not_bitq_a(self) -> None:
        chosen = choose_gate(
            rate_unit_known=True,
            legacy_coupling_blocks=False,
            transition_complete=True,
            conservation_ok=False,
            tests_ok=True,
            switch_persists=True,
            cross_uav_ok=True,
            no_double_decrement=True,
            payload_matches_service=True,
        )
        self.assertEqual(chosen["gate"], "BITQ-C")
        self.assertFalse(chosen["planner_ready"])
        self.assertEqual(chosen["phase5zp_gate"], "CONF-C")
        self.assertEqual(chosen["lambda_switch"], "NOT_DEFINED")

    def test_unknown_rate_unit_stops(self) -> None:
        chosen = choose_gate(
            rate_unit_known=False,
            legacy_coupling_blocks=False,
            transition_complete=True,
            conservation_ok=True,
            tests_ok=True,
            switch_persists=True,
            cross_uav_ok=True,
            no_double_decrement=True,
            payload_matches_service=True,
        )
        self.assertEqual(chosen["gate"], "BITQ-D")


if __name__ == "__main__":
    unittest.main()
