"""Phase-6F entry and isolation checks. The long J15 search is not repeated here."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6e.generators import build_pools
from wt.benchmark_v2.phase6e.value import batch_j15
from wt.benchmark_v2.phase6f_system_bringup.episode import _contamination, _fingerprint
from wt.benchmark_v2.phase6f_system_bringup.protocol import assert_allowed, choose_gate


class Phase6FTests(unittest.TestCase):
    def test_reserved_and_unused_ids_are_refused(self) -> None:
        for scenario in (320100, 320131, 310118, 320004, 320007, 320016):
            with self.assertRaises(RuntimeError):
                assert_allowed(scenario, split="smoke")
        assert_allowed(320000, split="smoke")
        assert_allowed(320008, split="integration")

    def test_shadow_does_not_change_the_live_environment(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=320000)
        pools, _packed, illegal = build_pools(env, (("G1_RATE", 5),))
        before = _fingerprint(env, None)
        _values, _elapsed, error = batch_j15(env, pools[("G1_RATE", 5)][:1], 1)
        after = _fingerprint(env, None)
        self.assertEqual(illegal, 0)
        self.assertEqual(_contamination(before, after), 0)
        self.assertLessEqual(error, float(env.tol_bits))
        self.assertEqual(int(env.t), 0)
        self.assertFalse(before["velocity_field_absent"] is False)

    def test_gate_stops_on_contamination_before_accounting(self) -> None:
        contaminated = choose_gate(
            contamination=1,
            accounting_failures=1,
            crashed=True,
            smoke_ran=0,
            smoke_complete=0,
            integration_complete=0,
            illegal=1,
            nan_inf=1,
            hold_ok=False,
            chain_ok=False,
            logs_ok=False,
        )
        self.assertEqual(contaminated["gate"], "RUN-C")
        accounting = choose_gate(
            contamination=0,
            accounting_failures=1,
            crashed=True,
            smoke_ran=0,
            smoke_complete=0,
            integration_complete=0,
            illegal=0,
            nan_inf=0,
            hold_ok=False,
            chain_ok=False,
            logs_ok=False,
        )
        self.assertEqual(accounting["gate"], "RUN-D")

    def test_package_does_not_call_decide(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6f_system_bringup"
        for path in root.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))
            self.assertNotIn("planner_ready = True", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
