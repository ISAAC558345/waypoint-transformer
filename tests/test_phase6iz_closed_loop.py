"""Gate rules fixed before any paired rollout."""
from __future__ import annotations

import unittest

from wt.benchmark_v2.phase6iz_closed_loop.protocol import BIT_TIE, VALIDATION_POOL, assign_gate


class Phase6IZClosedLoopTests(unittest.TestCase):
    def test_validation_pool_and_gate(self) -> None:
        self.assertEqual(tuple(VALIDATION_POOL), tuple(range(322048, 322064)))
        self.assertEqual(assign_gate(failures=1, ci_low=10.0, ci_high=20.0), "CLOSED_LOOP_AUDIT_BLOCKED")
        self.assertEqual(assign_gate(failures=0, ci_low=1.0, ci_high=2.0), "CLOSED_LOOP_GAIN_SUPPORTED")
        self.assertEqual(assign_gate(failures=0, ci_low=-1.0, ci_high=2.0), "CLOSED_LOOP_GAIN_INCONCLUSIVE")
        self.assertEqual(assign_gate(failures=0, ci_low=-5.0, ci_high=-1.0), "CLOSED_LOOP_GAIN_NEGATIVE")
        self.assertEqual(BIT_TIE, 1.0)
