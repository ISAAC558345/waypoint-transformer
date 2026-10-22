"""Phase-10.6 tests. No training and no untouched access beyond SHA/manifest checks."""
from __future__ import annotations

import json
import unittest

from wt.htw_phase101.protocol import HIGH_CKPT
from wt.htw_phase103.gate import InRangeZeroMotionGate
from wt.htw_phase105.protocol import SCENARIO_BASE as P105
from wt.htw_phase106.protocol import EXPECTED_HIGH_SHA, N_SCENARIOS, SCENARIO_BASE, candidate_untouched_seeds, historical_blocked_seeds, sha256_bytes
from wt.htw_phase106.stats import classify_final


class Phase106Tests(unittest.TestCase):
    def test_01_restore_sha_and_sys_a(self) -> None:
        self.assertTrue(HIGH_CKPT.is_file())
        self.assertEqual(sha256_bytes(HIGH_CKPT.read_bytes()), EXPECTED_HIGH_SHA)
        from wt.htw_phase106.protocol import ARCH_MANIFEST, FREEZE_FILE

        arch = json.loads(ARCH_MANIFEST.read_text(encoding="utf-8"))
        self.assertTrue(arch["architecture_frozen"])
        self.assertTrue(arch["training_frozen"])
        self.assertTrue(arch["hyperparameters_frozen"])
        sys105 = json.loads((ARCH_MANIFEST.parent / "phase105_system_gate.json").read_text(encoding="utf-8"))
        self.assertEqual(sys105["SYS"], "A")
        self.assertTrue(FREEZE_FILE.is_file())

    def test_02_untouched_candidates_disjoint(self) -> None:
        seeds = candidate_untouched_seeds()
        self.assertEqual(len(seeds), N_SCENARIOS)
        self.assertEqual(seeds, list(range(SCENARIO_BASE, SCENARIO_BASE + N_SCENARIOS)))
        blocked = historical_blocked_seeds()
        self.assertFalse(set(seeds) & blocked)
        self.assertFalse(set(seeds) & set(range(P105, P105 + 64)))

    def test_03_gate_and_baselines_unmodified_modules(self) -> None:
        self.assertEqual(InRangeZeroMotionGate.__module__, "wt.htw_phase103.gate")
        from wt.htw_phase105.baselines import exclusive_assignment, legal_myopic_assignment

        self.assertTrue(callable(legal_myopic_assignment))
        self.assertTrue(callable(exclusive_assignment))

    def test_04_final_gate_logic(self) -> None:
        pos = {"mean_delta": 1.0, "ci95": {"low": 0.2, "high": 2.0}}
        a = classify_final(vs_m0=pos, vs_m1=pos, vs_m2=pos, blocks_m3=10, blocks_m0=8, blocks_m2=7, legal_m3=1, legal_m0=1, legal_m2=1, drop_m3=1, drop_m0=2, drop_m2=2, idle_m3=0.5, idle_m0=0.5, idle_m2=0.4)
        self.assertEqual(a["FINAL"], "A")
        b = classify_final(vs_m0={"mean_delta": 1.0, "ci95": {"low": -0.1, "high": 2.0}}, vs_m1=pos, vs_m2=pos, blocks_m3=10, blocks_m0=8, blocks_m2=7, legal_m3=1, legal_m0=1, legal_m2=1, drop_m3=1, drop_m0=2, drop_m2=2, idle_m3=0.5, idle_m0=0.5, idle_m2=0.4)
        self.assertEqual(b["FINAL"], "B")
        c = classify_final(vs_m0={"mean_delta": -1.0, "ci95": {"low": -2.0, "high": 0.0}}, vs_m1=pos, vs_m2=pos, blocks_m3=10, blocks_m0=8, blocks_m2=7, legal_m3=1, legal_m0=1, legal_m2=1, drop_m3=1, drop_m0=2, drop_m2=2, idle_m3=0.5, idle_m0=0.5, idle_m2=0.4)
        self.assertEqual(c["FINAL"], "C")
        self.assertFalse(c["final_system_confirmed"])


if __name__ == "__main__":
    unittest.main()
