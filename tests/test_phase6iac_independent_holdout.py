"""Holdout gate checks. They do not open a test scenario episode."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6iaa_formal_planner.protocol import POLICY_FILE, assert_episode_scenario, assert_scenario
from wt.benchmark_v2.phase6iac_independent_holdout.protocol import (
    BIT_TIE,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    SCENARIOS,
    SMOKE,
    assert_output_path,
    assign_gate,
)
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import TRAIN_POOL, VALIDATION_POOL


class Phase6IACIndependentHoldoutTests(unittest.TestCase):
    def test_gate_is_fixed_before_the_holdout(self) -> None:
        self.assertEqual(assign_gate(failures=1, complete_pairs=16, ci_low=1.0, ci_high=2.0), "INDEPENDENT_TEST_AUDIT_BLOCKED")
        self.assertEqual(assign_gate(failures=0, complete_pairs=15, ci_low=1.0, ci_high=2.0), "INDEPENDENT_TEST_AUDIT_BLOCKED")
        self.assertEqual(assign_gate(failures=0, complete_pairs=16, ci_low=1.0, ci_high=2.0), "INDEPENDENT_GAIN_SUPPORTED")
        self.assertEqual(assign_gate(failures=0, complete_pairs=16, ci_low=-1.0, ci_high=2.0), "INDEPENDENT_GAIN_INCONCLUSIVE")
        self.assertEqual(assign_gate(failures=0, complete_pairs=16, ci_low=-2.0, ci_high=-1.0), "INDEPENDENT_GAIN_NEGATIVE")
        self.assertEqual((BIT_TIE, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED), (1.0, 10000, 0))
        self.assertEqual(SCENARIOS, tuple(range(322064, 322080)))
        self.assertEqual(SMOKE, (322064, 322065))
        self.assertTrue(set(SCENARIOS).isdisjoint(TRAIN_POOL))
        self.assertTrue(set(SCENARIOS).isdisjoint(VALIDATION_POOL))

    def test_default_entry_still_refuses_the_holdout(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_scenario(322064)
        with self.assertRaises(RuntimeError):
            assert_episode_scenario(322064, allow_holdout=False)
        assert_episode_scenario(322064, allow_holdout=True)
        assert_episode_scenario(322079, allow_holdout=True)
        with self.assertRaises(RuntimeError):
            assert_output_path(POLICY_FILE)

    def test_package_does_not_call_decide(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6iac_independent_holdout"
        for path in root.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
