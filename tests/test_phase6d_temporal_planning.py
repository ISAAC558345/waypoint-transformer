"""Phase-6D checks. They do not open the reserved confirmation pool."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.protocol import assert_development, choose_gate
from wt.benchmark_v2.phase6d.shadow import macro_return


class Phase6DTemporalPlanningTest(unittest.TestCase):
    def test_reserved_and_historical_ids_are_refused(self) -> None:
        for scenario in (320100, 320131, 310118, 310125):
            with self.assertRaises(RuntimeError):
                assert_development(scenario)
        assert_development(320000)

    def test_current_shadow_does_not_read_future_arrivals(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=320000)
        assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
        assignment[0] = 6
        current = macro_return(env, assignment, 15, use_realized_future_arrivals=False)
        future = macro_return(env, assignment, 15, use_realized_future_arrivals=True)
        self.assertLessEqual(current, 1.0)
        self.assertGreater(future, current)
        self.assertEqual(int(env.t), 0)

    def test_gate_stops_on_reproduction_before_planning_claims(self) -> None:
        failed = choose_gate(
            reproduction_ok=False,
            conservation_ok=True,
            coverage_failure=False,
            hold_aware_beats_max_rate=True,
            temporal_objective_mismatch=True,
            future_beats_current=True,
            future_beats_max_rate=True,
            hold_mismatch=True,
        )
        self.assertEqual(failed["gate"], "TEMP-E")
        self.assertFalse(failed["planner_ready"])
        self.assertFalse(failed["deployment_verified"])
        self.assertFalse(failed["shadow_is_intermediate_decision_layer"])
        uncovered = choose_gate(
            reproduction_ok=True,
            conservation_ok=True,
            coverage_failure=True,
            hold_aware_beats_max_rate=True,
            temporal_objective_mismatch=True,
            future_beats_current=False,
            future_beats_max_rate=False,
            hold_mismatch=False,
        )
        self.assertEqual(uncovered["gate"], "TEMP-F")

    def test_package_does_not_call_decide(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6d"
        for path in root.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
