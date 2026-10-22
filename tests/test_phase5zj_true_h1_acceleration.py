"""Gate and safety checks for the experimental H1 path."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.phase5zj_true_h1_acceleration.fast_path import info_without_counterfactual, observation_gap
from wt.lowlevel_v2.phase5zj_true_h1_acceleration.metrics import choose_gate
from wt.lowlevel_v2.phase5zj_true_h1_acceleration.protocol import (
    ATOL,
    DEPLOYMENT_VERIFIED,
    FROZEN_FOR_FRESH_CONFIRMATION,
    MEDIAN_SPEEDUP,
    PLANNER_READY,
    RTOL,
    assert_consumed,
)


class Phase5ZJTests(unittest.TestCase):
    def test_gate_priority(self) -> None:
        self.assertEqual(choose_gate(baseline_ok=False, batch_safe=True, equivalent=True, speedup_ok=True, latency_ci_positive=True), "FAST-E")
        self.assertEqual(choose_gate(baseline_ok=True, batch_safe=False, equivalent=True, speedup_ok=True, latency_ci_positive=True), "FAST-D")
        self.assertEqual(choose_gate(baseline_ok=True, batch_safe=True, equivalent=False, speedup_ok=True, latency_ci_positive=True), "FAST-C")
        self.assertEqual(choose_gate(baseline_ok=True, batch_safe=True, equivalent=True, speedup_ok=True, latency_ci_positive=True), "FAST-A")
        self.assertEqual(choose_gate(baseline_ok=True, batch_safe=True, equivalent=True, speedup_ok=False, latency_ci_positive=True), "FAST-B")
        self.assertEqual(choose_gate(baseline_ok=True, batch_safe=True, equivalent=True, speedup_ok=True, latency_ci_positive=False), "FAST-B")

    def test_assignment_change_is_not_accelerated(self) -> None:
        self.assertEqual(choose_gate(baseline_ok=True, batch_safe=True, equivalent=False, speedup_ok=True, latency_ci_positive=True), "FAST-C")

    def test_flags_and_tolerances_stay_locked(self) -> None:
        self.assertFalse(PLANNER_READY)
        self.assertFalse(DEPLOYMENT_VERIFIED)
        self.assertFalse(FROZEN_FOR_FRESH_CONFIRMATION)
        self.assertEqual(ATOL, 1.0e-6)
        self.assertEqual(RTOL, 1.0e-5)
        self.assertEqual(MEDIAN_SPEEDUP, 0.20)

    def test_fresh_confirmation_stays_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)

    def test_package_does_not_call_the_planner(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "lowlevel_v2" / "phase5zj_true_h1_acceleration"
        for path in root.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))

    def test_slim_info_keeps_interference_and_drops_only_the_unused_grid(self) -> None:
        env = L3BEnv()
        env.reset(seed=310110)
        movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
        assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
        assignment[0] = 0
        env.step(movement, assignment)
        original = L3BEnv.info(env)
        slim = info_without_counterfactual(env)
        self.assertTrue(np.array_equal(original["interference_uav_uav"], slim["interference_uav_uav"]))
        self.assertTrue(np.array_equal(original["sinr"], slim["sinr"]))
        self.assertTrue(np.array_equal(original["rate_bps"], slim["rate_bps"]))
        self.assertTrue(np.array_equal(original["service_progress"], slim["service_progress"]))
        self.assertEqual(float(original["cumulative_payload_bits"]), float(slim["cumulative_payload_bits"]))
        self.assertTrue(np.array_equal(slim["rate_if_assigned"], np.zeros_like(slim["rate_if_assigned"])))

    def test_cached_observation_matches_the_original_builder(self) -> None:
        env = L3BEnv()
        env.reset(seed=310110)
        movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
        assignment = np.asarray([0, 1, -1, 2, -1, 3], dtype=np.int64)
        self.assertEqual(observation_gap(env, assignment, movement), 0.0)


if __name__ == "__main__":
    unittest.main()
