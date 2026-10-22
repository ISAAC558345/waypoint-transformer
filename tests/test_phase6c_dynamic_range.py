"""Phase-6C gates and channel equivalence. They do not open the reserved pool."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import hypothetical_match, rates_match_environment
from wt.benchmark_v2.phase6c.controllers import one_step_optima, random_legal
from wt.benchmark_v2.phase6c.protocol import assert_development, choose_gate


class Phase6CTests(unittest.TestCase):
    def test_reserved_pool_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_development(320100)
        with self.assertRaises(RuntimeError):
            assert_development(310118)
        assert_development(320000)

    def test_fast_channel_matches_legacy_formula(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=320000)
        generator = np.random.default_rng(0)
        for _ in range(6):
            self.assertTrue(rates_match_environment(env, random_legal(env, generator)))
        self.assertTrue(hypothetical_match(env))

    def test_previous_assignment_does_not_change_one_step_optimum(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=320000)
        first = np.asarray(one_step_optima(env)["true_bit_assignment"], dtype=np.int64).copy()
        env.assignment[:] = np.arange(env.config.num_uavs)
        second = np.asarray(one_step_optima(env)["true_bit_assignment"], dtype=np.int64)
        self.assertTrue(np.array_equal(first, second))

    def test_accounting_regression_is_dyn_e(self) -> None:
        chosen = choose_gate(
            conservation_ok=False,
            best_coverage_at_3=1.0,
            dynamic_range_ok=True,
            relational_confirmed=True,
            short_horizon_upper_bound_available=False,
        )
        self.assertEqual(chosen["gate"], "DYN-E")
        self.assertFalse(chosen["planner_ready"])
        self.assertTrue(chosen["fresh_confirmation_untouched"])

    def test_weak_coverage_precedes_relational_claim(self) -> None:
        chosen = choose_gate(
            conservation_ok=True,
            best_coverage_at_3=0.5,
            dynamic_range_ok=True,
            relational_confirmed=True,
            short_horizon_upper_bound_available=False,
        )
        self.assertEqual(chosen["gate"], "DYN-F")

    def test_no_upper_bound_cannot_become_dyn_c(self) -> None:
        chosen = choose_gate(
            conservation_ok=True,
            best_coverage_at_3=0.99,
            dynamic_range_ok=True,
            relational_confirmed=False,
            short_horizon_upper_bound_available=False,
        )
        self.assertEqual(chosen["gate"], "DYN-A")
        self.assertEqual(chosen["low_level_status"], "LOW_LEVEL_RETRAIN_PENDING")


if __name__ == "__main__":
    unittest.main()
