"""Short-horizon gate stays closed when the three windows disagree."""
from __future__ import annotations

import unittest

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6iq_short_horizon.audit import PILOT_SCENARIOS, assign_gate, evaluate_metric, relation


class Phase6IQShortHorizonTests(unittest.TestCase):
    def test_open_pool_excludes_reserved_scenarios(self) -> None:
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))
        self.assertTrue(set(PILOT_SCENARIOS).isdisjoint(RESERVED_TEST))

    def test_missing_observation_is_not_a_zero(self) -> None:
        self.assertEqual(relation(None, 0.0, higher=True, floor=0.05), "missing")
        self.assertEqual(relation(0.0, 0.0, higher=True, floor=1.0), "tie")

    def test_gate_requires_one_metric_stable_at_every_horizon(self) -> None:
        self.assertEqual(assign_gate(True, 1), "SHORT_HORIZON_RESPONSE_BLOCKED")
        self.assertEqual(assign_gate(False, 1), "SHORT_HORIZON_RESPONSE_SUPPORTED")
        self.assertEqual(assign_gate(False, 0), "SHORT_HORIZON_RESPONSE_INCONCLUSIVE")

    def test_early_tie_does_not_support_a_later_separation(self) -> None:
        states = [
            {
                "items": [
                    {"j15": 2.0e6, "values": {"1": 0.0, "3": 10.0, "5": 30.0}},
                    {"j15": 0.0, "values": {"1": 0.0, "3": 0.0, "5": 0.0}},
                ]
            }
        ]
        report = evaluate_metric(states, higher=True, floor=1.0)
        self.assertEqual(report["supported"], 0)
        self.assertEqual(report["by_horizon"]["5"]["agree"], 1)
        self.assertEqual(report["by_horizon"]["1"]["tie"], 1)

    def test_matching_order_at_every_horizon_is_supported(self) -> None:
        states = [
            {
                "items": [
                    {"j15": 2.0e6, "values": {"1": 3.0, "3": 4.0, "5": 5.0}},
                    {"j15": 0.0, "values": {"1": 1.0, "3": 1.0, "5": 1.0}},
                ]
            }
        ]
        report = evaluate_metric(states, higher=True, floor=1.0)
        self.assertEqual(report["rank_stable"], 1)
        self.assertEqual(report["consistent"], 1)
        self.assertEqual(report["supported"], 1)


if __name__ == "__main__":
    unittest.main()
