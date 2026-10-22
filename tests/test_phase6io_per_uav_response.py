"""Diagnostic swaps stay inside the validation pool and the rollout budget."""
from __future__ import annotations

import unittest

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6io_per_uav_response.audit import CASES, MAX_NEW_ROLLOUTS, assign_gate, compose_swaps


class Phase6IOPerUavResponseTests(unittest.TestCase):
    def test_open_pool_excludes_reserved_scenarios(self) -> None:
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))
        self.assertTrue(all(int(case["scenario_id"]) in VALIDATION_POOL for case in CASES))

    def test_new_rollouts_stay_within_the_budget(self) -> None:
        fresh = {(int(case["scenario_id"]), tuple(assignment)) for case in CASES if not case["reuse"] for assignment in case["arms"].values()}
        self.assertLessEqual(len(fresh), MAX_NEW_ROLLOUTS)
        self.assertEqual(len(fresh), 9)
        self.assertEqual(len({case["name"] for case in CASES}), len(CASES))

    def test_two_swap_composition_matches_the_joint_assignment(self) -> None:
        self.assertEqual(compose_swaps((10, 6, 11, 8, 9, 7), (0, 2), (2, 4)), (11, 6, 9, 8, 10, 7))

    def test_gate_requires_every_large_contrast_to_stay_on_the_swapped_uavs(self) -> None:
        self.assertEqual(assign_gate(True, True, 4, 4), "PER_UAV_ATTRIBUTION_BLOCKED")
        self.assertEqual(assign_gate(False, False, 4, 4), "PER_UAV_ATTRIBUTION_BLOCKED")
        self.assertEqual(assign_gate(False, True, 4, 4), "PER_UAV_ATTRIBUTION_SUPPORTED")
        self.assertEqual(assign_gate(False, True, 1, 4), "PER_UAV_ATTRIBUTION_INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
