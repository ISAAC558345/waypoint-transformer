"""Distance-rule ties and the reserved-test boundary. No episode replay."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6i_execution_attribution.run_attribution import assert_validation_scenario
from wt.benchmark_v2.phase6i_t45_counterfactual.run_counterfactual import (
    assignment_geometry,
    choose_distance_assignment,
)


class T45DistanceRuleTest(unittest.TestCase):
    def test_reserved_scenario_is_refused(self):
        with self.assertRaises(RuntimeError):
            assert_validation_scenario(322064)

    def test_more_reachable_uavs_win_over_a_shorter_total(self):
        uavs = np.zeros((2, 2), dtype=np.float64)
        buoys = np.asarray([[10.0, 0.0], [80.0, 0.0], [200.0, 0.0]], dtype=np.float64)
        keys = [(2, 2), (0, 0), (1, 1), (0, 2)]
        choice = choose_distance_assignment(uavs, buoys, keys, 0, 16.0, 22.0, 3)
        self.assertEqual(choice["status"], "selected")
        self.assertEqual(choice["index"], 1)
        self.assertEqual(choice["reachable_uavs"], 2)

    def test_equal_reach_uses_total_distance_then_pool_order(self):
        uavs = np.zeros((2, 2), dtype=np.float64)
        buoys = np.asarray([[10.0, 0.0], [30.0, 0.0], [200.0, 0.0]], dtype=np.float64)
        keys = [(2, 2), (1, 0), (0, 1), (0, 0)]
        choice = choose_distance_assignment(uavs, buoys, keys, 0, 16.0, 22.0, 3)
        self.assertEqual(choice["index"], 3)
        same = [(2, 2), (0, 1), (1, 0)]
        tied = choose_distance_assignment(uavs, buoys, same, 0, 16.0, 22.0, 3)
        self.assertEqual(tied["index"], 1)

    def test_no_better_geometry_keeps_scheme_c(self):
        uavs = np.zeros((1, 2), dtype=np.float64)
        buoys = np.asarray([[200.0, 0.0]], dtype=np.float64)
        choice = choose_distance_assignment(uavs, buoys, [(0,), (-1,)], 0, 16.0, 22.0, 3)
        self.assertEqual(choice["status"], "no_better_geometric_candidate")
        self.assertEqual(choice["assignment"], (0,))

    def test_idle_is_not_a_reachable_target(self):
        uavs = np.zeros((1, 2), dtype=np.float64)
        buoys = np.asarray([[10.0, 0.0]], dtype=np.float64)
        reach, total, distances = assignment_geometry(uavs, buoys, (-1,), 16.0, 22.0, 3)
        self.assertEqual(reach, 0)
        self.assertEqual(total, 0.0)
        self.assertIsNone(distances[0])

    def test_reach_budget_is_three_steps(self):
        uavs = np.zeros((1, 2), dtype=np.float64)
        inside = assignment_geometry(uavs, np.asarray([[82.0, 0.0]]), (0,), 16.0, 22.0, 3)[0]
        outside = assignment_geometry(uavs, np.asarray([[83.0, 0.0]]), (0,), 16.0, 22.0, 3)[0]
        self.assertEqual(inside, 1)
        self.assertEqual(outside, 0)


if __name__ == "__main__":
    unittest.main()
