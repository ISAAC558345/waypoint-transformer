"""Preserving-assignment rules. No episode replay."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6i_execution_attribution.run_attribution import assert_validation_scenario
from wt.benchmark_v2.phase6i_t45_active_preserve.run_preserve import (
    choose_preserving_assignment,
    uav_roles,
)


class PreserveAssignmentRuleTest(unittest.TestCase):
    def test_reserved_scenario_is_refused(self):
        with self.assertRaises(RuntimeError):
            assert_validation_scenario(322064)

    def test_only_an_unreachable_assigned_uav_can_move(self):
        uavs = np.zeros((2, 2), dtype=np.float64)
        buoys = np.asarray([[10.0, 0.0], [90.0, 0.0], [50.0, 0.0], [79.0, 0.0]], dtype=np.float64)
        keys = [(0, 1), (2, 1), (0, 3), (0, -1), (0, 2)]
        choice = choose_preserving_assignment(uavs, buoys, keys, 0, 16.0, 22.0, 3)
        self.assertEqual(choice["status"], "selected")
        self.assertEqual(choice["assignment"], (0, 2))
        self.assertEqual(choice["reassignable_uavs"], [1])
        self.assertEqual(choice["reachable_uavs"], 1)

    def test_equal_reach_uses_distance_then_pool_order(self):
        uavs = np.zeros((2, 2), dtype=np.float64)
        buoys = np.asarray([[10.0, 0.0], [90.0, 0.0], [50.0, 0.0], [-50.0, 0.0]], dtype=np.float64)
        keys = [(0, 1), (0, 3), (0, 2)]
        choice = choose_preserving_assignment(uavs, buoys, keys, 0, 16.0, 22.0, 3)
        self.assertEqual(choice["index"], 1)
        closer = [(0, 1), (0, 2), (0, 3)]
        picked = choose_preserving_assignment(uavs, buoys, closer, 0, 16.0, 22.0, 3)
        self.assertEqual(picked["assignment"], (0, 2))

    def test_idle_and_in_range_targets_stay_fixed(self):
        uavs = np.zeros((2, 2), dtype=np.float64)
        buoys = np.asarray([[10.0, 0.0], [90.0, 0.0]], dtype=np.float64)
        roles, _distances = uav_roles(uavs, buoys, (-1, 1), 16.0, 22.0, 3)
        self.assertEqual(roles, ["fixed_idle", "reassignable_unreachable"])
        choice = choose_preserving_assignment(uavs, buoys, [(-1, 1), (0, 1)], 0, 16.0, 22.0, 3)
        self.assertEqual(choice["assignment"], (-1, 1))
        inside = uav_roles(uavs, buoys, (0,), 16.0, 22.0, 3)[0]
        self.assertEqual(inside, ["fixed_in_radius"])

    def test_boundary_and_no_reassignable_uav(self):
        uavs = np.zeros((1, 2), dtype=np.float64)
        self.assertEqual(uav_roles(uavs, np.asarray([[82.0, 0.0]]), (0,), 16.0, 22.0, 3)[0], ["fixed_outside_reachable"])
        self.assertEqual(uav_roles(uavs, np.asarray([[83.0, 0.0]]), (0,), 16.0, 22.0, 3)[0], ["reassignable_unreachable"])
        choice = choose_preserving_assignment(
            np.zeros((1, 2)),
            np.asarray([[10.0, 0.0]]),
            [(0,), (-1,)],
            0,
            16.0,
            22.0,
            3,
        )
        self.assertEqual(choice["status"], "no_reassignable_uav")
        self.assertEqual(choice["assignment"], (0,))


if __name__ == "__main__":
    unittest.main()
