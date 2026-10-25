"""t=30 preserving rule. Outside UAVs may move; in-range and idle stay fixed."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6i_execution_attribution.run_attribution import assert_validation_scenario
from wt.benchmark_v2.phase6i_t30_active_preserve.run_preserve import (
    choose_preserving_assignment,
    uav_roles,
)


class T30PreserveRuleTest(unittest.TestCase):
    def test_reserved_scenario_is_refused(self):
        with self.assertRaises(RuntimeError):
            assert_validation_scenario(322064)

    def test_outside_uav_can_move_to_a_closer_buoy(self):
        uavs = np.zeros((2, 2), dtype=np.float64)
        buoys = np.asarray([[10.0, 0.0], [40.0, 0.0], [25.0, 0.0]], dtype=np.float64)
        roles, _distances = uav_roles(uavs, buoys, (0, 1), 16.0)
        self.assertEqual(roles, ["fixed_in_radius", "reassignable_outside"])
        keys = [(0, 1), (2, 1), (0, 2)]
        choice = choose_preserving_assignment(uavs, buoys, keys, 0, 16.0, 22.0, 15)
        self.assertEqual(choice["status"], "selected")
        self.assertEqual(choice["assignment"], (0, 2))
        self.assertEqual(choice["reassignable_uavs"], [1])

    def test_idle_stays_idle_and_equal_distance_keeps_pool_order(self):
        uavs = np.zeros((2, 2), dtype=np.float64)
        buoys = np.asarray([[40.0, 0.0], [-25.0, 0.0], [25.0, 0.0]], dtype=np.float64)
        self.assertEqual(uav_roles(uavs, buoys, (-1, 0), 16.0)[0][0], "fixed_idle")
        kept = choose_preserving_assignment(uavs, buoys, [(-1, 0), (1, 0)], 0, 16.0, 22.0, 15)
        self.assertEqual(kept["assignment"], (-1, 0))
        tied = choose_preserving_assignment(uavs, buoys, [(0, 0), (0, 1), (0, 2)], 0, 16.0, 22.0, 15)
        self.assertEqual(tied["index"], 1)

    def test_fifty_metres_is_reassignable_on_a_fifteen_step_budget(self):
        uavs = np.zeros((1, 2), dtype=np.float64)
        roles, _distances = uav_roles(uavs, np.asarray([[50.0, 0.0]]), (0,), 16.0)
        self.assertEqual(roles, ["reassignable_outside"])


if __name__ == "__main__":
    unittest.main()
