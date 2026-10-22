"""Phase-4 planner swap checks. No HGNN, no env/reward edits."""
from __future__ import annotations

import unittest

import numpy as np

from wt.htw_phase4.planners import planner_a_action, planner_b_action, planner_window
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round30.scenarios import training_scenarios


class HTWPhase4PlannerTests(unittest.TestCase):
    def test_01_planner_b_can_differ_from_a(self) -> None:
        scenario = training_scenarios()[0]
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        action_a = planner_a_action(env, info)
        action_b = planner_b_action(env, info)
        self.assertEqual(action_a.shape, action_b.shape)
        self.assertTrue(np.all(np.isfinite(action_b)))

    def test_02_window_emits_tw_contract(self) -> None:
        scenario = training_scenarios()[0]
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        label, path = planner_window(env, info, planner_a_action, horizon=5)
        self.assertEqual(label.horizon, 5)
        self.assertEqual(label.waypoint_xy.shape, (2, 2))
        self.assertEqual(path.shape[0], 6)

    def test_03_planner_c_single_beam_window(self) -> None:
        from wt.htw_phase4.planners import planner_c_window

        scenario = training_scenarios()[0]
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        label, path = planner_c_window(env, info, horizon=5)
        self.assertEqual(label.horizon, 5)
        self.assertEqual(path.shape[0], 6)


if __name__ == "__main__":
    unittest.main()
