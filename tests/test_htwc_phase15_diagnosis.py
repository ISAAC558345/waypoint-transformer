"""Phase-1.5 diagnosis unit checks. No formal PPO in this file except tiny optional skip."""
from __future__ import annotations

import unittest

import numpy as np

from wt.htwc_phase1.oracle import oracle_planner_window, tw_enters_actor
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round30.features import feature_dict
from wt.round30.scenarios import training_scenarios


class HTWPhase15DiagnosisTests(unittest.TestCase):
    def test_01_oracle_waypoint_matches_planner_h_step_position(self) -> None:
        scenario = training_scenarios()[0]
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        label, path = oracle_planner_window(env, info, horizon=5)
        end = path[-1]
        width, height = float(env.config.area_width), float(env.config.area_height)
        expected = np.stack((2.0 * end[:, 0] / width - 1.0, 2.0 * end[:, 1] / height - 1.0), axis=-1)
        error = np.linalg.norm(np.asarray(label.waypoint_xy[: env.num_uavs]) - expected, axis=-1)
        self.assertTrue(np.all(error < 1.0e-6))

    def test_02_oracle_features_enter_conditioned_uav_width(self) -> None:
        from wt.htwc_phase1.oracle import oracle_conditioner

        scenario = training_scenarios()[0]
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        features = oracle_conditioner(info, env, horizon=5)
        flags = tw_enters_actor(features)
        self.assertTrue(flags["uav_width_is_conditioned"])
        self.assertTrue(flags["waypoint_nonzero"] or flags["goal_nonzero"])
        self.assertEqual(feature_dict(info, env.config)["uav"].shape[-1], 8)
        self.assertEqual(features["uav"].shape[-1], 16)


if __name__ == "__main__":
    unittest.main()
