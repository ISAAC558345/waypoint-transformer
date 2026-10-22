"""Phase-5.1 oracle option checks."""
from __future__ import annotations

import unittest

import numpy as np

from wt.htw_phase51.models import OptionActorCritic
from wt.htw_phase51.option import (
    IDLE_INDEX,
    OPTION_DIM,
    OPTION_UAV_DIM,
    completion_score,
    encode_option,
    oracle_assignment,
)
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round30.features import UAV_DIM
from wt.round30.scenarios import training_scenarios


class HTWPhase51OracleOptionTests(unittest.TestCase):
    def test_01_option_includes_idle_and_one_hot(self) -> None:
        encoded = encode_option(np.array([-1, 2]))
        self.assertEqual(encoded.shape, (2, OPTION_DIM))
        self.assertAlmostEqual(float(encoded[0, IDLE_INDEX]), 1.0)
        self.assertAlmostEqual(float(encoded[1, 2]), 1.0)
        self.assertAlmostEqual(float(np.sum(encoded[0])), 1.0)

    def test_02_oracle_assignment_is_exclusive(self) -> None:
        scenario = training_scenarios()[0]
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        assignment = oracle_assignment(info, env)
        chosen = [int(item) for item in assignment if int(item) >= 0]
        self.assertEqual(len(chosen), len(set(chosen)))

    def test_03_completion_and_actor_width(self) -> None:
        self.assertAlmostEqual(completion_score(np.array([1, -1]), np.array([1, -1]), 2), 1.0)
        self.assertAlmostEqual(completion_score(np.array([0, 0]), np.array([1, 1]), 2), 0.0)
        model = OptionActorCritic()
        self.assertEqual(model.actor.shape.uav_dim, OPTION_UAV_DIM)
        self.assertEqual(model.critic.shape.uav_dim, OPTION_UAV_DIM)
        self.assertEqual(OPTION_UAV_DIM, UAV_DIM + OPTION_DIM)


if __name__ == "__main__":
    unittest.main()
