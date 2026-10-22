"""Phase-3: TW conditions actor and critic; env reward is unmodified."""
from __future__ import annotations

import unittest

from wt.htwc_phase1.adapter import CONDITIONED_UAV_DIM
from wt.htwc_phase1.conditioned_actor import GoalConditionedActorCritic
from wt.round30.features import UAV_DIM
from wt.round30.models import DeepSetsActorCritic
from wt.round30.ppo import PPOAgent


class HTWPhase3GoalConditionedTests(unittest.TestCase):
    def test_01_critic_reads_tw_width(self) -> None:
        model = GoalConditionedActorCritic()
        self.assertEqual(model.actor.shape.uav_dim, CONDITIONED_UAV_DIM)
        self.assertEqual(model.critic.shape.uav_dim, CONDITIONED_UAV_DIM)
        self.assertEqual(DeepSetsActorCritic().critic.shape.uav_dim, UAV_DIM)
        self.assertTrue(model.uses_task_waypoint)
        self.assertFalse(model.uses_transformer)

    def test_02_ppo_update_source_has_no_tracking(self) -> None:
        import inspect

        source = inspect.getsource(PPOAgent.update)
        self.assertIn("clip_eps", source)
        self.assertNotIn("track", source.lower())


if __name__ == "__main__":
    unittest.main()
