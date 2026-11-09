"""Phase-2.5 alignment unit checks."""
from __future__ import annotations

import unittest

import numpy as np

from wt.legacy.htw_historical25.alignment import alignment_rows, summarize_alignment
from wt.legacy.htw_historical25.tracking_loss import TrackingLossAgent
from wt.legacy.htw_historical2.tracking import auxiliary_reward
from wt.legacy.deepsets_ppo.models import DeepSetsActorCritic
from wt.algorithms.ppo_common import PPOConfig
from wt.legacy.deepsets_ppo.scenarios import training_scenarios


class HTWPhase25AlignmentTests(unittest.TestCase):
    def test_01_zero_lambda_keeps_comm_reward(self) -> None:
        self.assertAlmostEqual(auxiliary_reward(0.4, 0.25, 0.0), 0.4)

    def test_02_tracking_loss_agent_keeps_clip(self) -> None:
        import inspect

        source = inspect.getsource(TrackingLossAgent.update)
        self.assertIn("clip_eps", source)
        self.assertIn("track_beta", source)
        agent = TrackingLossAgent(DeepSetsActorCritic(), PPOConfig(rollout_length=16, num_envs=2, num_minibatches=2, update_epochs=1), track_beta=0.05)
        self.assertEqual(agent.track_beta, 0.05)

    def test_03_alignment_rows_have_channel_fields(self) -> None:
        rows = alignment_rows(training_scenarios()[:2], horizon=5)
        self.assertGreater(len(rows), 0)
        self.assertIn("channel_gain_waypoint_target", rows[0])
        self.assertIn("rate_waypoint_target_bps", rows[0])
        summary = summarize_alignment(rows)
        self.assertTrue(np.isfinite(summary["mean_rate_delta_target_bps"]))


if __name__ == "__main__":
    unittest.main()