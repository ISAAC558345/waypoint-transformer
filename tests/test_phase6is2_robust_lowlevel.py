"""Temporal credit and observation checks without a high-level model."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from wt.control.mappo.worker import make_option
from wt.models.hgnn_assets.protocol import RESERVED_TEST
from wt.models.critic_state import expected_critic_dim
from wt.legacy.historical1.protocol import ALPHA_RESIDUAL
from wt.control.residual_policy import CentralizedCritic, ResidualActor
from wt.legacy.robust_lowlevel.audit import gae_audit, observation_audit
from wt.legacy.robust_lowlevel.protocol import EVAL_POOL, HELD_OUT_SOURCE, TRAIN_SCENARIOS, TRAIN_SOURCES
from wt.legacy.robust_lowlevel.worker import RobustTrainWorker


class _StubSelector:
    config_sha256 = "stub"

    def choose(self, env, source, rng, position_prev, predecessor):
        _ = (source, rng, position_prev, predecessor)
        assignment = np.arange(env.config.num_uavs, dtype=np.int64)
        option = make_option(env, assignment)
        return {"assignment": assignment, "option": option, "source": "scheme_c"}


class Phase6IS2RobustTests(unittest.TestCase):
    def test_observation_and_credit_keep_the_original_interface(self) -> None:
        self.assertEqual(float(ALPHA_RESIDUAL), 0.25)
        self.assertNotIn(HELD_OUT_SOURCE, TRAIN_SOURCES)
        self.assertTrue(set(TRAIN_SCENARIOS).isdisjoint(EVAL_POOL))
        self.assertTrue(set(TRAIN_SCENARIOS).isdisjoint(RESERVED_TEST))
        observation = observation_audit()
        self.assertFalse(observation["observation_changed"])
        self.assertTrue(observation["task_updates_with_new_option"])
        self.assertTrue(observation["communication_stays_on_committed_assignment"])
        self.assertEqual(observation["future_inputs"], 0)
        temporal = gae_audit()
        self.assertFalse(temporal["hold_is_terminal"])
        self.assertGreater(temporal["open_boundary_advantage"], temporal["terminal_advantage"])

    def test_hold_boundary_keeps_the_episode_and_the_step_reward(self) -> None:
        torch.set_num_threads(1)
        actor = ResidualActor()
        critic = CentralizedCritic(int(expected_critic_dim(6, 12)))
        worker = RobustTrainWorker(0, 0, actor, critic, _StubSelector(), 1)
        worker.reset()
        served = 0.0
        flags = []
        for _ in range(int(worker.env.config.episode_horizon)):
            record = worker.step()
            served += float(record["served_bits"])
            self.assertAlmostEqual(float(record["r_comm"]) * 1.0e6, float(record["served_bits"]), places=3)
            flags.append(record)
        decisions = worker.decisions[-4:]
        self.assertEqual([int(item["time"]) for item in decisions], [0, 15, 30, 45])
        self.assertAlmostEqual(sum(float(item["realized_served_bits"]) for item in decisions), served, places=3)
        self.assertTrue(any(int(item["time"]) == 15 and not item["done"] for item in flags))
        self.assertTrue(any(int(item["time"]) == 48 and item["done"] for item in flags))
        self.assertTrue(any(item["task_boundary"] and not item["done"] for item in flags))


if __name__ == "__main__":
    unittest.main()