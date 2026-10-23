"""Reproduce termination and GAE semantics. This test does not train."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from wt.benchmark_v2.phase6d.shadow import capture_state
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import TrainWorker, low_level_reward
from wt.benchmark_v2.phase6ir_scheme_c_align.worker import SchemeCTrainWorker
from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.lowlevel_v2.critic_state import expected_critic_dim
from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor
from wt.round30.ppo import compute_gae


def _nets():
    torch.manual_seed(0)
    return ResidualActor(), CentralizedCritic(expected_critic_dim(6, 12))


def _zero_act(worker):
    def act(_obs):
        width = int(worker.env.config.num_uavs)
        return np.zeros((width, 2), dtype=np.float64), np.zeros(width, dtype=np.float32)

    worker._act = act


class TemporalCreditAuditTests(unittest.TestCase):
    def test_done_removes_the_following_reward_from_gae(self) -> None:
        rewards = np.array([[1.0], [1.0], [10.0]], dtype=np.float32)
        values = np.zeros_like(rewards)
        last = np.zeros(1, dtype=np.float32)
        cut = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
        keep = np.zeros_like(cut)
        cut_adv, _cut_ret = compute_gae(rewards, values, cut, last, 0.99, 1.0)
        keep_adv, _keep_ret = compute_gae(rewards, values, keep, last, 0.99, 1.0)
        self.assertAlmostEqual(float(cut_adv[1, 0]), 1.0, places=5)
        self.assertLess(float(cut_adv[0, 0]), 2.1)
        self.assertGreater(float(keep_adv[0, 0]), float(cut_adv[0, 0]) + 5.0)

    def test_reward_identity_is_the_current_step_only(self) -> None:
        zeros = np.zeros((6, 2), dtype=np.float64)
        reward, comm = low_level_reward(250_000.0, zeros, zeros, zeros, 1_000_000.0)
        self.assertAlmostEqual(comm * 1_000_000.0, 250_000.0, places=3)
        self.assertAlmostEqual(reward, comm, places=6)

    def test_phase6g_hold_boundary_continues_the_episode(self) -> None:
        from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import HOLD
        from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option

        actor, critic = _nets()
        worker = TrainWorker(0, 0, actor, critic)
        _zero_act(worker)
        assignment = np.arange(worker.env.config.num_uavs, dtype=np.int64)
        worker.reset()
        rows = []
        for _ in range(48):
            if worker.pending <= 0:
                worker.option = make_option(worker.env, assignment)
                worker.pending = int(HOLD)
            before = int(worker.env.t)
            record = worker.step()
            rows.append((before, bool(record["done"]), float(record["served_bits"]), float(record["r_comm"]), float(record["payload"])))
        boundary = [item for item in rows if item[0] in (14, 29, 44)]
        self.assertEqual([item[0] for item in boundary], [14, 29, 44])
        self.assertTrue(all(item[1] is False for item in boundary))
        self.assertTrue(rows[-1][1])
        self.assertAlmostEqual(rows[-1][2], rows[-1][3] * 1_000_000.0, places=3)
        self.assertAlmostEqual(sum(item[2] for item in rows), rows[-1][4], places=3)

    def test_phase6ir_window_end_is_terminal_and_reloads_another_window(self) -> None:
        actor, critic = _nets()
        env = L3BBitQueueEnvV2()
        env.reset(seed=321000)
        snap = capture_state(env)
        assignment = np.arange(env.config.num_uavs, dtype=np.int64)
        row = {
            "scenario_id": 321000,
            "time": 0,
            "buoy_positions": np.asarray(env.buoy_positions, dtype=np.float64).copy(),
            "snap": snap,
            "initial_q_bits": np.asarray(env.initial_q_bits, dtype=np.float64).copy(),
            "assignment": assignment,
            "last_movement": np.zeros((env.config.num_uavs, 2), dtype=np.float64),
            "window_steps": 2,
        }
        worker = SchemeCTrainWorker(0, 0, actor, critic)
        _zero_act(worker)
        worker.bind_records([row, dict(row)], 1)
        worker.reset()
        first = worker.step()
        second = worker.step()
        self.assertFalse(bool(first["done"]))
        self.assertTrue(bool(second["done"]))
        self.assertEqual(int(worker.steps_left), 2)
        self.assertAlmostEqual(float(first["r_comm"]) * 1_000_000.0, float(first["served_bits"]), places=3)
        self.assertGreater(int(worker.steps_left), 0)
        collector_bootstrap = 0.0 if second["done"] else float(worker.bootstrap_value())
        self.assertEqual(collector_bootstrap, 0.0)
        worker.steps_left = 0
        self.assertEqual(float(worker.bootstrap_value()), 0.0)


if __name__ == "__main__":
    unittest.main()
