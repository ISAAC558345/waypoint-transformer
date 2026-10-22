"""Phase-6G interface checks. The 300-update run is not repeated here."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL, TRAIN_POOL, assert_allowed, choose_gate
from wt.benchmark_v2.phase6g_lowlevel_mappo.train import probe_initialization
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import TrainWorker, low_level_reward
from wt.lowlevel_v2.critic_state import expected_critic_dim
from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor


class Phase6GTests(unittest.TestCase):
    def test_pools_do_not_touch_reserved_or_benchmark_ids(self) -> None:
        for scenario in (320100, 320131, 320000, 320063, 310118):
            with self.assertRaises(RuntimeError):
                assert_allowed(scenario, split="train")
        assert_allowed(321000, split="train")
        assert_allowed(321096, split="dev")
        self.assertFalse(set(TRAIN_POOL) & set(DEV_POOL))
        self.assertFalse(set(TRAIN_POOL) & set(range(320100, 320132)))

    def test_legacy_actor_loads_and_reward_uses_served_bits(self) -> None:
        self.assertEqual(probe_initialization(), "LEGACY_WARM_START")
        reward, r_comm = low_level_reward(2.0e6, np.zeros((6, 2)), np.zeros((6, 2)), np.zeros((6, 2)), 1.0e6)
        self.assertAlmostEqual(r_comm, 2.0)
        self.assertAlmostEqual(reward, 2.0)

    def test_one_step_keeps_assignment_and_accounting(self) -> None:
        actor = ResidualActor(log_std_init=-2.2)
        critic = CentralizedCritic(expected_critic_dim(6, 12))
        worker = TrainWorker(0, 0, actor, critic)
        worker.reset()
        before = None
        record = worker.step()
        self.assertTrue(np.isfinite(record["reward"]))
        self.assertEqual(int(worker.env.assignment.shape[0]), 6)
        self.assertLessEqual(float(np.max(np.abs(worker.env.conservation_error_bits()))), float(worker.env.tol_bits))
        self.assertEqual(record["policy_active_mask"].shape, (6,))
        _ = before

    def test_gate_priority(self) -> None:
        failed = choose_gate(
            accounting_failures=1,
            interface_failures=0,
            nan_collapse=True,
            projection_rate=1.0,
            safety_regression=True,
            mean_delta=10.0,
            ci_low=1.0,
            positive_fraction=1.0,
            positive_seeds=5,
        )
        self.assertEqual(failed["gate"], "LL-E")
        unstable = choose_gate(
            accounting_failures=0,
            interface_failures=0,
            nan_collapse=False,
            projection_rate=0.0,
            safety_regression=False,
            mean_delta=10.0,
            ci_low=-1.0,
            positive_fraction=0.9,
            positive_seeds=5,
        )
        self.assertEqual(unstable["gate"], "LL-B")

    def test_package_does_not_call_decide_or_j15(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6g_lowlevel_mappo"
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("decide(", text)
            self.assertNotIn("batch_j15", text)
            self.assertNotIn("planner_ready = True", text)


if __name__ == "__main__":
    unittest.main()
