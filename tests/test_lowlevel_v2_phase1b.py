"""LL-V2.1B pre-training tests. No extra hyperparameter changes."""
from __future__ import annotations

import hashlib
import unittest

import numpy as np
import torch

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.base_executor import BaseExecutor
from wt.lowlevel_v2.contracts import HOLD_STEPS, ResidualConfig, env_target
from wt.lowlevel_v2.phase1.controller import ResidualWorker as Phase1Worker
from wt.lowlevel_v2.phase1.controller import load_frozen_high, make_networks
from wt.lowlevel_v2.phase1b.controller import ResidualWorker as Phase1BWorker
from wt.lowlevel_v2.phase1b.mappo import ICARMAPPOLocked
from wt.lowlevel_v2.phase1b.protocol import EXPECTED_HIGH_SHA, FINAL_CKPT, PPO, refuse_untouched
from wt.lowlevel_v2.phase1b.service_lock import apply_service_residual_lock, service_lock_mask
from wt.lowlevel_v2.protocol import PHASE106_UNTOUCHED
from wt.lowlevel_v2.residual_action import residual_desired
from wt.round30.ppo import seed_everything


def _option(env: L3BEnv, assoc) -> TaskOption:
    assoc = np.asarray(assoc, dtype=np.int64)
    return TaskOption(assoc, int(env.t), HOLD_STEPS, np.ones(6, bool), np.ones(12, bool), assoc.copy())


def _place_in_range(env: L3BEnv, uav: int, buoy: int) -> None:
    env.uav_positions[uav] = env.buoy_positions[buoy].copy()


class LowLevelV2Phase1BTests(unittest.TestCase):
    def test_01_unlocked_matches_phase1_residual(self) -> None:
        rng = np.random.default_rng(0)
        a_base = rng.normal(size=(6, 2))
        delta = rng.normal(size=(6, 2))
        cfg = ResidualConfig(alpha_residual=0.25, placeholder=False)
        a1 = residual_desired(a_base, delta, cfg)
        lock = np.zeros(6, dtype=bool)
        a2, used = apply_service_residual_lock(a_base, delta, lock, 0.25)
        np.testing.assert_allclose(a1, a2)
        np.testing.assert_allclose(used, 0.25 * np.tanh(delta))

    def test_02_lock_zeros_delta_used(self) -> None:
        a_base = np.zeros((6, 2))
        delta = np.ones((6, 2))
        lock = np.array([True, False, True, False, False, False])
        a_des, used = apply_service_residual_lock(a_base, delta, lock, 0.25)
        np.testing.assert_allclose(used[lock], 0.0)
        self.assertGreater(float(np.linalg.norm(used[~lock])), 0.0)
        np.testing.assert_allclose(a_des[lock], 0.0, atol=1e-12)

    def test_03_lock_a_des_equals_zero_base(self) -> None:
        a_base = np.zeros((6, 2))
        delta = np.random.default_rng(1).normal(size=(6, 2))
        lock = np.ones(6, dtype=bool)
        a_des, used = apply_service_residual_lock(a_base, delta, lock, 0.25)
        np.testing.assert_allclose(a_des, 0.0, atol=1e-12)
        np.testing.assert_allclose(used, 0.0, atol=1e-12)
        np.testing.assert_allclose(a_base, 0.0, atol=1e-12)

    def test_04_idle_does_not_lock(self) -> None:
        env = L3BEnv()
        env.reset(seed=221000)
        option = _option(env, np.full(6, -1))
        self.assertFalse(np.any(service_lock_mask(env, option)))

    def test_05_queue_zero_does_not_lock(self) -> None:
        env = L3BEnv()
        env.reset(seed=221000)
        option = _option(env, np.arange(6))
        _place_in_range(env, 0, 0)
        env.queue[:] = 0.0
        self.assertFalse(bool(service_lock_mask(env, option)[0]))

    def test_06_invalid_target_does_not_lock(self) -> None:
        env = L3BEnv()
        env.reset(seed=221000)
        option = _option(env, np.full(6, 99))
        self.assertEqual(env_target(99, 12), -1)
        self.assertFalse(np.any(service_lock_mask(env, option)))

    def test_07_outside_region_does_not_lock(self) -> None:
        env = L3BEnv()
        env.reset(seed=221000)
        option = _option(env, np.arange(6))
        env.queue[:] = 10.0
        env.uav_positions[0] = np.array([0.0, 0.0])
        far = float(np.linalg.norm(env.uav_positions[0] - env.buoy_positions[0]))
        self.assertGreater(far, float(env.config.service_radius_m))
        self.assertFalse(bool(service_lock_mask(env, option)[0]))

    def test_08_policy_mask_zero_when_locked(self) -> None:
        seed_everything(0)
        high, _ = load_frozen_high()
        actor, critic = make_networks()
        worker = Phase1BWorker(high, actor, critic, service_lock=True)
        env = worker.env
        info = worker.reset(221000)
        worker._refresh_option(info)
        env.queue[:] = 10.0
        for uav in range(6):
            tgt = int(worker.option.association[uav])
            if tgt >= 0:
                _place_in_range(env, uav, tgt)
        lock = service_lock_mask(env, worker.option)
        self.assertTrue(np.any(lock))
        info, rec = worker.step(info, deterministic=True)
        np.testing.assert_array_equal(rec.policy_active_mask, (~rec.service_lock).astype(np.float32))
        if np.any(rec.service_lock):
            self.assertTrue(np.allclose(rec.policy_active_mask[rec.service_lock], 0.0))

    def test_09_value_loss_includes_locked(self) -> None:
        seed_everything(0)
        actor, critic = make_networks()
        agent = ICARMAPPOLocked(actor, critic, PPO)
        storage = _toy_storage(lock_all=True)
        stats = agent.update(storage)
        self.assertTrue(np.isfinite(stats["value_loss"]))
        self.assertGreater(stats["value_loss"], 0.0)

    def test_10_policy_loss_ignores_locked(self) -> None:
        seed_everything(0)
        actor, critic = make_networks()
        agent = ICARMAPPOLocked(actor, critic, PPO)
        storage = _toy_storage(lock_all=True)
        before = {k: v.detach().clone() for k, v in actor.named_parameters()}
        agent.update(storage)
        actor_changed = any(not torch.allclose(before[k], v) for k, v in actor.named_parameters())
        # value still updates critic; actor policy/entropy terms are 0 but value backward may still hit shared? Actor and critic are separate nets.
        self.assertFalse(actor_changed)

    def test_11_high_sha_unchanged(self) -> None:
        digest = hashlib.sha256(FINAL_CKPT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_HIGH_SHA)

    def test_12_untouched_not_visited(self) -> None:
        self.assertIn(106000, PHASE106_UNTOUCHED)
        with self.assertRaises(RuntimeError):
            refuse_untouched(106000)

    def test_13_in_range_queue_locks(self) -> None:
        env = L3BEnv()
        env.reset(seed=221000)
        option = _option(env, np.arange(6))
        env.queue[:] = 5.0
        _place_in_range(env, 0, 0)
        self.assertTrue(env._in_range(0, 0))
        self.assertTrue(bool(service_lock_mask(env, option)[0]))


def _toy_storage(*, lock_all: bool) -> dict:
    from wt.lowlevel_v2.critic_state import expected_critic_dim

    horizon, n_env, m = 48, 4, 6
    rng = np.random.default_rng(3)
    glob_dim = expected_critic_dim(6, 12)
    mask = np.zeros((horizon, n_env, m), dtype=np.float32) if lock_all else np.ones((horizon, n_env, m), dtype=np.float32)
    return {
        "self": rng.normal(size=(horizon, n_env, m, 5)).astype(np.float32),
        "task": rng.normal(size=(horizon, n_env, m, 8)).astype(np.float32),
        "communication": rng.normal(size=(horizon, n_env, m, 4)).astype(np.float32),
        "neighbor_set": rng.normal(size=(horizon, n_env, m, 5, 11)).astype(np.float32),
        "d_nearest": rng.random((horizon, n_env, m)).astype(np.float32),
        "global_state": rng.normal(size=(horizon, n_env, glob_dim)).astype(np.float32),
        "pre_tanh": rng.normal(size=(horizon, n_env, m, 2)).astype(np.float32) * 0.01,
        "log_prob_uav": rng.normal(size=(horizon, n_env, m)).astype(np.float32) * 0.01,
        "policy_active_mask": mask,
        "values": rng.normal(size=(horizon, n_env)).astype(np.float32),
        "rewards": rng.normal(size=(horizon, n_env)).astype(np.float32) * 0.1,
        "dones": np.zeros((horizon, n_env), dtype=np.float32),
        "last_values": np.zeros(n_env, dtype=np.float32),
    }


if __name__ == "__main__":
    unittest.main()
