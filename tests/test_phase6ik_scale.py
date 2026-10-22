"""Scale transfer keeps the raw-observation actor algebraically identical to divided inputs."""
from __future__ import annotations

import hashlib
import unittest

import numpy as np
import torch

from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.benchmark_v2.phase6ik_lowlevel_opt.protocol import EVAL_POOL, SOURCE_ACTOR, TRAIN_POOL
from wt.benchmark_v2.phase6ik_lowlevel_opt.scale import (
    actor_feature_scales,
    critic_feature_scales,
    fold_scales_into_actor,
    neighbor_edge_scales,
    scale_actor_observation,
)
from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.critic_state import expected_critic_dim
from wt.lowlevel_v2.residual_policy import ResidualActor


class Phase6IKScaleTests(unittest.TestCase):
    def test_pools_stay_away_from_the_reserved_test(self) -> None:
        self.assertTrue(set(EVAL_POOL).isdisjoint(RESERVED_TEST))
        self.assertTrue(set(TRAIN_POOL).isdisjoint(RESERVED_TEST))
        self.assertTrue(set(EVAL_POOL).isdisjoint(TRAIN_POOL))
        self.assertEqual(EVAL_POOL[0], 321096)
        self.assertEqual(SOURCE_ACTOR.resolve(), primary_checkpoint().resolve())

    def test_widths_match_the_existing_networks(self) -> None:
        self.assertEqual(actor_feature_scales().shape, (34,))
        self.assertEqual(neighbor_edge_scales().shape, (11,))
        self.assertEqual(critic_feature_scales().shape, (expected_critic_dim(6, 12),))
        self.assertTrue(np.all(actor_feature_scales() > 0.0))

    def test_folded_raw_forward_matches_scaled_input(self) -> None:
        torch.manual_seed(0)
        actor = ResidualActor(log_std_init=-2.2)
        actor.eval()
        obs = {
            "self": np.random.default_rng(1).normal(size=(6, 5)),
            "task": np.random.default_rng(2).normal(size=(6, 8)) * 20.0,
            "communication": np.random.default_rng(3).normal(size=(6, 4)) * np.array([10.0, 1.0e5, 0.01, 10.0]),
            "neighbor_set": np.random.default_rng(4).normal(size=(6, 5, 11)) * 15.0,
            "d_nearest": np.random.default_rng(5).random(6) * 40.0,
        }
        scaled = scale_actor_observation(obs, L3BConfig())
        with torch.no_grad():
            left, _left_std = actor(
                torch.as_tensor(scaled["self"], dtype=torch.float32),
                torch.as_tensor(scaled["task"], dtype=torch.float32),
                torch.as_tensor(scaled["communication"], dtype=torch.float32),
                torch.as_tensor(scaled["neighbor_set"], dtype=torch.float32),
                torch.as_tensor(scaled["d_nearest"], dtype=torch.float32),
            )
            folded = fold_scales_into_actor(actor)
            folded.eval()
            right, _right_std = folded(
                torch.as_tensor(obs["self"], dtype=torch.float32),
                torch.as_tensor(obs["task"], dtype=torch.float32),
                torch.as_tensor(obs["communication"], dtype=torch.float32),
                torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
                torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
            )
        self.assertTrue(torch.allclose(left, right, atol=1.0e-5, rtol=1.0e-5))
        digest = hashlib.sha256(primary_checkpoint().read_bytes()).hexdigest()
        self.assertEqual(digest, "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e")


if __name__ == "__main__":
    unittest.main()
