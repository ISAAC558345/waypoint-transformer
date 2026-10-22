"""Phase-9.4.5 unit tests. No confirmation leakage checks on frozen constants."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from pathlib import Path
from wt.htw_phase9.const import EVAL_SEEDS, HOLD
from wt.htw_phase92.run_phase92 import COLLECT_SEEDS
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.permutation_audit import candidate_order_invariance
from wt.htw_phase943.protocol import AUDIT_SEEDS
from wt.htw_phase943.set_context import make_snap
from wt.htw_phase945.candidate_pool import generate_pool, is_legal
from wt.htw_phase945.protocol import (
    BLOCKED_SCENARIO_SEEDS,
    LAMBDA_RANK,
    LAMBDA_REF,
    POOL_SIZE,
    SCAN_START,
    TAU,
)
from wt.htw_phase945.utility_loss import utility_losses
from wt.htw_phase945.utility_rollout import shadow_h15_payload
from wt.htw_phase945.utility_targets import ranking_pairs, soft_targets
from wt.l3b.env import L3BEnv


class Phase945Tests(unittest.TestCase):
    def test_01_pool_deterministic(self) -> None:
        env_a = L3BEnv()
        env_b = L3BEnv()
        env_a.reset(seed=0)
        env_b.reset(seed=0)
        a = generate_pool(env_a, 0)
        b = generate_pool(env_b, 0)
        self.assertEqual(a.shape[0], POOL_SIZE)
        self.assertTrue(np.array_equal(a, b))

    def test_02_pool_model_independent(self) -> None:
        env = L3BEnv()
        env.reset(seed=1)
        pool = generate_pool(env, 1)
        self.assertEqual(pool.shape, (POOL_SIZE, 6))
        OrderFreeAssignmentModel("deepsets")
        OrderFreeAssignmentModel("hgnn")
        self.assertTrue(np.array_equal(pool, generate_pool(env, 1)))

    def test_03_candidates_legal(self) -> None:
        env = L3BEnv()
        env.reset(seed=2)
        pool = generate_pool(env, 2)
        for row in pool:
            self.assertTrue(is_legal(row, env.config.num_buoys))

    def test_04_utility_deterministic(self) -> None:
        env_a = L3BEnv()
        env_b = L3BEnv()
        env_a.reset(seed=3)
        env_b.reset(seed=3)
        pool = generate_pool(env_a, 3)
        ua = shadow_h15_payload(env_a, pool[0])
        ub = shadow_h15_payload(env_b, pool[0])
        self.assertEqual(ua, ub)

    def test_05_normalization_and_softmax(self) -> None:
        q = soft_targets(np.asarray([0.0, 10.0, 20.0, 10.0]))
        self.assertAlmostEqual(float(q.sum()), 1.0, places=6)
        self.assertGreater(q[2], q[0])

    def test_06_soft_target_sum(self) -> None:
        q = soft_targets(np.ones(POOL_SIZE))
        self.assertAlmostEqual(float(q.sum()), 1.0, places=6)

    def test_07_near_tie_skipped(self) -> None:
        u = np.asarray([100.0, 100.1, 0.0])
        pairs = ranking_pairs(u)
        tied = [(i, j) for i, j, _, _ in pairs if {i, j} == {0, 1}]
        self.assertEqual(tied, [])

    def test_08_pairwise_sign(self) -> None:
        u = np.asarray([0.0, 1000.0])
        pairs = ranking_pairs(u)
        self.assertEqual(len(pairs), 1)
        i, j, y, w = pairs[0]
        self.assertEqual((i, j), (0, 1))
        self.assertEqual(y, -1.0)
        self.assertGreater(w, 0.0)

    def test_09_losses_finite(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=4)
        from wt.htw_phase9.features import build_graph, feature_dict_l3b, stack_feature_dicts
        from wt.htw_phase943.evaluation import encode_env

        feats = stack_feature_dicts([feature_dict_l3b(info, env)])
        edge, edge_uu = build_graph(info, env)
        feats["edge"] = edge[None, ...]
        feats["edge_uu"] = edge_uu[None, ...]
        model = OrderFreeAssignmentModel("deepsets")
        model.eval()
        encoded = encode_env(model, feats)
        snap = make_snap(env.uav_positions, env.buoy_positions, env.config)
        pool = generate_pool(env, 4)
        u = np.linspace(0, 1000, POOL_SIZE)
        packed = utility_losses(model, encoded, snap, pool, u, pool[0], feats["buoy_mask"][0])
        self.assertTrue(torch.isfinite(packed["utility"]))
        self.assertTrue(torch.isfinite(packed["rank"]))
        self.assertTrue(torch.isfinite(packed["ref"]))
        self.assertTrue(torch.isfinite(packed["total"]))

    def test_10_lambda_weights(self) -> None:
        self.assertAlmostEqual(LAMBDA_RANK, 0.25)
        self.assertAlmostEqual(LAMBDA_REF, 0.10)
        self.assertAlmostEqual(TAU, 1.0)
        self.assertEqual(HOLD, 15)

    def test_11_no_confirmation_in_train_split(self) -> None:
        for seed in list(COLLECT_SEEDS) + list(EVAL_SEEDS):
            self.assertNotIn(int(seed), set(range(24, 40)))
            self.assertLess(int(seed), 24)

    def test_12_blocked_covers_prior_confirmations(self) -> None:
        blocked = set(BLOCKED_SCENARIO_SEEDS)
        self.assertTrue(set(range(0, 1321)).issubset(blocked))
        self.assertGreaterEqual(SCAN_START, 1321)
        self.assertNotIn(SCAN_START, blocked)

    def test_13_checkpoint_rule_ignores_confirmation_payload(self) -> None:
        from wt.htw_phase945.training import train_utility_model

        src = Path(train_utility_model.__code__.co_filename).read_text(encoding="utf-8")
        self.assertIn("top1_regret", src)
        self.assertNotIn("confirmation", src.lower())

    def test_14_candidate_order_invariance_random_decoder(self) -> None:
        model = OrderFreeAssignmentModel("deepsets")
        model.eval()
        row = candidate_order_invariance(model, AUDIT_SEEDS[:4], n_shuffle=8)
        self.assertGreaterEqual(row["candidate_order_invariance_rate"], 0.999)

    def test_15_hold_aligned_only(self) -> None:
        self.assertEqual(HOLD, 15)


if __name__ == "__main__":
    unittest.main()
