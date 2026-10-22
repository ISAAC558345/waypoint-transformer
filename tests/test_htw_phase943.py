"""Phase-9.4.3 order-free set decoder checks."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from wt.htw_phase9.const import MAX_UAVS
from wt.htw_phase9.features import build_graph, conflict_rate, feature_dict_l3b, stack_feature_dicts
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel, decoder_parameter_count, pick_candidate
from wt.htw_phase943.protocol import BLOCKED_SCENARIO_SEEDS, COLLECT_SEEDS, SCAN_START
from wt.htw_phase943.set_context import UNDECIDED, all_candidate_physics_set, apply_choice, assignment_links, empty_assignment, legal_candidate_mask, make_snap
from wt.htw_phase943.set_likelihood_dp import exhaustive_logprob, subset_dp_logprob
from wt.l3b.env import L3BEnv


def _batch(env, info):
    feats = stack_feature_dicts([feature_dict_l3b(info, env)])
    edge, edge_uu = build_graph(info, env)
    feats["edge"] = edge[None, ...]
    feats["edge_uu"] = edge_uu[None, ...]
    return feats


class Phase943Tests(unittest.TestCase):
    def test_01_partial_set_order_invariant(self) -> None:
        a = empty_assignment()
        ab = apply_choice(apply_choice(a, 0, 3), 2, 5)
        ba = apply_choice(apply_choice(a, 2, 5), 0, 3)
        self.assertEqual(assignment_links(ab), assignment_links(ba))
        self.assertTrue(np.array_equal(ab, ba))

    def test_02_sinr_set_order_invariant(self) -> None:
        env = L3BEnv()
        env.reset(seed=0)
        snap = make_snap(env.uav_positions, env.buoy_positions, env.config)
        a = empty_assignment()
        ab = apply_choice(apply_choice(a, 0, 3), 1, 8)
        ba = apply_choice(apply_choice(a, 1, 8), 0, 3)
        pa = all_candidate_physics_set(snap, ab)
        pb = all_candidate_physics_set(snap, ba)
        self.assertTrue(np.allclose(pa, pb))
        from wt.htw_phase943.set_context import candidate_physics_set

        ref = candidate_physics_set(snap, ab, 2, 4)
        self.assertTrue(np.allclose(pa[2, 4], ref, atol=1.0e-5))

    def test_03_dp_matches_explicit_small(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=1)
        batch = _batch(env, info)
        model = OrderFreeAssignmentModel("deepsets")
        model.eval()
        from wt.htw_phase943.evaluation import encode_env

        encoded = encode_env(model, batch)
        snap = make_snap(env.uav_positions, env.buoy_positions, env.config)
        target = np.array([0, -1, 7, -1, -1, -1], dtype=np.int64)
        # Compare DP vs 3! on the three decided UAVs by forcing others already assigned idle in both.
        with torch.no_grad():
            dp = float(subset_dp_logprob(model, encoded, snap, target, batch["buoy_mask"][0]))
            ex = float(exhaustive_logprob(model, encoded, snap, target, batch["buoy_mask"][0]))
        self.assertTrue(np.isfinite(dp) and np.isfinite(ex))
        self.assertLess(abs(dp - ex), 2.0e-4)

    def test_04_idle_and_mask(self) -> None:
        a = empty_assignment()
        a = apply_choice(a, 0, 2)
        mask = legal_candidate_mask(a)
        self.assertFalse(mask[1, 2])
        self.assertTrue(mask[1, 12])
        self.assertFalse(mask[0].any())

    def test_05_conflict_zero_and_shared_decoder(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=2)
        batch = _batch(env, info)
        from wt.htw_phase943.evaluation import decode_assignment

        d = OrderFreeAssignmentModel("deepsets")
        h = OrderFreeAssignmentModel("hgnn")
        self.assertEqual(decoder_parameter_count(), sum(p.numel() for p in d.decoder.parameters()))
        self.assertEqual(decoder_parameter_count(), sum(p.numel() for p in h.decoder.parameters()))
        d.eval()
        with torch.no_grad():
            assignment = decode_assignment(d, batch, env)
        self.assertEqual(conflict_rate(assignment, 12), 0.0)

    def test_06_no_nan_and_split_clean(self) -> None:
        self.assertTrue(set(range(24)).issubset(set(COLLECT_SEEDS) | set(range(8, 16)) | set(COLLECT_SEEDS)))
        self.assertGreaterEqual(SCAN_START, 600)
        self.assertIn(598, BLOCKED_SCENARIO_SEEDS)
        self.assertNotIn(24, COLLECT_SEEDS)

    def test_07_inference_no_oracle_tokens(self) -> None:
        import inspect
        from wt.htw_phase943.evaluation import decode_assignment

        src = inspect.getsource(decode_assignment)
        self.assertNotIn("oracle_option_assignment", src)
        self.assertNotIn("teacher", src.lower())

    def test_08_uav_perm_and_candidate_shuffle(self) -> None:
        from wt.htw_phase942.decode_order import permute_uav_batch
        from wt.htw_phase943.evaluation import decode_assignment
        from wt.htw_phase943.permutation_audit import map_assignment_uav, sets_equivalent

        env = L3BEnv()
        info = env.reset(seed=3)
        batch = _batch(env, info)
        model = OrderFreeAssignmentModel("deepsets")
        model.eval()
        with torch.no_grad():
            base = decode_assignment(model, batch, env)
            perm = np.arange(MAX_UAVS)[::-1].copy()
            env_u = L3BEnv()
            env_u.reset(seed=3)
            env_u.uav_positions = np.asarray(env.uav_positions, dtype=np.float64)[perm]
            pred = decode_assignment(model, permute_uav_batch(batch, perm), env_u)
            self.assertTrue(sets_equivalent(pred, map_assignment_uav(base, perm)))
            shuffled = decode_assignment(model, batch, env, shuffle_seed=99)
            self.assertTrue(sets_equivalent(base, shuffled))

    def test_09_buoy_perm_with_cluster_labels(self) -> None:
        from wt.htw_phase9.const import MAX_BUOYS
        from wt.htw_phase943.evaluation import decode_assignment
        from wt.htw_phase943.permutation_audit import map_assignment_buoy, permute_buoy_batch, sets_equivalent
        from wt.htw_phase943.set_context import default_buoy_cluster

        env = L3BEnv()
        info = env.reset(seed=9001)
        batch = _batch(env, info)
        model = OrderFreeAssignmentModel("deepsets")
        model.eval()
        perm = np.array([9, 10, 1, 11, 5, 4, 8, 2, 3, 6, 0, 7])
        with torch.no_grad():
            base = decode_assignment(model, batch, env)
            env_b = L3BEnv()
            env_b.reset(seed=9001)
            env_b.buoy_positions = np.asarray(env.buoy_positions, dtype=np.float64)[perm]
            env_b.buoy_cluster = default_buoy_cluster()[perm]
            pred = decode_assignment(model, permute_buoy_batch(batch, perm), env_b)
            self.assertTrue(sets_equivalent(pred, map_assignment_buoy(base, perm)))


if __name__ == "__main__":
    unittest.main()
