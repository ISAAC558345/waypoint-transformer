"""Phase-9.4 masked autoregressive assignment decoder checks."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from wt.htw_phase9.const import IDLE_INDEX, MAX_BUOYS, MAX_UAVS
from wt.htw_phase9.features import build_graph, conflict_rate, feature_dict_l3b, stack_feature_dicts
from wt.htw_phase94.assignment_context import (
    all_candidate_physics,
    candidate_physics,
    legality_mask,
    make_snap,
    occupied_buoys,
    partial_vector,
    prefix_links,
)
from wt.htw_phase94.autoregressive_decoder import AssignmentARModel, decoder_parameter_count
from wt.htw_phase94.evaluation import decode_sequence
from wt.htw_phase94.oracle_sequence import sequence_to_assignment, serialize_oracle_assignment
from wt.htw_phase94.training import precompute_teacher_context, teacher_sequences
from wt.l3b.env import L3BEnv
from wt.l3b.policies import oracle_option_assignment


class Phase94Tests(unittest.TestCase):
    def test_01_oracle_sequence_deterministic(self) -> None:
        assignment = np.array([2, -1, 0, 7, -1, 3], dtype=np.int64)
        first = serialize_oracle_assignment(assignment)
        second = serialize_oracle_assignment(assignment)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(int(first[0]), 2)
        self.assertEqual(int(first[1]), IDLE_INDEX)
        self.assertTrue(np.array_equal(sequence_to_assignment(first, 6), assignment))

    def test_02_buoy_exclusivity_mask(self) -> None:
        partial = np.array([3, -1, -1, -1, -1, -1], dtype=np.int64)
        mask = legality_mask(partial, 1)
        self.assertFalse(bool(mask[3]))
        self.assertTrue(bool(mask[0]))
        self.assertTrue(bool(occupied_buoys(partial, 1, 12)[3]))

    def test_03_idle_always_legal(self) -> None:
        partial = np.array([0, 1, 2, 3, 4, -1], dtype=np.int64)
        mask = legality_mask(partial, 5)
        self.assertTrue(bool(mask[IDLE_INDEX]))

    def test_04_unavailable_edge_mask(self) -> None:
        avail = np.zeros(12, dtype=bool)
        avail[1] = True
        mask = legality_mask(np.full(6, -1, dtype=np.int64), 0, buoy_available=avail)
        self.assertTrue(bool(mask[1]))
        self.assertFalse(bool(mask[0]))
        self.assertTrue(bool(mask[IDLE_INDEX]))

    def test_05_partial_context_updates(self) -> None:
        empty = np.full(6, -1, dtype=np.int64)
        after = empty.copy()
        after[0] = 4
        left = partial_vector(empty, 1)
        right = partial_vector(after, 1)
        self.assertFalse(np.allclose(left, right))
        self.assertEqual(int(prefix_links(after, 1)[0][1]), 4)

    def test_06_sinr_context_depends_on_prefix(self) -> None:
        env = L3BEnv()
        env.reset(seed=0)
        snap = make_snap(env.uav_positions, env.buoy_positions, env.config)
        empty = np.full(6, -1, dtype=np.int64)
        prefix = empty.copy()
        prefix[0] = 0
        idle_empty = candidate_physics(snap, empty, 1, IDLE_INDEX)
        idle_after = candidate_physics(snap, prefix, 1, IDLE_INDEX)
        link_empty = candidate_physics(snap, empty, 1, 1)
        link_after = candidate_physics(snap, prefix, 1, 1)
        self.assertFalse(np.allclose(link_empty, link_after))
        changed = (not np.allclose(idle_empty, idle_after)) or (not np.allclose(link_empty, link_after))
        self.assertTrue(changed)

    def test_07_teacher_forcing_shapes(self) -> None:
        data = {
            "assignment": np.array([[0, -1, 1, -1, 6, -1]], dtype=np.int64),
            "uav_positions": np.zeros((1, 6, 2), dtype=np.float64),
            "buoy_positions": np.zeros((1, 12, 2), dtype=np.float64),
            "buoy_mask": np.ones((1, 12), dtype=bool),
        }
        ctx = precompute_teacher_context(data)
        self.assertEqual(tuple(ctx["physics"].shape), (1, MAX_UAVS, MAX_BUOYS + 1, 4))
        self.assertEqual(tuple(ctx["legal"].shape), (1, MAX_UAVS, MAX_BUOYS + 1))
        self.assertEqual(tuple(teacher_sequences(data["assignment"]).shape), (1, MAX_UAVS))

    def test_08_free_running_ignores_future_oracle(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=1)
        feats = stack_feature_dicts([feature_dict_l3b(info, env)])
        edge, edge_uu = build_graph(info, env)
        feats["edge"] = edge[None, ...]
        feats["edge_uu"] = edge_uu[None, ...]
        model = AssignmentARModel("deepsets")
        model.eval()
        oracle = serialize_oracle_assignment(oracle_option_assignment(env))
        with torch.no_grad():
            fr = decode_sequence(model, feats, env, teacher_tokens=None)
            tf = decode_sequence(model, feats, env, teacher_tokens=oracle)
        self.assertEqual(fr["tokens"].shape[0], MAX_UAVS)
        # Free-running path does not require Oracle tokens; TF uses them only as prefixes.
        self.assertTrue(np.isfinite(tf["nll"]))

    def test_09_identical_decoder_structure(self) -> None:
        deep = AssignmentARModel("deepsets")
        hgnn = AssignmentARModel("hgnn")
        self.assertEqual(decoder_parameter_count(), sum(int(p.numel()) for p in deep.decoder.parameters()))
        self.assertEqual(decoder_parameter_count(), sum(int(p.numel()) for p in hgnn.decoder.parameters()))
        self.assertEqual(type(deep.decoder).__name__, type(hgnn.decoder).__name__)

    def test_10_conflict_zero_and_argmax_legal(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=2)
        feats = stack_feature_dicts([feature_dict_l3b(info, env)])
        edge, edge_uu = build_graph(info, env)
        feats["edge"] = edge[None, ...]
        feats["edge_uu"] = edge_uu[None, ...]
        model = AssignmentARModel("hgnn")
        model.eval()
        with torch.no_grad():
            decoded = decode_sequence(model, feats, env, teacher_tokens=None)
        self.assertEqual(conflict_rate(decoded["assignment"], 12), 0.0)
        self.assertEqual(float(decoded["conflict"]), 0.0)

    def test_11_no_nan_inf(self) -> None:
        env = L3BEnv()
        env.reset(seed=3)
        snap = make_snap(env.uav_positions, env.buoy_positions, env.config)
        physics = all_candidate_physics(snap, np.full(6, -1, dtype=np.int64), 0)
        self.assertTrue(np.isfinite(physics).all())
        model = AssignmentARModel("deepsets")
        uav = torch.zeros(1, 6, 8)
        buoy = torch.zeros(1, 12, 7)
        um = torch.ones(1, 6, dtype=torch.bool)
        bm = torch.ones(1, 12, dtype=torch.bool)
        glob = torch.zeros(1, 3)
        edge = torch.zeros(1, 6, 12, 3)
        uu = torch.zeros(1, 6, 6, 2)
        encoded = model.encode(uav, buoy, um, bm, glob, edge, uu)
        legal = torch.ones(1, 13, dtype=torch.bool)
        logits = model.step_logits(encoded, 0, torch.zeros(1, 20), torch.zeros(1, 13, 4), legal)
        self.assertTrue(torch.isfinite(logits).all())

    def test_12_inference_deterministic(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=4)
        feats = stack_feature_dicts([feature_dict_l3b(info, env)])
        edge, edge_uu = build_graph(info, env)
        feats["edge"] = edge[None, ...]
        feats["edge_uu"] = edge_uu[None, ...]
        model = AssignmentARModel("deepsets")
        model.eval()
        with torch.no_grad():
            a = decode_sequence(model, feats, env, teacher_tokens=None)["tokens"]
            b = decode_sequence(model, feats, env, teacher_tokens=None)["tokens"]
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
