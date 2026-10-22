"""Assignment-conditioned head checks that do not open reserved scenarios."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, TRAIN_POOL, VALIDATION_POOL
from wt.benchmark_v2.phase6is_assignment_head.features import assignment_raw_features
from wt.benchmark_v2.phase6is_assignment_head.head import AssignmentConditionedHead
from wt.benchmark_v2.phase6is_assignment_head.loss import pairwise_rank_loss
from wt.benchmark_v2.phase6is_assignment_head.protocol import RAW_NAMES, SEEDS, assign_gate, seed_supported
from wt.lowlevel_v2.phase5k_joint_decoder.loss import joint_rank_loss


def _arrays(config) -> dict[str, np.ndarray]:
    uavs = int(config.num_uavs)
    buoys = int(config.num_buoys)
    positions = np.stack((np.linspace(10.0, 70.0, uavs), np.full(uavs, 40.0)), axis=1)
    buoy_positions = np.stack((np.linspace(20.0, 140.0, buoys), np.full(buoys, 80.0)), axis=1)
    distance = np.linalg.norm(positions[:, None, :] - buoy_positions[None, :, :], axis=-1)
    return {
        "uav_position_m": positions,
        "buoy_position_m": buoy_positions,
        "uav_buoy_distance_m": distance,
        "uav_buoy_gain": np.maximum(distance, float(config.min_link_m)) ** (-float(config.path_loss_exp)),
        "uav_buoy_in_range": (distance <= float(config.service_radius_m)).astype(np.uint8),
        "uav_buoy_hypothetical_rate": np.full((uavs, buoys), 1.0e5),
        "q_bits": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0e6, 4.0e6, 4.0e6, 4.0e6, 4.0e6, 4.0e6]),
        "association": np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        "service_progress": np.linspace(0.0, 0.5, uavs),
        "uav_uav_distance_m": np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1),
    }


class Phase6ISAssignmentHeadTests(unittest.TestCase):
    def test_pools_do_not_include_reserved_scenarios(self) -> None:
        self.assertTrue(set(TRAIN_POOL).isdisjoint(RESERVED_TEST))
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))
        self.assertEqual(len(SEEDS), 2)

    def test_chunked_pairwise_loss_matches_the_epoch6_loss(self) -> None:
        scores = torch.tensor([0.2, -0.4, 0.7, 0.0, 0.1], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([5.0, 1.0, 9.0, 1.0, 4.0], dtype=torch.float64)
        left = pairwise_rank_loss(scores, labels, chunk=2)
        right = joint_rank_loss(scores, labels)
        self.assertLess(abs(float(left.detach()) - float(right.detach())), 1.0e-8)

    def test_identical_assignments_and_uav_permutation_keep_the_score(self) -> None:
        torch.manual_seed(0)
        head = AssignmentConditionedHead().eval()
        assignments = torch.tensor([[6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11], [11, 10, 9, 8, 7, 6]], dtype=torch.long)
        h_u = torch.randn(6, 128)
        h_b = torch.randn(12, 128)
        edge_h = torch.randn(6, 12, 128)
        raw = torch.randn(1, 6, len(RAW_NAMES)).repeat(3, 1, 1)
        with torch.no_grad():
            scores = head(h_u, h_b, edge_h, raw, assignments)
            perm = torch.tensor([2, 0, 1, 4, 3, 5])
            permuted = head(h_u[perm], h_b, edge_h[perm], raw[:, perm], assignments[:, perm])
        self.assertLess(abs(float(scores[0] - scores[1])), 1.0e-6)
        self.assertGreater(abs(float(scores[0] - scores[2])), 1.0e-5)
        self.assertLess(float((scores - permuted).abs().max()), 1.0e-5)

    def test_swap_changes_the_selected_edge_token(self) -> None:
        torch.manual_seed(1)
        head = AssignmentConditionedHead().eval()
        h_u = torch.randn(6, 128)
        h_b = torch.randn(12, 128)
        edge_h = torch.randn(6, 12, 128)
        raw = torch.zeros(1, 6, len(RAW_NAMES))
        base = torch.tensor([[9, 6, 10, 8, 11, 7]], dtype=torch.long)
        swapped = torch.tensor([[9, 10, 6, 8, 11, 7]], dtype=torch.long)
        with torch.no_grad():
            before = head.tokens(h_u, h_b, edge_h, raw, base)
            after = head.tokens(h_u, h_b, edge_h, raw, swapped)
        self.assertGreater(float((before[0, 1, 256:384] - after[0, 1, 256:384]).abs().sum()), 0.0)
        self.assertLess(float((before[0, 0, 256:384] - after[0, 0, 256:384]).abs().max()), 1.0e-8)

    def test_feature_permutation_follows_the_uav_axis(self) -> None:
        config = L3BBitQueueEnvV2().config
        arrays = _arrays(config)
        assignment = np.array([[6, 7, 8, 9, 10, 11], [7, 6, 8, 9, 10, 11]], dtype=np.int64)
        original = assignment_raw_features(arrays, assignment, config)
        perm = np.array([1, 0, 2, 3, 4, 5])
        moved = {key: np.array(value, copy=True) for key, value in arrays.items()}
        for key in ("uav_position_m", "association", "service_progress", "uav_buoy_distance_m", "uav_buoy_gain", "uav_buoy_in_range", "uav_buoy_hypothetical_rate"):
            moved[key] = arrays[key][perm]
        moved["uav_uav_distance_m"] = arrays["uav_uav_distance_m"][perm][:, perm]
        permuted = assignment_raw_features(moved, assignment[:, perm], config)
        self.assertLess(float(np.max(np.abs(permuted - original[:, perm]))), 1.0e-6)
        self.assertEqual(original.shape[-1], len(RAW_NAMES))
        self.assertGreater(float(np.max(np.abs(original[0] - original[1]))), 0.0)

    def test_gate_rejects_a_mean_only_improvement(self) -> None:
        empty = {"supported": False}
        self.assertEqual(assign_gate(True, True, [empty, empty]), "ASSIGNMENT_CONDITIONED_REPRESENTATION_BLOCKED")
        self.assertEqual(assign_gate(False, False, [empty, empty]), "ASSIGNMENT_CONDITIONED_REPRESENTATION_BLOCKED")
        self.assertEqual(assign_gate(False, True, [empty, {"supported": True}]), "ASSIGNMENT_CONDITIONED_REPRESENTATION_INCONCLUSIVE")
        passed = {"supported": True}
        self.assertEqual(assign_gate(False, True, [passed, passed]), "ASSIGNMENT_CONDITIONED_REPRESENTATION_SUPPORTED")
        report = {name: False for name in (
            "top1_above_epoch6",
            "top1_above_scheme_c",
            "t15_top1_above_epoch6",
            "t15_top1_above_scheme_c",
            "mean_ci_above_epoch6",
            "mean_ci_above_scheme_c",
            "median_above_epoch6",
            "median_above_scheme_c",
            "t15_mean_ci_above_epoch6",
            "t15_mean_ci_above_scheme_c",
            "scenarios_consistent_epoch6",
            "scenarios_consistent_scheme_c",
        )}
        report["mean_ci_above_epoch6"] = True
        report["mean_ci_above_scheme_c"] = True
        self.assertFalse(seed_supported(report))
