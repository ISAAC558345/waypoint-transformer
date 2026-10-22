"""Phase-5 interface checks. They do not run the bounded probe."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase100.contract import TaskOption
from wt.htw_phase9.const import BUOY_DIM, EDGE_UB_DIM, EDGE_UU_DIM, UAV_DIM
from wt.htw_phase943.order_free_decoder import OrderFreeSetDecoder
from wt.l3b.env import L3BEnv

from wt.lowlevel_v2.phase5_hgnn_urgency.actor import ORIGINAL_OBS_DIM, URGENCY_OBS_DIM, UrgencyResidualActor
from wt.lowlevel_v2.phase5_hgnn_urgency.encoder import Phase5HGNNEncoder, Phase5OrderFreeModel
from wt.lowlevel_v2.phase5_hgnn_urgency.features import graph_batch, observable_exclusive_assignment
from wt.lowlevel_v2.phase5_hgnn_urgency.protocol import FORBIDDEN_RANGES, assert_phase5_seed
from wt.lowlevel_v2.phase5_hgnn_urgency.reward import unchanged_step_reward
from wt.lowlevel_v2.phase5_hgnn_urgency.urgency import shared_task_urgency
from wt.lowlevel_v2.phase5_hgnn_urgency.worker import Phase5Worker, make_actor, make_critic


ROOT = Path(__file__).resolve().parents[1]


class Phase5InterfaceTests(unittest.TestCase):
    def test_urgency_formula_and_range(self) -> None:
        queue = np.array([0.0, 2.0, 4.0, 1.0], dtype=np.float64)
        urgency = shared_task_urgency(queue)
        np.testing.assert_allclose(urgency, queue / (4.0 + 1.0e-8))
        self.assertGreaterEqual(float(np.min(urgency)), 0.0)
        self.assertLessEqual(float(np.max(urgency)), 1.0)
        self.assertNotIn("remaining_arrivals", shared_task_urgency.__doc__)

    def test_forbidden_seeds(self) -> None:
        for low, high in FORBIDDEN_RANGES:
            with self.assertRaises(RuntimeError):
                assert_phase5_seed(low)
            with self.assertRaises(RuntimeError):
                assert_phase5_seed(high - 1)
        assert_phase5_seed(310000)

    def test_task_option_default_and_copy(self) -> None:
        assignment = np.arange(6, dtype=np.int64)
        option = TaskOption(assignment, 0, 15, np.ones(6, bool), np.ones(12, bool), assignment.copy())
        self.assertIsNone(option.urgency)
        urgency = np.linspace(0.0, 1.0, 12)
        option.urgency = urgency
        copied = option.copy()
        copied.urgency[1] = -1.0
        self.assertGreater(float(option.urgency[1]), 0.0)
        self.assertFalse(np.shares_memory(option.urgency, copied.urgency))

    def test_hgnn_encoder_shape_and_no_buoy_buoy(self) -> None:
        encoder = Phase5HGNNEncoder(buoy_dim=BUOY_DIM + 1)
        self.assertFalse(hasattr(encoder, "bb_msg"))
        uav = torch.zeros(1, 6, UAV_DIM)
        buoy = torch.zeros(1, 12, BUOY_DIM + 1)
        edge = torch.zeros(1, 6, 12, EDGE_UB_DIM)
        edge_uu = torch.zeros(1, 6, 6, EDGE_UU_DIM)
        uav_mask = torch.ones(1, 6, dtype=torch.bool)
        buoy_mask = torch.ones(1, 12, dtype=torch.bool)
        h_u, h_b, edge_h = encoder.encode_nodes(uav, buoy, uav_mask, buoy_mask, edge, edge_uu)
        self.assertEqual(tuple(h_u.shape), (1, 6, 128))
        self.assertEqual(tuple(h_b.shape), (1, 12, 128))
        self.assertEqual(tuple(edge_h.shape), (1, 6, 12, 128))

    def test_decoder_class_unchanged(self) -> None:
        model = Phase5OrderFreeModel(buoy_dim=BUOY_DIM)
        self.assertIsInstance(model.decoder, OrderFreeSetDecoder)

    def test_observation_width(self) -> None:
        self.assertEqual(ORIGINAL_OBS_DIM, 34)
        self.assertEqual(URGENCY_OBS_DIM, 35)
        actor = UrgencyResidualActor()
        self.assertEqual(actor.in_dim, 35)
        self.assertEqual(int(actor.mu.out_features), 2)

    def test_reward_formula_not_rewritten(self) -> None:
        source = (ROOT / "src" / "wt" / "lowlevel_v2" / "phase1b" / "controller.py").read_text(encoding="utf-8")
        high = (ROOT / "src" / "wt" / "htw_phase100" / "high_level.py").read_text(encoding="utf-8")
        self.assertIn("r_payload - LAMBDA_PROJECT * c_proj - LAMBDA_RESIDUAL * c_res", source)
        self.assertNotIn("urgency", source)
        self.assertIn("load_deepsets_util", high)
        self.assertNotIn("urgency", high)
        reward = unchanged_step_reward(1.0e6, np.zeros((6, 2)), np.zeros((6, 2)), np.zeros((6, 2)))
        self.assertGreater(reward, 0.0)

    def test_exclusive_label_and_urgency_channel(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=310000)
        label = observable_exclusive_assignment(env)
        chosen = [int(item) for item in label if int(item) >= 0]
        self.assertEqual(len(chosen), len(set(chosen)))
        last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        urgency = np.linspace(0.0, 1.0, env.config.num_buoys)
        batch = graph_batch(info, env, last_arrival, last_positions, urgency=urgency)
        self.assertEqual(int(batch["buoy"].shape[-1]), BUOY_DIM + 1)
        self.assertNotIn("remaining_arrivals", batch)

    def test_hold_does_not_redecide_or_recompute_urgency(self) -> None:
        class _Planner:
            def __init__(self) -> None:
                self.calls = 0

            def decide(self, env, info, last_arrival, last_positions):
                self.calls += 1
                label = observable_exclusive_assignment(env)
                urgency = shared_task_urgency(env.queue)
                return label, urgency

        planner = _Planner()
        actor = make_actor(True)
        critic = make_critic(True)
        worker = Phase5Worker(planner, actor, critic, use_urgency=True, seeds=(310000,))
        info = worker.reset(310000)
        info, _ = worker.step(info, deterministic=True)
        frozen = np.asarray(worker.option.urgency, dtype=np.float64).copy()
        self.assertEqual(planner.calls, 1)
        for _ in range(10):
            info, _ = worker.step(info, deterministic=True)
        np.testing.assert_allclose(worker.option.urgency, frozen)
        self.assertEqual(planner.calls, 1)


if __name__ == "__main__":
    unittest.main()
