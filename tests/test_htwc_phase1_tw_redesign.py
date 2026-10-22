"""HTW-C Phase-1 Task-Waypoint representation tests. No HGNN, no formal PPO training."""
from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import numpy as np
import torch

from wt.htwc_phase1.adapter import CONDITIONED_UAV_DIM, attach_task_waypoint
from wt.htwc_phase1.conditioned_actor import GoalConditionedActorCritic
from wt.htwc_phase1.contract import DEFAULT_HORIZON, GOAL_DIM, TaskWaypointV2, validate_tw
from wt.htwc_phase1.labels import collect_planner_window, labels_from_planner_window
from wt.htwc_phase1.predictor import TaskWaypointV2Predictor, TWPredictorConfig, task_waypoint_v2_loss
from wt.round19.scenarios import materialize_scenario, planning_scenarios
from wt.round30.features import MAX_BUOYS, MAX_UAVS, UAV_DIM, feature_dict
from wt.round30.models import DeepSetsActorCritic
from wt.round30.ppo import PPOAgent, PPOConfig


ROOT = Path(__file__).resolve().parents[1]
ROUND21_MODELS = ROOT / "src" / "wt" / "round21" / "models.py"
ROUND21_MODELS_SHA256 = hashlib.sha256(ROUND21_MODELS.read_bytes()).hexdigest().upper()


class HTWPhase1TaskWaypointTests(unittest.TestCase):
    def test_01_does_not_modify_hgnn_or_graph_layer_source(self) -> None:
        text = ROUND21_MODELS.read_text(encoding="utf-8")
        self.assertIn("class GraphLayer", text)
        self.assertIn("class HeterogeneousTaskWaypointPredictor", text)
        self.assertEqual(hashlib.sha256(ROUND21_MODELS.read_bytes()).hexdigest().upper(), ROUND21_MODELS_SHA256)

    def test_02_synthetic_planner_window_builds_goal_and_subgoal(self) -> None:
        scenario = planning_scenarios()[0]
        env, _, info = materialize_scenario(scenario)
        horizon = DEFAULT_HORIZON
        m_count, k_count = env.num_uavs, env.num_buoys
        service = np.zeros((horizon, m_count, k_count), dtype=np.float64)
        service[:, 0, 0] = 1.0
        service[:, 1, min(1, k_count - 1)] = 2.0
        start = np.asarray(info["uav_positions_m"], dtype=np.float64)
        end = start + np.array(((8.0, -3.0), (-4.0, 6.0)))
        sample = labels_from_planner_window(service, start, end, config=env.config, horizon=horizon)
        self.assertEqual(sample.goal_distribution.shape, (MAX_UAVS, GOAL_DIM))
        self.assertAlmostEqual(float(sample.goal_distribution[0, 0]), 1.0, places=5)
        self.assertEqual(int(sample.horizon), horizon)
        self.assertEqual(sample.waypoint_xy.shape, (MAX_UAVS, 2))

    def test_03_collects_labels_from_existing_independent_planner(self) -> None:
        scenario = planning_scenarios()[0]
        env, _, info = materialize_scenario(scenario)
        sample, later = collect_planner_window(env, info, horizon=DEFAULT_HORIZON, policy="independent_queue_rate")
        validate_tw(sample)
        self.assertEqual(int(later["time_step"]), DEFAULT_HORIZON)
        self.assertTrue(np.all(np.isfinite(sample.waypoint_xy)))

    def test_04_predictor_heads_match_tw_new_contract(self) -> None:
        scenario = planning_scenarios()[0]
        env, _, info = materialize_scenario(scenario)
        features = feature_dict(info, env.config)
        batch = {key: torch.as_tensor(value[None, ...], dtype=torch.bool if key.endswith("mask") else torch.float32) for key, value in features.items()}
        model = TaskWaypointV2Predictor(TWPredictorConfig())
        output = model(batch["uav"], batch["buoy"], batch["uav_mask"], batch["buoy_mask"], batch["global"])
        self.assertEqual(tuple(output["goal_distribution"].shape), (1, MAX_UAVS, MAX_BUOYS))
        self.assertEqual(tuple(output["waypoint_xy"].shape), (1, MAX_UAVS, 2))
        goal = output["goal_distribution"][0, 0, : env.num_buoys].detach()
        self.assertAlmostEqual(float(goal.sum()), 1.0, places=4)

    def test_05_v2_loss_is_goal_ce_plus_waypoint_mse(self) -> None:
        torch.manual_seed(0)
        model = TaskWaypointV2Predictor(TWPredictorConfig(waypoint_lambda=1.0))
        uav = torch.zeros(2, MAX_UAVS, UAV_DIM)
        buoy = torch.zeros(2, MAX_BUOYS, 7)
        uav_mask = torch.ones(2, MAX_UAVS, dtype=torch.bool)
        buoy_mask = torch.zeros(2, MAX_BUOYS, dtype=torch.bool)
        buoy_mask[:, :3] = True
        global_ = torch.zeros(2, 3)
        prediction = model(uav, buoy, uav_mask, buoy_mask, global_)
        goal = torch.zeros(2, MAX_UAVS, MAX_BUOYS)
        goal[:, :, 0] = 1.0
        waypoint = torch.zeros(2, MAX_UAVS, 2)
        goal_valid = torch.ones(2, MAX_UAVS, dtype=torch.bool)
        loss, parts = task_waypoint_v2_loss(prediction, goal, waypoint, uav_mask, goal_valid, waypoint_lambda=1.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("goal_loss", parts)
        self.assertIn("waypoint_loss", parts)
        loss.backward()
        grad_norm = sum(float(p.grad.detach().square().sum()) for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_norm, 0.0)

    def test_06_ppo_adapter_keeps_original_algorithm_and_action_shape(self) -> None:
        scenario = planning_scenarios()[0]
        env, _, info = materialize_scenario(scenario)
        features = feature_dict(info, env.config)
        goal = np.zeros((MAX_UAVS, MAX_BUOYS), dtype=np.float32)
        goal[0, 0] = 1.0
        valid = np.zeros(MAX_UAVS, dtype=bool)
        valid[0] = True
        tw = TaskWaypointV2(
            goal_distribution=goal,
            waypoint_xy=np.zeros((MAX_UAVS, 2), dtype=np.float32),
            horizon=DEFAULT_HORIZON,
            goal_valid_uav_mask=valid,
            uav_mask=features["uav_mask"],
            buoy_mask=features["buoy_mask"],
        )
        conditioned = attach_task_waypoint(features, tw)
        self.assertEqual(conditioned["uav"].shape[-1], CONDITIONED_UAV_DIM)
        self.assertEqual(set(conditioned), {"uav", "buoy", "uav_mask", "buoy_mask", "global"})
        state_only = DeepSetsActorCritic()
        self.assertFalse(state_only.uses_task_waypoint)
        agent = PPOAgent(GoalConditionedActorCritic(), PPOConfig(device="cpu"))
        batched = {key: value[None, ...] for key, value in conditioned.items()}
        action = agent.act(batched, deterministic=True)["action"]
        self.assertEqual(tuple(action.shape), (1, MAX_UAVS, 2))
        source = (ROOT / "src" / "wt" / "round30" / "ppo.py").read_text(encoding="utf-8")
        self.assertIn("compute_gae", source)
        self.assertIn("clip_eps", source)

    def test_07_baselines_are_interface_distinct(self) -> None:
        self.assertFalse(DeepSetsActorCritic().uses_task_waypoint)
        self.assertTrue(GoalConditionedActorCritic().uses_task_waypoint)
        self.assertNotEqual(GoalConditionedActorCritic().shape.uav_dim, DeepSetsActorCritic().shape.uav_dim)

    def test_08_three_validation_baselines_have_distinct_input_width(self) -> None:
        from wt.htwc_phase1.validation import interface_shapes

        shapes = interface_shapes()
        self.assertEqual(shapes["state_only"], 8)
        self.assertEqual(shapes["tw_old"], 19)
        self.assertEqual(shapes["htw_new"], 16)


if __name__ == "__main__":
    unittest.main()
