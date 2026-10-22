import compileall
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.env import MaritimeEnv
from wt.round18.audit import single_uav_regression
from wt.round18.multi_env import MultiUAVMaritimeEnv, deterministic_matching, minimum_pairwise_distance, project_uav_separation
from wt.round18.planner import ACTION_CANDIDATE_NAMES, PLANNING_HORIZONS, independent_decision, joint_plan
from wt.round18.scenarios import SCENARIO_NAMESPACE, materialize_scenario, planning_scenarios


ROOT = Path(__file__).resolve().parents[1]


def config(k=3, **extra):
    values = dict(num_uavs=2, num_buoys=k, task_semantics_version="v1", buoy_drift_mode="static", channel_variation_mode="static", initial_uav_positions_m=((85.0, 90.0), (115.0, 90.0)), initial_queue_bits_by_buoy=tuple(2.5e6 for _ in range(k)), terminate_on_task_success=False, episode_horizon=8)
    values.update(extra)
    return MaritimeEnvConfig(**values)


class Round18MultiUAVTests(unittest.TestCase):
    def test_01_round17_lock_unchanged(self): self.assertEqual(hashlib.sha256((ROOT / "round_17/ROUND_LOCK.json").read_bytes()).hexdigest().upper(), "A393148768835CB28296999D73490C77CAD51279B8060E9B99D10EC8581285FD")
    def test_02_no_optimizer(self): self.assertNotIn("optimizer", (ROOT / "src/wt/round18").joinpath("planner.py").read_text())
    def test_03_no_backward(self): self.assertNotIn("backward", (ROOT / "src/wt/round18").joinpath("planner.py").read_text())
    def test_04_no_neural_checkpoint(self): self.assertFalse(list((ROOT / "src/wt/round18").glob("*.pt")))
    def test_05_m1_exact_regression(self): self.assertTrue(all(row["passed"] for row in single_uav_regression()))
    def test_06_configurable_num_uavs(self): self.assertEqual(MaritimeEnvConfig(num_uavs=4).num_uavs, 4)
    def test_07_joint_action_shape(self): self.assertEqual(MultiUAVMaritimeEnv(config()).action_shape, (2, 2))
    def test_08_per_uav_speed_bound(self):
        env = MultiUAVMaritimeEnv(config()); env.reset(); _, _, _, _, info = env.step(np.array(((5., 0.), (0., -5.))))
        self.assertTrue(np.all(info["movement_distance_m_per_uav"] <= 10.0 + 1e-9))
    def test_09_boundary(self):
        env = MultiUAVMaritimeEnv(config(initial_uav_positions_m=((1., 1.), (199., 199.)))); env.reset(); _, _, _, _, info = env.step(np.array(((-1., -1.), (1., 1.))))
        self.assertTrue(np.all(info["uav_positions_m"] >= 0.0) and np.all(info["uav_positions_m"] <= 200.0))
    def test_10_one_shared_queue(self): self.assertEqual(MultiUAVMaritimeEnv(config()).reset()[1]["raw_queue_bits"].shape, (3,))
    def test_11_no_duplicate_payload(self):
        env = MultiUAVMaritimeEnv(config()); _, info = env.reset(); _, _, _, _, after = env.step(np.zeros((2, 2)))
        self.assertTrue(np.all(after["transmitted_payload_bits"] <= after["queue_before_bits"] + 1e-6))
    def test_12_matching_uav_at_most_one(self): self.assertLessEqual(np.count_nonzero(deterministic_matching(np.ones(3), np.ones((2, 3)), 1.0)[0] >= 0), 2)
    def test_13_matching_buoy_at_most_one(self):
        assignment, _, _ = deterministic_matching(np.ones(3), np.ones((2, 3)), 1.0); self.assertEqual(len(set(assignment[assignment >= 0])), np.count_nonzero(assignment >= 0))
    def test_14_matching_tie_break(self): self.assertTrue(np.array_equal(deterministic_matching(np.ones(3), np.ones((2, 3)), 1.0)[0], np.array((0, 1))))
    def test_15_variable_k(self):
        for k in (3, 5): self.assertEqual(MultiUAVMaritimeEnv(config(k)).reset()[1]["raw_queue_bits"].shape, (k,))
    def test_16_collision_distance(self):
        p, _, _, distance = project_uav_separation(np.array(((100., 100.), (105., 100.))), 15., 200., 200.); self.assertGreaterEqual(distance, 15. - 1e-6)
    def test_17_symmetric_joint_safety(self):
        source = np.array(((95., 100.), (105., 100.))); first = project_uav_separation(source, 15., 200., 200.)[0]; second = project_uav_separation(source[::-1], 15., 200., 200.)[0][::-1]; self.assertTrue(np.allclose(first, second))
    def test_18_primary_drift_static(self): self.assertTrue(all(s.config().buoy_drift_mode == "static" for s in planning_scenarios()))
    def test_19_primary_channel_static(self): self.assertTrue(all(s.config().channel_variation_mode == "static" for s in planning_scenarios()))
    def test_20_round16_17_dynamics_disabled(self):
        dynamic = config(buoy_drift_mode="constant", buoy_drift_speed_fraction=.1)
        with self.assertRaises(ValueError): MultiUAVMaritimeEnv(dynamic)
    def test_21_candidate_set_frozen(self): self.assertEqual(len(ACTION_CANDIDATE_NAMES), 7)
    def test_22_horizon_set_frozen(self): self.assertEqual(PLANNING_HORIZONS, (1, 5, 10))
    def test_23_independent_has_no_future_trace(self):
        env, _, info = materialize_scenario(planning_scenarios()[0]); self.assertFalse(independent_decision(env, info).future_trace_accessed)
    def test_24_h1_current_joint_state_only(self):
        env, _, info = materialize_scenario(planning_scenarios()[0]); self.assertFalse(joint_plan(env, info, horizon=1).future_trace_accessed)
    def test_25_h5_no_future_trace(self):
        env, _, info = materialize_scenario(planning_scenarios()[0]); self.assertFalse(joint_plan(env, info, horizon=5).future_trace_accessed)
    def test_26_identical_scenario_snapshots(self):
        _, obs_a, info_a = materialize_scenario(planning_scenarios()[2]); _, obs_b, info_b = materialize_scenario(planning_scenarios()[2]); self.assertTrue(np.array_equal(obs_a, obs_b) and np.array_equal(info_a["raw_queue_bits"], info_b["raw_queue_bits"]))
    def test_27_queue_conservation(self):
        env = MultiUAVMaritimeEnv(config()); env.reset(); _, _, _, _, info = env.step(np.zeros((2, 2))); self.assertLess(np.max(np.abs(info["queue_conservation_residual_bits"])), 1e-6)
    def test_28_payload_leq_queue(self):
        env = MultiUAVMaritimeEnv(config()); env.reset(); _, _, _, _, info = env.step(np.zeros((2, 2))); self.assertTrue(np.all(info["transmitted_payload_bits"] <= info["queue_before_bits"] + 1e-6))
    def test_29_payload_leq_capacity(self):
        env = MultiUAVMaritimeEnv(config()); env.reset(); _, _, _, _, info = env.step(np.zeros((2, 2))); self.assertTrue(np.all(info["transmitted_payload_bits"] <= info["link_capacity_bits"] + 1e-6))
    def test_30_no_nan_inf(self):
        env = MultiUAVMaritimeEnv(config()); observation, info = env.reset(); self.assertTrue(np.all(np.isfinite(observation)) and np.all(np.isfinite(info["uav_buoy_rates_bps"])))
    def test_31_diagnostic_seed_isolation(self): self.assertTrue(all(s.seed >= 18000 and SCENARIO_NAMESPACE == "round18_multi_uav_planning_diagnostic" for s in planning_scenarios()))
    def test_32_old_dev_zero(self): self.assertEqual(json.loads((ROOT / "round_18/round_manifest.json").read_text())["old_development_test_access_count"], 0)
    def test_33_fresh_dev_zero(self): self.assertEqual(json.loads((ROOT / "round_18/round_manifest.json").read_text())["fresh_development_access_count"], 0)
    def test_34_holdout_zero(self): self.assertEqual(json.loads((ROOT / "round_18/round_manifest.json").read_text())["untouched_holdout_new_access_count"], 0)
    def test_35_hgnn_false(self): self.assertFalse(json.loads((ROOT / "round_18/round_manifest.json").read_text())["hgnn_training_performed"])
    def test_36_ppo_false(self): self.assertFalse(json.loads((ROOT / "round_18/round_manifest.json").read_text())["ppo_training_performed"])
    def test_37_compileall(self): self.assertTrue(compileall.compile_dir(str(ROOT / "src/wt/round18"), quiet=1))


if __name__ == "__main__": unittest.main()
