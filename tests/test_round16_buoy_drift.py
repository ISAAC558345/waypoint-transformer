"""Round-16 observable-drift contracts; no historical dataset is opened."""
from __future__ import annotations
import inspect,json
from pathlib import Path
import unittest
import numpy as np
from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.env import MaritimeEnv, reflect_buoy_motion
from wt.round15.planner import ACTION_CANDIDATE_NAMES as R15_ACTIONS
from wt.round16.audit import _static_regression
from wt.round16.planner import ACTION_CANDIDATE_NAMES, _causal_score, finite_lattice_plan
from wt.round16.scenarios import materialize_scenario, planning_scenarios

ROOT=Path(__file__).resolve().parents[1]

class Round16DriftTests(unittest.TestCase):
 def setUp(self):
  self.static=MaritimeEnv(MaritimeEnvConfig(num_buoys=3,task_semantics_version="v1",randomize_buoys=True,seed=16000))
  self.dynamic=MaritimeEnv(MaritimeEnvConfig(num_buoys=3,task_semantics_version="v1",randomize_buoys=True,seed=16000,buoy_drift_mode="constant",buoy_drift_speed_fraction=.1))
 def test_001_round15_lock_is_locked(self): self.assertEqual(json.loads((ROOT/"round_15/ROUND_LOCK.json").read_text())["status"],"locked")
 def test_002_round14_lock_is_locked(self): self.assertEqual(json.loads((ROOT/"round_14/ROUND_LOCK.json").read_text())["status"],"locked")
 def test_003_later_round_preserves_round16_lock_chain(self):
  self.assertTrue((ROOT/"round_17").is_dir());self.assertEqual(json.loads((ROOT/"round_16/ROUND_LOCK.json").read_text())["status"],"locked");self.assertEqual(json.loads((ROOT/"round_18/ROUND_LOCK.json").read_text())["status"],"locked")
 def test_004_static_velocity_zero(self): _,i=self.static.reset(seed=16000);np.testing.assert_array_equal(i["buoy_drift_velocity_mps"],np.zeros((3,2)))
 def test_005_static_state_dim_unchanged(self): self.assertEqual(self.static.state_dim,9+7*3)
 def test_006_dynamic_state_dim_has_velocity(self): self.assertEqual(self.dynamic.state_dim,9+9*3)
 def test_007_velocity_is_observed(self): _,i=self.dynamic.reset(seed=16000);self.assertIn("buoy_drift_velocity_mps",i)
 def test_008_velocity_normalization(self): o,i=self.dynamic.reset(seed=16000);nodes=o[9:].reshape(3,9);np.testing.assert_allclose(nodes[:,2:4],i["buoy_drift_velocity_mps"]/i["buoy_drift_speed_max_mps"],atol=1e-6)
 def test_009_constant_position_update(self): _,i=self.dynamic.reset(seed=16000);p=i["buoy_positions_m"].copy();v=i["buoy_drift_velocity_mps"].copy();_,_,_,_,after=self.dynamic.step(np.zeros(2));expected,_=reflect_buoy_motion(p,v,1.,self.dynamic.config);np.testing.assert_allclose(after["buoy_positions_m"],expected)
 def test_010_reflection_reverses_velocity(self):
  c=MaritimeEnvConfig(num_buoys=1,task_semantics_version="v1",buoy_drift_mode="constant",buoy_drift_speed_fraction=.1,area_width=100,area_height=100,buoy_margin=10)
  p,v=reflect_buoy_motion(np.array([[89.,50.]]),np.array([[2.,0.]]),1.,c);np.testing.assert_allclose(p,[[89.,50.]]);np.testing.assert_allclose(v,[[-2.,0.]])
 def test_011_reflection_is_legal_after_many_steps(self):
  _,i=self.dynamic.reset(seed=16000)
  for _ in range(128):
   _,_,t,x,i=self.dynamic.step(np.zeros(2));self.assertTrue(np.all(i["buoy_positions_m"]>=self.dynamic.config.buoy_margin));self.assertTrue(np.all(i["buoy_positions_m"]<=i["area_bounds_m"]-self.dynamic.config.buoy_margin));
   if t or x:break
 def test_012_static_schema_is_v1(self): self.assertEqual(self.static.observation_schema["schema_version"],"v1")
 def test_013_dynamic_schema_marks_drift(self): self.assertEqual(self.dynamic.observation_schema["schema_version"],"v1_drift")
 def test_014_action_lattice_unchanged(self): self.assertEqual(ACTION_CANDIDATE_NAMES,R15_ACTIONS)
 def test_015_horizon_set_rejects_two(self):
  o,i=self.dynamic.reset(seed=16000)
  with self.assertRaises(ValueError):finite_lattice_plan(self.dynamic,o,i,horizon=2,mode="causal")
 def test_016_causal_uses_no_deepcopy(self): self.assertNotIn("deepcopy",inspect.getsource(_causal_score))
 def test_017_oracle_is_non_deployable(self):
  o,i=self.dynamic.reset(seed=16000);d=finite_lattice_plan(self.dynamic,o,i,horizon=1,mode="oracle");self.assertTrue(d.oracle_future_drift);self.assertFalse(d.deployable_diagnostic)
 def test_018_causal_is_deployable_diagnostic(self):
  o,i=self.dynamic.reset(seed=16000);d=finite_lattice_plan(self.dynamic,o,i,horizon=1,mode="causal");self.assertTrue(d.deployable_diagnostic);self.assertFalse(d.oracle_future_drift)
 def test_019_safe_actions_are_legal(self):
  o,i=self.dynamic.reset(seed=16000);d=finite_lattice_plan(self.dynamic,o,i,horizon=5,mode="causal");self.assertTrue(self.dynamic.action_space.contains(d.action))
 def test_020_queue_conservation(self):
  self.dynamic.reset(seed=16000);_,_,_,_,i=self.dynamic.step(np.zeros(2));self.assertLessEqual(float(np.max(np.abs(i["queue_conservation_residual_bits"]))),1e-6)
 def test_021_payload_accounting(self):
  self.dynamic.reset(seed=16000);_,_,_,_,i=self.dynamic.step(np.zeros(2));self.assertTrue(np.all(i["transmitted_payload_bits"]<=i["queue_before_bits"]+1e-6))
 def test_022_dynamic_values_are_finite(self):
  o,i=self.dynamic.reset(seed=16000);self.assertTrue(np.all(np.isfinite(o)));self.assertTrue(np.all(np.isfinite(i["buoy_drift_velocity_mps"])))
 def test_023_isolated_scenario_seeds(self): self.assertTrue(all("round16_dynamic_planning_diagnostic" in s.scenario_id for s in planning_scenarios()))
 def test_024_scenarios_cover_primary_tiers(self): self.assertTrue({"static","low","medium","high"}.issubset({s.drift_tier for s in planning_scenarios()}))
 def test_025_scenarios_cover_slow_diagnostic(self): self.assertIn("slowly_varying_secondary",{s.drift_tier for s in planning_scenarios()})
 def test_026_scenarios_cover_buoy_counts(self): self.assertEqual({s.num_buoys for s in planning_scenarios()},{3,5})
 def test_027_no_optimizer_or_backward(self):
  source="\n".join(p.read_text(encoding="utf-8") for p in (ROOT/"src/wt/round16").glob("*.py"));self.assertNotIn("torch",source);self.assertNotIn(".backward(",source)
 def test_028_controls_forbid_holdout(self):
  m=json.loads((ROOT/"round_16/round_manifest.json").read_text());self.assertEqual(m["untouched_holdout_new_access_count"],0);self.assertFalse(m["untouched_holdout_new_accessed"])
 def test_029_controls_forbid_hgnn_ppo_multi_uav(self):
  m=json.loads((ROOT/"round_16/round_manifest.json").read_text());self.assertFalse(m["hgnn_training_performed"]);self.assertFalse(m["ppo_training_performed"]);self.assertFalse(m["multi_uav_enabled"])
 def test_030_dynamic_requires_v1_visibility(self):
  with self.assertRaises(ValueError):MaritimeEnvConfig(buoy_drift_mode="constant",buoy_drift_speed_fraction=.1)
 def test_031_static_regression_control(self): self.assertTrue(all(r["passed"] for r in _static_regression()))

if __name__=="__main__":unittest.main()
