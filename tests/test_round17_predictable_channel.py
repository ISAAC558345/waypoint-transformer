"""Round-17 contracts: predictable channels only, no protected data or training."""
from __future__ import annotations

import compileall
import inspect
import json
import math
from pathlib import Path
import unittest

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.env import MaritimeEnv
from wt.round15.planner import ACTION_CANDIDATE_NAMES as R15_ACTIONS
from wt.round17.audit import _static_regression
from wt.round17.planner import ACTION_CANDIDATE_NAMES, _causal_score, advance_channel_phase, finite_lattice_plan, predictable_channel_multiplier
from wt.round17.scenarios import materialize_scenario, planning_scenarios


ROOT=Path(__file__).resolve().parents[1]


class Round17PredictableChannelTests(unittest.TestCase):
    def setUp(self):
        self.static=MaritimeEnv(MaritimeEnvConfig(num_buoys=3,task_semantics_version="v1",randomize_buoys=True,seed=17000))
        self.dynamic=MaritimeEnv(MaritimeEnvConfig(num_buoys=3,task_semantics_version="v1",randomize_buoys=True,seed=17000,channel_variation_mode="slow_oscillation",channel_log_amplitude=.15,channel_initial_phase_by_buoy=(math.pi/2,-math.pi/2,.2)))
    def test_001_round16_lock_is_locked(self): self.assertEqual(json.loads((ROOT/"round_16/ROUND_LOCK.json").read_text())["status"],"locked")
    def test_002_round18_lock_chain_is_present(self): self.assertEqual(json.loads((ROOT/"round_18/ROUND_LOCK.json").read_text())["status"],"locked")
    def test_003_static_channel_schema_unchanged(self): self.assertEqual(self.static.observation_schema["schema_version"],"v1")
    def test_004_static_channel_state_dim_unchanged(self): self.assertEqual(self.static.state_dim,9+7*3)
    def test_005_dynamic_channel_state_dim(self): self.assertEqual(self.dynamic.state_dim,9+10*3)
    def test_006_static_drift_required_for_primary(self): self.assertEqual(next(s for s in planning_scenarios() if s.channel_tier=="medium").config().buoy_drift_mode,"static")
    def test_007_static_exact_regression(self): self.assertTrue(all(row["passed"] for row in _static_regression()))
    def test_008_multiplier_positive(self): _,info=self.dynamic.reset(seed=17000);self.assertTrue(np.all(info["channel_multipliers"]>0))
    def test_009_multiplier_matches_phase(self): _,info=self.dynamic.reset(seed=17000);np.testing.assert_allclose(info["channel_multipliers"],np.exp(.15*np.sin(info["channel_phase_rad"])))
    def test_010_phase_transition_correct(self): _,info=self.dynamic.reset(seed=17000);phase=info["channel_phase_rad"];_,_,_,_,after=self.dynamic.step(np.zeros(2));np.testing.assert_allclose(after["channel_phase_rad"],advance_channel_phase(phase,self.dynamic.config))
    def test_011_phase_features_observed(self): obs,info=self.dynamic.reset(seed=17000);nodes=obs[9:].reshape(3,10);np.testing.assert_allclose(nodes[:,2],np.sin(info["channel_phase_rad"]),atol=1e-6);np.testing.assert_allclose(nodes[:,3],np.cos(info["channel_phase_rad"]),atol=1e-6)
    def test_012_frequency_normalized(self): obs,_=self.dynamic.reset(seed=17000);nodes=obs[9:].reshape(3,10);np.testing.assert_allclose(nodes[:,4],20/24,atol=1e-6)
    def test_013_phase_state_is_bounded(self): obs,_=self.dynamic.reset(seed=17000);self.assertTrue(self.dynamic.observation_space.contains(obs))
    def test_014_padding_contract_is_zero(self): self.assertIn("zero",json.loads((ROOT/"round_17/dynamic_channel_schema.json").read_text())["padding_rule"])
    def test_015_asynchronous_seeded_phases_are_deterministic(self):
        s=next(x for x in planning_scenarios() if x.phase_relationship=="seeded_async" and x.channel_tier=="medium");_,a=materialize_scenario(s)[1:];_,b=materialize_scenario(s)[1:];np.testing.assert_allclose(a["channel_phase_rad"],b["channel_phase_rad"])
    def test_016_phase_conflict_is_registered(self): self.assertIn("channel_phase_conflict",{s.phase_relationship for s in planning_scenarios()})
    def test_017_phase_conflict_reverses_h10_multiplier_ranking(self):
        s=next(x for x in planning_scenarios() if x.phase_relationship=="channel_phase_conflict" and x.channel_tier=="medium");p=np.asarray(s.phases());future=p.copy()
        for _ in range(10): future=advance_channel_phase(future,s.config())
        self.assertNotEqual(int(np.argmax(predictable_channel_multiplier(p,s.config()))),int(np.argmax(predictable_channel_multiplier(future,s.config()))))
    def test_018_causal_uses_no_deepcopy(self): self.assertNotIn("deepcopy",inspect.getsource(_causal_score))
    def test_019_causal_prediction_is_finite(self):
        o,i=self.dynamic.reset(seed=17000);score,nodes=_causal_score(self.dynamic,i,np.zeros(2),10);self.assertTrue(np.isfinite(score));self.assertEqual(nodes,10)
    def test_020_oracle_non_deployable(self):
        o,i=self.dynamic.reset(seed=17000);d=finite_lattice_plan(self.dynamic,o,i,horizon=1,mode="oracle");self.assertTrue(d.oracle_future_channel);self.assertFalse(d.deployable_diagnostic)
    def test_021_causal_deployable(self):
        o,i=self.dynamic.reset(seed=17000);d=finite_lattice_plan(self.dynamic,o,i,horizon=1,mode="causal");self.assertFalse(d.oracle_future_channel);self.assertTrue(d.deployable_diagnostic)
    def test_022_action_lattice_unchanged(self): self.assertEqual(ACTION_CANDIDATE_NAMES,R15_ACTIONS)
    def test_023_frozen_lattice_expansion(self):
        o,i=self.dynamic.reset(seed=17000);d=finite_lattice_plan(self.dynamic,o,i,horizon=5,mode="causal");self.assertEqual(d.nodes_expanded,len(R15_ACTIONS)*5)
    def test_024_safe_action_is_legal(self):
        o,i=self.dynamic.reset(seed=17000);self.assertTrue(self.dynamic.action_space.contains(finite_lattice_plan(self.dynamic,o,i,horizon=20,mode="causal").action))
    def test_025_reward_and_resource_baseline_static(self):
        a=MaritimeEnv(MaritimeEnvConfig(num_buoys=3,task_semantics_version="v1",randomize_buoys=True,seed=7));b=MaritimeEnv(MaritimeEnvConfig(num_buoys=3,task_semantics_version="v1",randomize_buoys=True,seed=7,channel_variation_mode="static"));a.reset(seed=7);b.reset(seed=7);_,ra,_,_,ia=a.step(np.zeros(2));_,rb,_,_,ib=b.step(np.zeros(2));self.assertEqual(ra,rb);np.testing.assert_allclose(ia["bandwidth_allocation_fraction"],ib["bandwidth_allocation_fraction"])
    def test_026_queue_conservation(self): self.dynamic.reset(seed=17000);_,_,_,_,i=self.dynamic.step(np.zeros(2));self.assertLessEqual(float(np.max(np.abs(i["queue_conservation_residual_bits"]))),1e-6)
    def test_027_payload_accounting(self): self.dynamic.reset(seed=17000);_,_,_,_,i=self.dynamic.step(np.zeros(2));self.assertTrue(np.all(i["transmitted_payload_bits"]<=i["queue_before_bits"]+1e-6))
    def test_028_no_nan_or_inf(self): o,i=self.dynamic.reset(seed=17000);self.assertTrue(np.all(np.isfinite(o)));self.assertTrue(np.all(np.isfinite(i["rates_bps"])))
    def test_029_scenario_namespace_is_isolated(self): self.assertTrue(all(s.scenario_id.startswith("round17_channel_planning_diagnostic") for s in planning_scenarios()))
    def test_030_scenarios_cover_primary_tiers(self): self.assertEqual({s.channel_tier for s in planning_scenarios()},{"static","low","medium","high"})
    def test_031_scenarios_cover_three_and_five(self): self.assertEqual({s.num_buoys for s in planning_scenarios()},{3,5})
    def test_032_no_optimizer_backward_or_torch(self):
        source="\n".join(p.read_text(encoding="utf-8") for p in (ROOT/"src/wt/round17").glob("*.py"));self.assertNotIn("torch",source);self.assertNotIn(".backward(",source);self.assertNotIn("optimizer",source)
    def test_033_controls_forbid_protected_access(self):
        m=json.loads((ROOT/"round_17/round_manifest.json").read_text());self.assertEqual(m["old_development_test_access_count"],0);self.assertEqual(m["fresh_development_access_count"],0);self.assertEqual(m["untouched_holdout_new_access_count"],0)
    def test_034_controls_forbid_hgnn_ppo_multi_uav(self):
        m=json.loads((ROOT/"round_17/round_manifest.json").read_text());self.assertFalse(m["hgnn_training_performed"]);self.assertFalse(m["ppo_training_performed"]);self.assertFalse(m["multi_uav_enabled"])
    def test_035_dynamic_channel_requires_v1(self):
        with self.assertRaises(ValueError): MaritimeEnvConfig(channel_variation_mode="slow_oscillation",channel_log_amplitude=.15)
    def test_036_static_mode_requires_zero_amplitude(self):
        with self.assertRaises(ValueError): MaritimeEnvConfig(channel_log_amplitude=.15)
    def test_037_dynamic_channel_schema_forbids_future_fields(self):
        d=json.loads((ROOT/"round_17/dynamic_channel_schema.json").read_text());self.assertIn("future_rate",d["future_channel_fields_forbidden"])
    def test_038_compileall_round17(self): self.assertTrue(compileall.compile_dir(str(ROOT/"src/wt/round17"),quiet=1))


if __name__=="__main__": unittest.main()
