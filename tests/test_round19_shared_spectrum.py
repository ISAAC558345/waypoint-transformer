import compileall
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.round18.multi_env import MultiUAVMaritimeEnv, deterministic_matching
from wt.round18.planner import ACTION_CANDIDATE_NAMES as R18_ACTIONS, PLANNING_HORIZONS as R18_HORIZONS
from wt.round19.audit import single_uav_regression
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv, SharedSpectrumOptions, assignment_link_metrics, gain_matrix, joint_aware_matching
from wt.round19.planner import ACTION_CANDIDATE_NAMES, PLANNING_HORIZONS, independent_decision, joint_plan
from wt.round19.scenarios import SCENARIO_NAMESPACE, materialize_scenario, planning_scenarios


ROOT = Path(__file__).resolve().parents[1]
R18_LOCK = "8A9462FC87F1DB546E30877DB8814FE591BF90CCB5DE81B0A6C9C63E96C0DA9B"


def config(k=3, **extra):
    values = dict(num_uavs=2, num_buoys=k, task_semantics_version="v1", buoy_drift_mode="static", channel_variation_mode="static", initial_uav_positions_m=((85., 90.), (115., 90.)), initial_queue_bits_by_buoy=tuple(2.5e6 for _ in range(k)), terminate_on_task_success=False, episode_horizon=8)
    values.update(extra)
    return MaritimeEnvConfig(**values)


class Round19SharedSpectrumTests(unittest.TestCase):
    def test_01_round18_lock_unchanged(self): self.assertEqual(hashlib.sha256((ROOT / "round_18/ROUND_LOCK.json").read_bytes()).hexdigest().upper(), R18_LOCK)
    def test_02_no_optimizer(self): self.assertNotIn("optimizer", (ROOT / "src/wt/round19/planner.py").read_text())
    def test_03_no_backward(self): self.assertNotIn("backward", (ROOT / "src/wt/round19/planner.py").read_text())
    def test_04_no_neural_checkpoint(self): self.assertFalse(list((ROOT / "src/wt/round19").glob("*.pt")))
    def test_05_m1_interference_zero(self): self.assertTrue(all(row["passed"] for row in single_uav_regression()))
    def test_06_eta0_orthogonal_exact_one_step(self):
        c=config(); old=MultiUAVMaritimeEnv(c); new=SharedSpectrumMultiUAVEnv(c, options=SharedSpectrumOptions("orthogonal", 0.0)); old.reset(seed=12); new.reset(seed=12); a=np.array(((.2,-.4),(.7,.1))); x=old.step(a); y=new.step(a); self.assertTrue(np.array_equal(x[0],y[0]) and x[1] == y[1] and np.array_equal(x[4]["raw_queue_bits"],y[4]["raw_queue_bits"]))
    def test_07_desired_signal(self):
        c=config(); gains=np.array(((2.,3.,4.),(5.,6.,7.))); d,_,_,_,_=assignment_link_metrics(np.array((1,2)),gains,c,.5); self.assertTrue(np.allclose(d, [.3,.7]))
    def test_08_cross_interference(self):
        c=config(); gains=np.array(((2.,3.,4.),(5.,6.,7.))); _,i,_,_,_=assignment_link_metrics(np.array((1,2)),gains,c,.5); self.assertTrue(np.allclose(i,[.2,.3]))
    def test_09_sinr_formula(self):
        c=config(); d,i,s,_,_=assignment_link_metrics(np.array((0,1)),np.array(((2.,3.,4.),(5.,6.,7.))),c,.25); self.assertTrue(np.allclose(s,d/(c.noise_power_w+i)))
    def test_10_rate_formula(self):
        c=config(); _,_,s,_,r=assignment_link_metrics(np.array((0,1)),np.array(((2.,3.,4.),(5.,6.,7.))),c,.25); self.assertTrue(np.allclose(r,c.total_bandwidth_hz*np.log2(1+s)))
    def test_11_exclusive_matching(self):
        a,_,_,_=joint_aware_matching(np.ones(3),np.ones((2,3)),config(),.5); self.assertEqual(len(set(a[a>=0])),np.count_nonzero(a>=0))
    def test_12_joint_matching_deterministic(self):
        first=joint_aware_matching(np.ones(3),np.ones((2,3)),config(),.5)[0]; second=joint_aware_matching(np.ones(3),np.ones((2,3)),config(),.5)[0]; self.assertTrue(np.array_equal(first,second))
    def test_13_no_future_in_matching(self): self.assertNotIn("future", (ROOT / "src/wt/round19/multi_env.py").read_text().split("def joint_aware_matching",1)[1].split("def shared_shadow_step",1)[0])
    def test_14_k3_support(self): self.assertEqual(SharedSpectrumMultiUAVEnv(config(3),options=SharedSpectrumOptions("shared",.25)).reset()[1]["raw_queue_bits"].shape,(3,))
    def test_15_k5_support(self): self.assertEqual(SharedSpectrumMultiUAVEnv(config(5),options=SharedSpectrumOptions("shared",.25)).reset()[1]["raw_queue_bits"].shape,(5,))
    def test_16_padding_state_dim(self): self.assertNotEqual(SharedSpectrumMultiUAVEnv(config(3),options=SharedSpectrumOptions("shared",.25)).state_dim,SharedSpectrumMultiUAVEnv(config(5),options=SharedSpectrumOptions("shared",.25)).state_dim)
    def test_17_queue_conservation(self):
        e=SharedSpectrumMultiUAVEnv(config(),options=SharedSpectrumOptions("shared",.5)); e.reset(); self.assertLess(np.max(np.abs(e.step(np.zeros((2,2)))[4]["queue_conservation_residual_bits"])),1e-6)
    def test_18_payload_leq_queue(self):
        e=SharedSpectrumMultiUAVEnv(config(),options=SharedSpectrumOptions("shared",.5)); e.reset(); i=e.step(np.zeros((2,2)))[4]; self.assertTrue(np.all(i["transmitted_payload_bits"] <= i["queue_before_bits"]+1e-6))
    def test_19_payload_leq_capacity(self):
        e=SharedSpectrumMultiUAVEnv(config(),options=SharedSpectrumOptions("shared",.5)); e.reset(); i=e.step(np.zeros((2,2)))[4]; self.assertTrue(np.all(i["transmitted_payload_bits"] <= i["link_capacity_bits"]+1e-6))
    def test_20_fixed_power_unchanged(self): self.assertEqual(config().buoy_tx_power_w,.1)
    def test_21_bandwidth_semantics(self): self.assertEqual(config().total_bandwidth_hz,1.0e6)
    def test_22_primary_drift_static(self): self.assertTrue(all(item.config().buoy_drift_mode == "static" for item in planning_scenarios()))
    def test_23_primary_channel_static(self): self.assertTrue(all(item.config().channel_variation_mode == "static" for item in planning_scenarios()))
    def test_24_safety_unchanged(self): self.assertEqual(config().multi_uav_min_separation_m,15.0)
    def test_25_action_set_unchanged(self): self.assertEqual(ACTION_CANDIDATE_NAMES,R18_ACTIONS)
    def test_26_horizon_set_is_frozen_except_authorized_h20_extension(self):
        amendment = ROOT / "round_32" / "ROUND_LOCK_AMENDMENT.json"
        self.assertEqual(PLANNING_HORIZONS, R18_HORIZONS if not amendment.is_file() else (1, 5, 10, 20))
    def test_27_independent_no_other_intent(self):
        e,_,i=materialize_scenario(next(x for x in planning_scenarios() if x.spectrum_tier=="low")); self.assertFalse(independent_decision(e,i).future_trace_accessed)
    def test_28_h1_current_only(self):
        e,_,i=materialize_scenario(next(x for x in planning_scenarios() if x.spectrum_tier=="low")); self.assertFalse(joint_plan(e,i,horizon=1).future_trace_accessed)
    def test_29_h5_no_future_trace(self):
        e,_,i=materialize_scenario(next(x for x in planning_scenarios() if x.spectrum_tier=="low")); self.assertFalse(joint_plan(e,i,horizon=5).future_trace_accessed)
    def test_30_conflict_classifier_fixed(self): self.assertEqual(SharedSpectrumOptions("shared",.5).conflict_isr_threshold,.25)
    def test_31_diagnostic_seeds_isolated(self): self.assertTrue(all(s.seed>=19000 and SCENARIO_NAMESPACE=="round19_interference_coordination_diagnostic" for s in planning_scenarios()))
    def test_32_old_dev_zero(self): self.assertEqual(json.loads((ROOT / "round_19/round_manifest.json").read_text())["old_development_test_access_count"],0)
    def test_33_fresh_dev_zero(self): self.assertEqual(json.loads((ROOT / "round_19/round_manifest.json").read_text())["fresh_development_access_count"],0)
    def test_34_holdout_zero(self): self.assertEqual(json.loads((ROOT / "round_19/round_manifest.json").read_text())["untouched_holdout_new_access_count"],0)
    def test_35_no_hgnn(self): self.assertFalse(json.loads((ROOT / "round_19/round_manifest.json").read_text())["hgnn_training_performed"])
    def test_36_no_ppo_or_marl(self):
        m=json.loads((ROOT / "round_19/round_manifest.json").read_text()); self.assertFalse(m["ppo_training_performed"] or m["multi_agent_rl_performed"])
    def test_37_compileall(self): self.assertTrue(compileall.compile_dir(str(ROOT / "src/wt/round19"),quiet=1))


if __name__ == "__main__": unittest.main()
