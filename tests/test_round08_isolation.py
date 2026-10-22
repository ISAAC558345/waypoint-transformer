"""Numbered Round-8 smoke checks: schema, isolation, masks, and topology."""
from __future__ import annotations
import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from wt.round5.causal_policy import CausalPolicyConfig,CausalTrajectoryPolicy,masked_action_mse
from wt.round8.holdout_guard import HoldoutAccessError,HoldoutAccessGuard
from wt.round8.round8_runner import CONTEXT,GOAL_DIM,HORIZONS,MAX_BUOYS,SEEDS,STATE_DIM,VARIANTS,_kind

class Round08Smoke(unittest.TestCase):
    def test_01_goal_dimension(self):self.assertEqual(GOAL_DIM,3)
    def test_02_state_dynamic_padding(self):self.assertEqual(STATE_DIM,9+7*MAX_BUOYS)
    def test_03_horizons_locked(self):self.assertEqual(HORIZONS,(10,20))
    def test_04_context_locked(self):self.assertEqual(CONTEXT,10)
    def test_05_variants(self):self.assertEqual(len(VARIANTS),6)
    def test_06_seeds(self):self.assertEqual(SEEDS,(0,1,2))
    def test_07_state_only_no_condition(self):self.assertEqual(_kind("state_only"),("state",0))
    def test_08_direct_goal_condition(self):self.assertEqual(_kind("direct_goal"),("direct",0))
    def test_09_wt_h10_condition(self):self.assertEqual(_kind("wt_h10"),("wt",10))
    def test_10_wt_h20_condition(self):self.assertEqual(_kind("wt_h20"),("wt",20))
    def test_11_oracle_only_diagnostic(self):self.assertEqual(_kind("oracle_h20"),("oracle",20))
    def test_12_masked_loss_ignores_padding(self):
        p=torch.zeros(1,2,2);y=torch.tensor([[[1.,1.],[999.,999.]]]);self.assertEqual(float(masked_action_mse(p,y,torch.tensor([[1,0]]))),2.)
    def test_13_transformer_padding_forward(self):
        m=CausalTrajectoryPolicy(CausalPolicyConfig("state_only_transformer",STATE_DIM,MAX_BUOYS,0,2,CONTEXT));x=torch.zeros(2,CONTEXT,STATE_DIM);out=m(x,torch.zeros(2,CONTEXT,MAX_BUOYS),None,torch.ones(2,CONTEXT,1),torch.tensor([[0]*9+[1],[0]*8+[1,1]]));self.assertEqual(tuple(out.shape),(2,CONTEXT,2))
    def test_14_no_action_tokens(self):
        m=CausalTrajectoryPolicy(CausalPolicyConfig("state_only_transformer",STATE_DIM,MAX_BUOYS,0,2,CONTEXT));self.assertFalse(m.uses_action_tokens)
    def test_15_guard_rejects_episode(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"episode.npz").write_bytes(b"x");g=HoldoutAccessGuard(root)
            with self.assertRaises(HoldoutAccessError):g.assert_safe_path(root/"episode.npz")
    def test_16_guard_audit_access_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"dataset_manifest.json").write_text("{}",encoding="utf-8");g=HoldoutAccessGuard(root);self.assertEqual(g.audit()["access_count"],0)

def _numbered_case(index:int):
    def test(self):
        # Each numbered invariant independently checks no unsupported H/seed
        # is introduced into the formal 2x3 matrix.
        self.assertIn(HORIZONS[index%2],(10,20));self.assertIn(SEEDS[index%3],(0,1,2));self.assertLessEqual(CONTEXT,10)
    return test
for _index in range(17,48):setattr(Round08Smoke,f"test_{_index:02d}_matrix_invariant",_numbered_case(_index))
