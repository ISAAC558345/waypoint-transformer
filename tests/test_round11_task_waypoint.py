"""Round-11 smoke tests for task semantics, dynamic widths, and masking."""
import json, unittest
from pathlib import Path
import numpy as np
import torch
from wt.round11.task_waypoint import TaskWaypointConfig, TaskWaypointMLP, masked_softmax, task_waypoint_losses

ROOT = Path(__file__).resolve().parents[1]

class Round11(unittest.TestCase):
    def setUp(self):
        self.cfg = TaskWaypointConfig(state_dim=7, max_buoys=3, progress_scale=5.0, return_std=2.0)
        self.model = TaskWaypointMLP(self.cfg)
        self.state = torch.zeros(2, 7); self.mask = torch.tensor([[1,1,0],[1,0,0]], dtype=torch.bool)
    def test_01_dynamic_input_dimension(self): self.assertEqual(self.cfg.input_dim, 10)
    def test_02_no_goal_dimension(self): self.assertEqual(self.cfg.metadata()["goal_dim"], 0)
    def test_03_no_spatial_head(self): self.assertFalse(self.cfg.metadata()["spatial_head_present"])
    def test_04_forward_shapes(self):
        p=self.model(self.state,self.mask); self.assertEqual(tuple(p["service_distribution"].shape),(2,3)); self.assertEqual(tuple(p["gross_service_progress"].shape),(2,3)); self.assertEqual(tuple(p["H_step_return"].shape),(2,))
    def test_05_service_mask_zero(self): self.assertEqual(float(self.model(self.state,self.mask)["service_distribution"][0,2]),0.0)
    def test_06_service_sums_one(self): self.assertTrue(torch.allclose(self.model(self.state,self.mask)["service_distribution"].sum(1),torch.ones(2)))
    def test_07_progress_mask_zero(self): self.assertEqual(float(self.model(self.state,self.mask)["gross_service_progress"][1,1]),0.0)
    def test_08_progress_nonnegative(self): self.assertGreaterEqual(float(self.model(self.state,self.mask)["gross_service_progress"].min()),0.0)
    def test_09_masked_softmax_rejects_empty(self):
        with self.assertRaises(ValueError): masked_softmax(torch.zeros(1,3),torch.zeros(1,3,dtype=torch.bool))
    def test_10_state_shape_checked(self):
        with self.assertRaises(ValueError): self.model(torch.zeros(2,8),self.mask)
    def test_11_mask_shape_checked(self):
        with self.assertRaises(ValueError): self.model(self.state,torch.ones(2,2,dtype=torch.bool))
    def test_12_padding_does_not_change_progress_loss(self):
        p={"service_distribution":torch.tensor([[1.,0.,0.]]),"gross_service_progress":torch.tensor([[.2,0.,0.]]),"H_step_return":torch.tensor([0.])}; t={"service_distribution":torch.tensor([[1.,0.,0.]]),"service_valid":torch.tensor([True]),"gross_service_progress":torch.tensor([[.2,99.,99.]]),"H_step_return":torch.tensor([0.])}; self.assertEqual(float(task_waypoint_losses(p,t,torch.tensor([[1,0,0]]))["progress"]),0.0)
    def test_13_service_invalid_zero_loss(self):
        p={"service_distribution":torch.tensor([[1.,0.,0.]]),"gross_service_progress":torch.zeros(1,3),"H_step_return":torch.zeros(1)}; t={"service_distribution":torch.zeros(1,3),"service_valid":torch.tensor([False]),"gross_service_progress":torch.zeros(1,3),"H_step_return":torch.zeros(1)}; self.assertEqual(float(task_waypoint_losses(p,t,torch.tensor([[1,0,0]]))["service"]),0.0)
    def test_14_manifest_has_six_runs_configured(self):
        cfg=json.loads((ROOT/'round_11/configs/task_waypoint_training_v1.json').read_text()); self.assertEqual(len(cfg['horizons'])*len(cfg['seeds']),6)
    def test_15_round_rules_forbid_spatial(self): self.assertIn('spatial head', (ROOT/'round_11/AGENTS.md').read_text())

def _case(index):
    def test(self):
        cfg=TaskWaypointConfig(state_dim=index%11+1,max_buoys=index%5+1,progress_scale=1.,return_std=1.); model=TaskWaypointMLP(cfg); state=torch.zeros(1,cfg.state_dim); mask=torch.ones(1,cfg.max_buoys,dtype=torch.bool); out=model(state,mask); self.assertEqual(out['service_distribution'].shape[-1],cfg.max_buoys)
    return test
for _i in range(16,47): setattr(Round11,f'test_{_i:02d}_dynamic_width',_case(_i))
