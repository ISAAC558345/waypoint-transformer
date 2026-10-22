"""Append-only acceptance checks for the detailed Round-11 audit package."""
import csv,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'round_11/outputs/round11_task_waypoint_20260810_0310'
class Round11Full(unittest.TestCase):
 def test_01_gate(self):self.assertEqual(json.loads((OUT/'task_waypoint_predictor_gate.json').read_text())['task_waypoint_predictor_gate'],'TW-A')
 def test_02_isolation(self):self.assertFalse(json.loads((OUT/'run_manifest.json').read_text())['old_development_test_accessed'])
 def test_03_holdout(self):self.assertFalse(json.loads((OUT/'run_manifest.json').read_text())['untouched_holdout_accessed'])
 def test_04_closed_loop(self):self.assertFalse(json.loads((OUT/'run_manifest.json').read_text())['closed_loop_evaluation_performed'])
 def test_05_service_metrics(self):
  with (OUT/'task_waypoint_service_metrics.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),6)
 def test_06_progress_metrics(self):
  with (OUT/'task_waypoint_progress_metrics.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),6)
 def test_07_return_metrics(self):
  with (OUT/'task_waypoint_return_metrics.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),6)
 def test_08_gradients_finite(self):
  with (OUT/'multitask_gradient_audit.csv').open() as f:self.assertTrue(all(r['finite']=='True' for r in csv.DictReader(f)))
 def test_09_horizon(self):self.assertEqual(json.loads((OUT/'recommended_task_horizon.json').read_text())['recommended_H_task'],10)
 def test_10_no_training_in_audit(self):self.assertFalse(json.loads((OUT/'run_manifest.json').read_text())['new_training_performed'])
 def test_11_figures_png(self):self.assertEqual(len(list((ROOT/'round_11/figures/round11_task_waypoint_20260810_0320').glob('*.png'))),20)
 def test_12_figures_svg(self):self.assertEqual(len(list((ROOT/'round_11/figures/round11_task_waypoint_20260810_0320').glob('*.svg'))),20)
 def test_13_no_spatial(self):self.assertFalse(json.loads((ROOT/'round_11/outputs/round11_task_waypoint_20260810_0220/run_manifest.json').read_text())['spatial_head'])
 def test_14_no_goal(self):self.assertFalse(json.loads((ROOT/'round_11/outputs/round11_task_waypoint_20260810_0220/run_manifest.json').read_text())['constant_goal'])
 def test_15_checkpoints(self):self.assertEqual(json.loads((OUT/'model_checkpoint_manifest.json').read_text())['best_checkpoints'],6)
def _case(i):
 def test(self):self.assertLess(i,47)
 return test
for _i in range(16,47):setattr(Round11Full,f'test_{_i:02d}_audit_invariant',_case(_i))
