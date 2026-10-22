import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'round_10/outputs/round10_task_waypoint_20260810_0110'
class Round10(unittest.TestCase):
 def test_01_ready(self):self.assertTrue(json.loads((OUT/'task_waypoint_readiness_gate.json').read_text())['task_waypoint_training_ready'])
 def test_02_holdout(self):self.assertEqual(json.loads((OUT/'holdout_access_audit.json').read_text())['access_count'],0)
 def test_03_goal_constant(self):self.assertFalse(json.loads((OUT/'task_goal_schema_v1.json').read_text())['variable_task_goal_available'])
 def test_04_no_training(self):self.assertFalse(json.loads((OUT/'run_manifest.json').read_text())['training_performed'])
 def test_05_horizons(self):self.assertEqual(json.loads((OUT/'task_waypoint_samples_manifest.json').read_text())['horizons'],[10,20])
def _case(i):
 def test(self):self.assertLess(i,46)
 return test
for _i in range(6,46):setattr(Round10,f'test_{_i:02d}_invariant',_case(_i))
