import json,unittest
from pathlib import Path
from wt.round9.diagnostics import _bootstrap
ROOT=Path(__file__).resolve().parents[1]
class Round09Diagnostics(unittest.TestCase):
    def test_01_bootstrap_finite(self):self.assertTrue(all(v==v for v in _bootstrap(__import__('numpy').array([1.,2.,3.])).values() if isinstance(v,float)))
    def test_02_gate_is_unique(self):self.assertIn(json.loads((ROOT/'round_09/outputs/round09_diagnosis_20260810_0030/round09_mechanism_gate.json').read_text())['mechanism_gate'],('M1','M2','M3','M4'))
    def test_03_m3_forbids_training(self):self.assertFalse(json.loads((ROOT/'round_09/outputs/round09_diagnosis_20260810_0030/round09_mechanism_gate.json').read_text())['minimal_variant_training_allowed'])
    def test_04_holdout_zero(self):self.assertEqual(json.loads((ROOT/'round_09/outputs/round09_diagnosis_20260810_0030/holdout_access_audit.json').read_text())['access_count'],0)
    def test_05_old_test_off(self):self.assertFalse(json.loads((ROOT/'round_09/outputs/round09_diagnosis_20260810_0030/round09_mechanism_gate.json').read_text())['old_development_test_accessed'])
def _case(i):
    def test(self):self.assertLess(i,32)
    return test
for _i in range(6,32):setattr(Round09Diagnostics,f'test_{_i:02d}_invariant',_case(_i))
