"""Phase-10.4 tests. Do not train extra seeds here."""
from __future__ import annotations

import unittest

import numpy as np

from wt.htw_phase101.protocol import DESIGN_TRAIN_SEEDS, DESIGN_VAL_SEEDS, PPO
from wt.htw_phase103.gate import InRangeZeroMotionGate
from wt.htw_phase104.protocol import HIGH_CKPT, LOW_SEEDS, N_SCENARIOS, SEED0_FINAL, TRANSITIONS, UPDATES, sha256_bytes
from wt.htw_phase104.scenarios import select_independent_seeds
from wt.htw_phase104.stats import bootstrap_deltas, classify_llexec


class Phase104Tests(unittest.TestCase):
    def test_01_independent_seeds_disjoint_and_count(self) -> None:
        from wt.htw_phase104.protocol import all_blocked_scenario_seeds

        seeds = select_independent_seeds()
        self.assertEqual(len(seeds), N_SCENARIOS)
        self.assertEqual(len(set(seeds)), N_SCENARIOS)
        blocked = all_blocked_scenario_seeds()
        self.assertFalse(set(seeds) & blocked)
        self.assertFalse(set(seeds) & set(DESIGN_TRAIN_SEEDS))
        self.assertFalse(set(seeds) & set(DESIGN_VAL_SEEDS))
        self.assertTrue(all(s >= 104000 for s in seeds))

    def test_02_seed0_checkpoint_sha_matches_phase101(self) -> None:
        self.assertTrue(SEED0_FINAL.is_file())
        self.assertEqual(sha256_bytes(SEED0_FINAL.read_bytes()), "8cb0529e7f9c2851e0a102da5793b29d317e9c48c900e10ac976791b678aebb7")
        self.assertTrue(HIGH_CKPT.is_file())
        self.assertEqual(sha256_bytes(HIGH_CKPT.read_bytes()), "dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109")

    def test_03_budget_and_ppo_frozen(self) -> None:
        self.assertEqual(UPDATES, 100)
        self.assertEqual(int(PPO.rollout_length), 48)
        self.assertEqual(int(PPO.num_envs), 4)
        self.assertEqual(TRANSITIONS, 19200)
        self.assertEqual(tuple(LOW_SEEDS), (0, 1, 2, 3, 4))

    def test_04_gate_is_phase103_class(self) -> None:
        self.assertEqual(InRangeZeroMotionGate.__module__, "wt.htw_phase103.gate")

    def test_05_bootstrap_respects_shared_b3(self) -> None:
        b3 = np.array([1.0, 2.0, 3.0, 4.0])
        b2 = np.vstack([b3 + 1.0, b3 + 2.0])
        delta = b2 - b3[None, :]
        out = bootstrap_deltas(delta, n_boot=200, seed=1)
        self.assertEqual(out["n_seeds"], 2)
        self.assertEqual(out["n_scenarios"], 4)
        self.assertAlmostEqual(out["overall_mean_delta"], 1.5)
        self.assertGreater(out["hierarchical_ci95"]["low"], 0.0)

    def test_06_llexec_thresholds(self) -> None:
        a = classify_llexec(mean_delta=1.0, hier_lo=0.1, seed_lo=0.1, n_positive_seeds=4, n_seeds=5, median_delta=0.5, mean_blocks_b2=10, mean_blocks_b3=9, exclusive_ok_b2=1, exclusive_ok_b3=1)
        self.assertEqual(a["LLEXEC"], "A")
        self.assertTrue(a["stable_low_level_advantage"])
        b = classify_llexec(mean_delta=1.0, hier_lo=-0.1, seed_lo=-0.2, n_positive_seeds=3, n_seeds=5, median_delta=0.1, mean_blocks_b2=10, mean_blocks_b3=9, exclusive_ok_b2=1, exclusive_ok_b3=1)
        self.assertEqual(b["LLEXEC"], "B")
        c = classify_llexec(mean_delta=-1.0, hier_lo=-2.0, seed_lo=-2.0, n_positive_seeds=1, n_seeds=5, median_delta=-1.0, mean_blocks_b2=8, mean_blocks_b3=9, exclusive_ok_b2=1, exclusive_ok_b3=1)
        self.assertEqual(c["LLEXEC"], "C")
        self.assertFalse(c["stable_low_level_advantage"])

    def test_07_scenario_freeze_does_not_need_models(self) -> None:
        from wt.htw_phase104.scenarios import build_scenario_manifest

        payload = build_scenario_manifest(consumed=False)
        self.assertEqual(payload["n_scenarios"], 64)
        self.assertTrue(payload["not_confirmation"])
        self.assertFalse(payload["consumed"])


if __name__ == "__main__":
    unittest.main()
