"""Phase-10.5 tests. No training."""
from __future__ import annotations

import unittest

import numpy as np

from wt.htw_phase101.protocol import DESIGN_TRAIN_SEEDS, DESIGN_VAL_SEEDS, HIGH_CKPT
from wt.htw_phase103.gate import InRangeZeroMotionGate
from wt.htw_phase104.protocol import SCENARIO_BASE as P104
from wt.htw_phase105.baselines import exclusive_assignment, legal_myopic_assignment
from wt.htw_phase105.protocol import HIGH_SEED, N_SCENARIOS, SCENARIO_BASE, sha256_bytes
from wt.htw_phase105.stats import paired_bootstrap
from wt.l3b.env import L3BEnv, conflict_mask
from wt.l3b.policies import myopic_assignment


class Phase105Tests(unittest.TestCase):
    def test_01_independent_seeds_disjoint(self) -> None:
        from wt.htw_phase105.protocol import all_blocked_scenario_seeds
        from wt.htw_phase105.run_phase105 import select_seeds

        seeds = select_seeds()
        self.assertEqual(len(seeds), N_SCENARIOS)
        blocked = all_blocked_scenario_seeds()
        self.assertFalse(set(seeds) & blocked)
        self.assertFalse(set(seeds) & set(DESIGN_TRAIN_SEEDS))
        self.assertFalse(set(seeds) & set(DESIGN_VAL_SEEDS))
        self.assertFalse(set(seeds) & set(range(P104, P104 + 64)))
        self.assertTrue(all(s >= SCENARIO_BASE for s in seeds))

    def test_02_high_checkpoint_frozen(self) -> None:
        self.assertEqual(HIGH_SEED, 0)
        self.assertTrue(HIGH_CKPT.is_file())
        self.assertEqual(sha256_bytes(HIGH_CKPT.read_bytes()), "dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109")

    def test_03_gate_is_phase103(self) -> None:
        self.assertEqual(InRangeZeroMotionGate.__module__, "wt.htw_phase103.gate")

    def test_04_legal_myopic_has_no_collision(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        raw = myopic_assignment(env)
        legal = legal_myopic_assignment(env)
        self.assertFalse(bool(np.any(conflict_mask(legal, env.config.num_buoys))))
        excl = exclusive_assignment(env)
        self.assertFalse(bool(np.any(conflict_mask(excl, env.config.num_buoys))))
        self.assertEqual(raw.shape, legal.shape)

    def test_05_bootstrap_single_seed_pairing(self) -> None:
        delta = np.array([1.0, 2.0, 3.0, 4.0])
        out = paired_bootstrap(delta, n_boot=200, seed=1)
        self.assertFalse(out["model_seed_ci_invented"])
        self.assertAlmostEqual(out["mean_delta"], 2.5)
        self.assertEqual(out["n"], 4)

    def test_06_no_training_flag_in_protocol_constants(self) -> None:
        from wt.htw_phase105.protocol import HOLD

        self.assertEqual(HOLD, 15)


if __name__ == "__main__":
    unittest.main()
