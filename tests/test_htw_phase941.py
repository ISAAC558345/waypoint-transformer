"""Phase-9.4.1 confirmation protocol and statistics checks. No architecture changes."""
from __future__ import annotations

import unittest

import numpy as np

from wt.htw_phase941.bootstrap import hierarchical_bootstrap, seed_cluster_bootstrap
from wt.htw_phase941.protocol import CONFIRMATION_SEEDS, USED_L3B_SCENARIO_SEEDS, confirmation_ok, dump_frozen, protocol_payload
from wt.htw_phase941.strata import label_competition, label_interference


class Phase941Tests(unittest.TestCase):
    def test_01_confirmation_seeds_unused(self) -> None:
        confirmation_ok()
        self.assertEqual(CONFIRMATION_SEEDS, tuple(range(24, 40)))
        self.assertEqual(len(set(CONFIRMATION_SEEDS) & set(USED_L3B_SCENARIO_SEEDS)), 0)
        self.assertTrue(set(range(24)) <= set(USED_L3B_SCENARIO_SEEDS))

    def test_02_protocol_sha_stable(self) -> None:
        import tempfile
        from pathlib import Path

        payload = protocol_payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            first = dump_frozen(path, payload)
            second = dump_frozen(path, payload)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_03_hierarchical_bootstrap_shape(self) -> None:
        delta = np.arange(15, dtype=np.float64).reshape(5, 3)
        hier = hierarchical_bootstrap(delta, n_boot=200, seed=1)
        cluster = seed_cluster_bootstrap(delta, n_boot=200, seed=1)
        self.assertIn("ci_lo", hier)
        self.assertLessEqual(hier["ci_lo"], hier["ci_hi"])
        self.assertLessEqual(cluster["ci_lo"], cluster["ci_hi"])

    def test_04_strata_labels_frozen_thresholds(self) -> None:
        self.assertEqual(label_interference(0.0, 1.0, 2.0), "low")
        self.assertEqual(label_interference(1.5, 1.0, 2.0), "medium")
        self.assertEqual(label_interference(3.0, 1.0, 2.0), "high")
        self.assertEqual(label_competition(0.1, 0.2), "low")
        self.assertEqual(label_competition(0.3, 0.2), "high")

    def test_05_no_outcome_seed_swap_in_protocol(self) -> None:
        proto = protocol_payload()
        self.assertEqual(proto["confirmation_seeds"], list(range(24, 40)))
        self.assertTrue(proto["checkpoint_rule_frozen_before_training"])
        self.assertFalse(proto["ppo"])
        self.assertFalse(proto["modified_decoder"])


if __name__ == "__main__":
    unittest.main()
