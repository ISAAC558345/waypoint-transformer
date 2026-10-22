"""Phase-6I-C dataset contract. The smoke step is not run here."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import load_decision
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import GRAPH_KEYS, assert_open


class Phase6ICDatasetTests(unittest.TestCase):
    def test_reserved_scenarios_stay_closed(self) -> None:
        for scenario in (322064, 322079):
            with self.assertRaises(RuntimeError):
                assert_open(scenario, "train")
            with self.assertRaises(RuntimeError):
                load_decision(scenario, 0)

    def test_graph_batch_excludes_labels_and_keeps_checkpoint_widths(self) -> None:
        sample = load_decision(322000, 0)
        self.assertEqual(tuple(sample["graph"]), GRAPH_KEYS)
        self.assertNotIn("j15_bits", sample["graph"])
        self.assertNotIn("utility_rank", sample["graph"])
        self.assertNotIn("assignment", sample["graph"])
        self.assertEqual(sample["graph"]["uav"].shape, (1, 6, 8))
        self.assertEqual(sample["graph"]["buoy"].shape, (1, 12, 8))
        self.assertEqual(sample["graph"]["edge"].shape[-1], 3)
        self.assertEqual(sample["graph"]["edge_uu"].shape, (1, 6, 6, 2))
        self.assertEqual(sample["assignments"].shape[0], sample["j15_bits"].shape[0])
        self.assertGreater(sample["candidate_count"], 1000)
        self.assertTrue(np.all(np.isfinite(sample["j15_bits"])))


if __name__ == "__main__":
    unittest.main()
