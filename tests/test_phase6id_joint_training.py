"""Phase-6I-D metric and checkpoint-path checks. Training is not launched here."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import DecisionDataset
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.benchmark_v2.phase6id_joint_training.metrics import state_metrics
from wt.benchmark_v2.phase6id_joint_training.protocol import (
    ENCODER_CKPT,
    clear_improvement,
    selection_key,
    assert_output_path,
)


class Phase6IDTrainingTests(unittest.TestCase):
    def test_validation_index_excludes_reserved_scenarios(self) -> None:
        seen = {int(scenario) for scenario, _time_step in DecisionDataset("train").index + DecisionDataset("validation").index}
        self.assertTrue(seen.isdisjoint(RESERVED_TEST))
        self.assertEqual(min(seen), 322000)
        self.assertEqual(max(seen), 322063)

    def test_top1_uses_utility_rank_group_and_exact_alignment(self) -> None:
        labels = np.asarray([10.0, 9.0, 8.0, 1.0], dtype=np.float64)
        ranks = np.asarray([1, 1, 2, 3], dtype=np.int32)
        tied = state_metrics(np.asarray([0.0, 5.0, 4.0, 3.0]), labels, ranks)
        self.assertEqual(tied["top1"], 1.0)
        self.assertEqual(tied["chosen"], 1.0)
        self.assertAlmostEqual(tied["payload"], 9.0)
        self.assertAlmostEqual(tied["oracle_payload"], 10.0)
        reversed_scores = state_metrics(np.asarray([0.0, 1.0, 2.0, 3.0]), labels, ranks)
        self.assertEqual(reversed_scores["top1"], 0.0)
        self.assertEqual(reversed_scores["top5"], 1.0)
        self.assertGreater(reversed_scores["normalized_regret"], 0.0)

    def test_constant_scores_count_as_collapsed(self) -> None:
        labels = np.asarray([3.0, 2.0, 1.0], dtype=np.float64)
        ranks = np.asarray([1, 2, 3], dtype=np.int32)
        measured = state_metrics(np.asarray([1.0, 1.0, 1.0]), labels, ranks)
        self.assertEqual(measured["collapsed"], 1.0)
        self.assertTrue(np.isnan(measured["spearman"]))

    def test_selection_prefers_validation_rank_over_train_loss(self) -> None:
        weaker = {"spearman": 0.2, "top1": 1.0, "normalized_regret": 0.0, "loss": 0.01}
        stronger = {"spearman": 0.4, "top1": 0.5, "normalized_regret": 0.2, "loss": 1.0}
        self.assertGreater(selection_key(stronger), selection_key(weaker))

    def test_clear_improvement_requires_payload_and_rank(self) -> None:
        frozen = {"spearman": 0.1, "top1": 0.2, "normalized_regret": 0.4, "payload": 10.0}
        better = {"spearman": 0.3, "top1": 0.2, "normalized_regret": 0.3, "payload": 12.0}
        rank_only = {"spearman": 0.3, "top1": 0.2, "normalized_regret": 0.3, "payload": 10.0}
        self.assertTrue(clear_improvement(better, frozen))
        self.assertFalse(clear_improvement(rank_only, frozen))

    def test_original_checkpoint_path_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_output_path(ENCODER_CKPT)
        with self.assertRaises(RuntimeError):
            assert_output_path(Path("phase5l_joint_encoder") / "epoch_001.pt")


if __name__ == "__main__":
    unittest.main()
