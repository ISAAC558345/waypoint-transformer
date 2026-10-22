"""Bootstrap and tie-break rules. The validation scoring pass is not run here."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import DecisionDataset
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.benchmark_v2.phase6ig_candidate_audit.audit import (
    assign_gate,
    cluster_means,
    decision_view,
)


class Phase6IGAuditTests(unittest.TestCase):
    def test_validation_index_excludes_reserved_scenarios(self) -> None:
        seen = {int(scenario) for scenario, _time in DecisionDataset("validation").index}
        self.assertTrue(seen.isdisjoint(RESERVED_TEST))
        self.assertEqual(seen, set(range(322048, 322064)))

    def test_cluster_resample_keeps_a_scenario_times_together(self) -> None:
        groups = [np.asarray([1.0, 3.0]), np.asarray([10.0, 30.0])]
        samples = cluster_means(groups, draws=200, seed=0)
        self.assertEqual(samples.shape, (200,))
        self.assertTrue(np.all(np.isfinite(samples)))
        repeated = cluster_means([np.asarray([1.0, 3.0])], draws=20, seed=1)
        self.assertTrue(np.allclose(repeated, 2.0))

    def test_reverse_tie_break_can_change_a_hit(self) -> None:
        labels = np.asarray([5.0, 4.0, 0.0], dtype=np.float64)
        ranks = np.asarray([1, 2, 3], dtype=np.int64)
        view = decision_view(np.asarray([1.0, 1.0, 0.0]), labels, ranks)
        self.assertEqual(view["chosen"], 0)
        self.assertEqual(view["score_tie_count"], 2)
        self.assertEqual(view["tie_break_changes_top1"], 1)
        self.assertEqual(view["best_group_in_top5_not_top1"], 0)

    def test_optimum_inside_top5_but_not_selected_is_flagged(self) -> None:
        labels = np.asarray([0.0, 0.0, 9.0, 1.0, 1.0, 8.0], dtype=np.float64)
        ranks = np.asarray([3, 3, 1, 2, 2, 2], dtype=np.int64)
        scores = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float64)
        view = decision_view(scores, labels, ranks)
        self.assertEqual(view["metric"]["top1"], 0.0)
        self.assertEqual(view["metric"]["top5"], 1.0)
        self.assertEqual(view["best_group_in_top5_not_top1"], 1)
        self.assertEqual(view["exact_max_predicted_rank"], 3)

    def test_support_rule_rejects_a_zero_median(self) -> None:
        positive = {
            "mean_ci95": [10.0, 20.0],
            "median": 0.0,
            "scenarios_positive": 16.0,
            "n_scenarios": 16.0,
        }
        frozen = {"mean_ci95": [10.0, 20.0]}
        times = [{"mean_ci95": [-1.0, 5.0]}, {"mean_ci95": [1.0, 2.0]}, {"mean_ci95": [1.0, 2.0]}]
        self.assertEqual(assign_gate(positive, frozen, times), "RANKING_CANDIDATE_GAIN_INCONCLUSIVE")
        positive["median"] = 2.0
        positive["scenarios_positive"] = 8.0
        self.assertEqual(assign_gate(positive, frozen, times), "RANKING_CANDIDATE_GAIN_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
