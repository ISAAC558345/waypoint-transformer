"""Accounting checks for the saved t=30 loss decomposition."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_t30_loss_attribution.analyze import (
    BIT_TIE,
    P4_RESULTS,
    VALIDATION_IDS,
    _read_csv,
    analyze_rows,
    classify_tail_effect,
    dominant_interval,
)


class T30LossAttributionTest(unittest.TestCase):
    def test_tail_effect_labels(self):
        self.assertEqual(classify_tail_effect(-10.0, -4.0), "enlarges")
        self.assertEqual(classify_tail_effect(-10.0, 4.0), "offsets")
        self.assertEqual(classify_tail_effect(0.0, 0.0), "tie")
        self.assertEqual(dominant_interval(-10.0, -3.0), "window")
        self.assertEqual(dominant_interval(-2.0, -9.0), "tail")
        self.assertEqual(dominant_interval(0.0, 0.0), "neither")

    def test_saved_p4_tables_balance(self):
        result = analyze_rows(_read_csv(P4_RESULTS / "scenario_pairs.csv"), _read_csv(P4_RESULTS / "uav_window_rows.csv"))
        self.assertEqual(result["gate"], "T30_LOSS_ATTRIBUTION_COMPLETE")
        self.assertEqual(result["scenario_ids"], list(VALIDATION_IDS))
        self.assertNotIn(322064, result["scenario_ids"])
        self.assertFalse(result["reserved_test_used"])
        self.assertFalse(result["replay_performed"])
        self.assertFalse(result["scheme_c_score"]["future_payload_used_to_select"])
        self.assertLessEqual(result["accounting"]["max_identity_error_bits"], BIT_TIE)
        self.assertEqual(result["n_changed_scenarios"], 10)
        for row in result["scenarios"]:
            self.assertLessEqual(abs(row["delta_episode_bits"] - row["delta_window_bits"] - row["delta_tail_bits"]), BIT_TIE)
            self.assertLessEqual(abs(row["reassigned_window_bits"] + row["fixed_window_bits"] - row["delta_window_bits"]), BIT_TIE)
            self.assertLessEqual(abs(row["reassigned_tail_bits"] + row["fixed_tail_bits"] - row["delta_tail_bits"]), BIT_TIE)
        changed = {row["scenario_id"] for row in result["scenarios"] if row["comparison_status"] == "selected"}
        reassigned = [row for row in result["uavs"] if row["scenario_id"] in changed and row["group"] == "reassigned"]
        fixed = [row for row in result["uavs"] if row["scenario_id"] in changed and row["group"] == "fixed"]
        self.assertEqual(len(reassigned) + len(fixed), 60)
        self.assertTrue(all(row["t30_assignment_changed"] == 0 for row in fixed))
        self.assertTrue(all(row["t30_target_scheme"] == row["t30_target_preserve"] for row in fixed))

    def test_broken_episode_identity_is_blocked(self):
        scenarios = [dict(row) for row in _read_csv(P4_RESULTS / "scenario_pairs.csv")]
        scenarios[0]["delta_episode_bits"] = str(float(scenarios[0]["delta_episode_bits"]) + 10.0)
        result = analyze_rows(scenarios, _read_csv(P4_RESULTS / "uav_window_rows.csv"))
        self.assertEqual(result["gate"], "T30_LOSS_ATTRIBUTION_BLOCKED")
        self.assertGreater(result["accounting"]["max_identity_error_bits"], BIT_TIE)

    def test_analysis_source_has_no_decision_entry(self):
        source = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_t30_loss_attribution" / "analyze.py"
        self.assertNotIn("decide(", source.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
