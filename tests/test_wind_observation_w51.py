"""Equal-budget send timing has to keep the saved forecast and one message per pair.

archive forecast: a moved score coordinate changes the saved wind vector.

one per pair: early, late, or online sends a count other than 30.

late position: the second beat is usable at the same decision that sends it.

archive rmse: the offline latest-sample replay leaves the W5 table.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w51 import (  # noqa: E402
    message_is_usable,
    pair_slots,
    pick_beat,
    run,
)


class TimingRuleTest(unittest.TestCase):
    def test_tie_sends_on_the_first_beat(self):
        self.assertEqual(pick_beat(1.0, 1.0, 1e-9), "early")
        self.assertEqual(pick_beat(1.0, 1.0 - 1e-12, 1e-9), "early")
        self.assertEqual(pick_beat(2.0, 1.0, 1e-9), "late")
        self.assertEqual(pick_beat(1.0, 1.1, 1e-9), "early")

    def test_a_message_waits_until_the_next_decision(self):
        self.assertFalse(message_is_usable(0.0, 0.0, 0.082176))
        self.assertTrue(message_is_usable(0.0, 30.0, 0.082176))
        self.assertFalse(message_is_usable(30.0, 30.0, 0.082176))
        self.assertTrue(message_is_usable(30.0, 60.0, 0.082176))

    def test_mission_end_drops_the_late_message_window(self):
        self.assertEqual(pair_slots(0.0, 1800.0, 1770.0), [(30.0, 30), (30.0, 60), (60.0, 30), (60.0, 60)])
        self.assertEqual(pair_slots(1740.0, 1800.0, 1770.0), [(1770.0, 30)])


class EqualBudgetTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run()

    def test_archive_budget_and_forecast_boundary(self):
        summary = self.summary
        self.assertTrue(summary["trajectory_match"])
        self.assertTrue(summary["archive_rmse_match"])
        self.assertTrue(summary["sample_choice_still_identical"])
        self.assertTrue(summary["forecast_unchanged_when_score_point_moves"])
        self.assertTrue(summary["error_changes_when_score_point_moves"])
        self.assertFalse(summary["default_changed"])
        self.assertEqual(summary["send_count"], {"early": 30, "late": 30, "online": 30, "reference60": 60})
        self.assertEqual(summary["payload_bytes"], 32)
        self.assertEqual(summary["n_pairs"], 30)
        self.assertEqual(summary["usable_sends"]["early"], 30)
        self.assertEqual(summary["usable_sends"]["late"], 29)
        self.assertEqual(summary["usable_sends"]["reference60"], 59)
        self.assertEqual(len(summary["position_mismatch_t_s"]), 3)
        self.assertEqual(summary["config_status"], "registered_before_run")
        config_path = ROOT / "configs" / "experiments" / "wind_observation_w51.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "registered_before_run")
        self.assertEqual(summary["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())
        for row in summary["scores"]:
            if row["subset"] == "full" and row["horizon_s"] == 30:
                self.assertAlmostEqual(row["rmse_reference60"], row["rmse_reference60_offline"], places=6)
        local_counts = {row["online_early_sends"] for row in summary["scores"] if row["field"] == "local"}
        self.assertEqual(len(local_counts), 1)


if __name__ == "__main__":
    unittest.main()
