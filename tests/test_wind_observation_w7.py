"""A second lookahead has to run on a copy and leave the archived track untouched.

archive trajectory: delivery, energy, or the three recorded 60-second gaps move.

future isolated: forecast steps leave the belief mean that was known at the decision.

copy unchanged: the stored replay positions or beliefs change while forecasting.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w7 import run, select_added  # noqa: E402


class ScenarioRuleTest(unittest.TestCase):
    def test_the_name_order_picks_the_first_tight_and_window(self):
        rows = [
            {"id": "b3_loose", "condition": "loose"},
            {"id": "b1_window", "condition": "window"},
            {"id": "b2_tight", "condition": "tight"},
            {"id": "b1_tight", "condition": "tight"},
        ]
        found = [row["id"] for row in select_added(rows, "b3_loose")]
        self.assertEqual(found, ["b1_tight", "b1_window"])


class TrajectoryForecastTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run()

    def test_archive_and_copies_stay_in_place(self):
        summary = self.summary
        self.assertTrue(summary["archive_match"])
        self.assertTrue(summary["future_isolated"])
        self.assertTrue(summary["copy_unchanged"])
        self.assertFalse(summary["default_changed"])
        self.assertEqual(summary["added_scenario_ids"], ["b1_tight", "b1_window"])
        self.assertEqual([row["t_s"] for row in summary["known_gaps"]], [0, 30, 120])
        self.assertEqual(summary["config_status"], "registered_before_run")
        config_path = ROOT / "configs" / "experiments" / "wind_observation_w7.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "registered_before_run")
        self.assertEqual(summary["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())
        airborne = [row for row in summary["scores"] if row["phase"] == "airborne" and row["kind"] == "all" and row["arm"] == "a"]
        self.assertEqual([row["scenario_id"] for row in airborne], ["b3_loose", "b3_loose", "b1_tight", "b1_tight", "b1_window", "b1_window"])
        for row in airborne:
            if row["scenario_id"] == "b3_loose":
                self.assertGreater(row["n"], 0)


if __name__ == "__main__":
    unittest.main()
