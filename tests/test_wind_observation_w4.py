"""Checks for the early wind-message comparison.

headwind continuity: a zero-wind prefix was spliced in, so the first interval
still has the frozen energy and the later buoy-3 request.

message airtime: the sample is visible at the decision that takes it, before
the transmission has finished.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w1 import M16_DIR, _read_csv  # noqa: E402
from wt.active_model_identification.run_w4 import _condition, _visible, baseline, build_link, load_w4  # noqa: E402


class EarlyWindMessageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.case = load_w4()
        cls.condition = _condition(cls.case, "headwind_return")
        cls.played = baseline(cls.case, cls.condition)
        cls.link = build_link(cls.played, cls.case, cls.condition, cls.cfg)

    def test_headwind_from_takeoff_is_not_the_zero_wind_prefix(self):
        frozen = _read_csv(M16_DIR / "decisions.csv")
        calm = next(
            row
            for row in frozen
            if row.get("branch") == "core"
            and row["scenario_id"] == "b3_loose"
            and row["method"] == "lookahead_calibration_off"
            and int(row["epoch"]) == 0
            and int(row["uav"]) == 0
        )
        found = next(row for row in self.played["rows"] if row["epoch"] == 0 and row["uav"] == 0)
        partner = next(row for row in self.played["rows"] if row["epoch"] == 0 and row["uav"] == 1)
        self.assertEqual(found["requested"], "service:1")
        self.assertEqual(partner["requested"], "service:2")
        self.assertGreater(found["energy_j"] - float(calm["energy_j"]), 100.0)
        later = next(row for row in self.played["rows"] if row["epoch"] == 4 and row["uav"] == 0)
        calm_later = next(
            row
            for row in frozen
            if row.get("branch") == "core"
            and row["scenario_id"] == "b3_loose"
            and row["method"] == "lookahead_calibration_off"
            and int(row["epoch"]) == 4
            and int(row["uav"]) == 0
        )
        self.assertNotEqual(later["requested"], calm_later["requested"])
        self.assertEqual(self.link["team_commit_t_s"], 30.0)
        self.assertEqual(self.link["team_commit_uav"], 0)

    def test_sample_is_not_usable_before_the_airtime(self):
        self.assertGreater(self.link["sample_t_s"], 0.0)
        self.assertGreater(self.link["timely_t_s"], self.link["sample_t_s"])
        self.assertEqual(self.link["late_t_s"], self.link["timely_t_s"] + 30.0)
        self.assertFalse(_visible("timely", 0.0, self.link))
        self.assertFalse(_visible("timely", self.link["sample_t_s"], self.link))
        self.assertFalse(_visible("no_send", self.link["late_t_s"], self.link))
        self.assertTrue(_visible("timely", self.link["timely_t_s"], self.link))
        self.assertFalse(_visible("late", self.link["timely_t_s"], self.link))
        self.assertGreaterEqual(self.link["team_shortfall_s"], self.link["airtime_s"])


if __name__ == "__main__":
    unittest.main()
