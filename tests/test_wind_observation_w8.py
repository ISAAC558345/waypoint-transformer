"""One registered observation point has to keep the W7 mission and the message budget.

baseline replay: the original scheduler leaves the W7 delivery or energy.

budget match: the two arms do not send the same number of bytes.

timeline: a forecast uses the remote sample before the registered usable time.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w8 import run  # noqa: E402


class ObservationPointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run()

    def test_budget_timeline_and_baseline(self):
        summary = self.summary
        self.assertTrue(summary["baseline_replay"])
        self.assertTrue(summary["budget_match"])
        self.assertTrue(summary["timeline_ok"])
        self.assertFalse(summary["default_changed"])
        self.assertEqual(summary["timeline"], {"choose_s": 30, "sample_s": 60, "send_s": 60, "usable_s": 90, "score_s": [120, 150], "horizons_s": [30, 60]})
        self.assertEqual(summary["config_status"], "registered_before_run")
        config_path = ROOT / "configs" / "experiments" / "wind_observation_w8.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "registered_before_run")
        self.assertEqual(summary["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())
        self.assertEqual(summary["n_tasks"], 9)
        self.assertEqual(set(summary["opportunities"]), {"b3_loose", "b1_tight", "b1_window"})
        for task in summary["tasks"]:
            self.assertEqual(task["messages"] * 32, task["bytes"])
            self.assertIn(task["messages"], (0, 1))
            self.assertLessEqual(task["n_candidates"], 3)
            if not task["changed"]:
                self.assertEqual(task["extra_path_m"], 0.0)
                self.assertEqual(task["bits_baseline"], task["bits_experiment"])
            else:
                self.assertEqual(task["messages"], 1)
                self.assertNotEqual(task["chosen"], task["original"])
        pairs = json.loads((ROOT / "results" / "wind_observation_w8" / "w8_20261118" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(pairs["n_tasks"], summary["n_tasks"])
        by_key = {}
        for row in summary["tasks"]:
            by_key.setdefault((row["scenario_id"], row["field"], row["seed"]), row)
        self.assertEqual(len(by_key), 9)


if __name__ == "__main__":
    unittest.main()
