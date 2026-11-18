"""The 60-second replay has to reproduce the W8.1 route before a later candidate is compared.

baseline replay: the original action after 60 seconds leaves the W8.1 packets or energy.

snapshot shared: a candidate does not start from the same 60-second queue.

budget timeline: the remote sample enters a forecast before 120 seconds, or the arms send different bytes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w82 import run  # noqa: E402


class LaterPacketReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run()

    def test_snapshot_budget_and_original_route(self):
        summary = self.summary
        self.assertTrue(summary["baseline_replay"])
        self.assertTrue(summary["snapshot_shared"])
        self.assertTrue(summary["budget_timeline"])
        self.assertTrue(summary["packets_on_uav0"])
        self.assertFalse(summary["default_changed"])
        self.assertEqual(summary["config_status"], "registered_before_run")
        self.assertEqual(summary["timeline"]["choose_s"], 60)
        self.assertEqual(summary["timeline"]["usable_s"], 120)
        config_path = ROOT / "configs" / "experiments" / "wind_observation_w82.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "registered_before_run")
        self.assertEqual(summary["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())
        self.assertEqual(summary["onboard_choice_s"], [0, 30, 60])
        self.assertEqual(summary["onboard_forecast_s"], [0, 30, 60, 90, 120])
        watched = {row["packet"] for row in summary["watched_baseline"]}
        self.assertEqual(watched, {"w4-b0-p0", "w4-b0-p1"})
        if summary["stopped"]:
            self.assertTrue(summary["stop_reason"])
            return
        self.assertEqual(summary["messages"], 1)
        self.assertEqual(summary["bytes"], 32)
        self.assertGreater(summary["airtime_s"], 0.0)
        self.assertLess(summary["airtime_s"], 30.0)
        actions = {row["action"] for row in summary["candidates"]}
        self.assertIn(summary["actions_at_choice"][0], actions)
        self.assertGreater(len(actions), 1)
        for row in summary["candidates"]:
            self.assertEqual(row["messages"], 1)
            self.assertEqual(row["bytes"], 32)
            self.assertEqual(len(row["watched"]), 2)


if __name__ == "__main__":
    unittest.main()
