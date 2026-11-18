"""Packet tracing has to reproduce the frozen W8 delivery before a new candidate is compared.

baseline reuse: the original action leaves the W8 delivery or energy.

chosen reuse: service:1 leaves the W8 delivery, energy, or wind error.

budget timeline: a candidate sends a different message, or a score uses the remote sample early.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w81 import run  # noqa: E402


class PacketReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run()

    def test_reuse_budget_and_both_tables(self):
        summary = self.summary
        self.assertTrue(summary["baseline_reuse"])
        self.assertTrue(summary["chosen_reuse"])
        self.assertTrue(summary["budget_timeline"])
        self.assertFalse(summary["default_changed"])
        self.assertEqual(summary["config_status"], "registered_before_run")
        config_path = ROOT / "configs" / "experiments" / "wind_observation_w81.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "registered_before_run")
        self.assertEqual(summary["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())
        self.assertEqual(summary["messages"], 1)
        self.assertEqual(summary["bytes"], 32)
        actions = {row["action"] for row in summary["candidates"]}
        self.assertEqual(actions, {"service:0", "service:1", "service:3"})
        self.assertEqual(len(summary["candidates"]), 9)
        self.assertGreater(summary["packet_diff_counts"]["service:1"]["missing"], 0)
        self.assertIsNotNone(summary["earliest"]["service:1"])
        for row in summary["candidates"]:
            self.assertEqual(row["messages"], 1)
            self.assertEqual(row["bytes"], 32)


if __name__ == "__main__":
    unittest.main()
