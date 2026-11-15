"""The shared-link account has to reproduce the archived 240 s execution.

archive actions: a later request or energy differs from the W4 online log.

archive arrival: buoy 3 does not arrive at 336.258 s and 344.258 s.

solo quote: the partner-waits script misses 253.645 s and 261.645 s.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w42 import run  # noqa: E402


class SharedLinkTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run()

    def test_archived_execution_and_two_factor_split(self):
        self.assertTrue(self.summary["archive_match"])
        self.assertEqual(self.summary["queue_residual_bits"], 0.0)
        first = self.summary["splits"][0]
        self.assertAlmostEqual(first["shared_link_s"], 22.612912, places=6)
        self.assertEqual(first["reselection_s"], 60.0)
        self.assertAlmostEqual(first["residual_s"], 0.0, places=9)
        opening = self.summary["intervals"][0]
        self.assertEqual(opening["bits_sent"], [70000.0, 24800.0])
        self.assertAlmostEqual(opening["granted_air_s"][0], opening["requested_air_s"][0], places=6)
        self.assertLess(opening["granted_air_s"][1], opening["requested_air_s"][1])
        for interval in self.summary["intervals"][1:3]:
            self.assertEqual(interval["granted_air_s"], [0, 0])

    def test_joint_script_changes_admission_and_the_closed_loop_regresses(self):
        self.assertEqual(self.summary["solo_choice"]["action"], "backhaul:None")
        self.assertEqual(self.summary["joint_choice"]["action"], "return:None")
        self.assertTrue(self.summary["choice_changed"])
        self.assertFalse(self.summary["arrival_used_in_score"])
        joint = next(row for row in self.summary["joint_scores"] if row["action"] == "backhaul:None")
        self.assertFalse(joint["admitted"])
        self.assertLess(joint["terminal_j"], 4000.0)
        self.assertEqual(joint["bits"], 70000.0)
        self.assertAlmostEqual(self.summary["energy_delta_j"], 212.46798182465136, places=6)
        self.assertFalse(self.summary["recommend_more"])
        self.assertFalse(self.summary["default_changed"])


if __name__ == "__main__":
    unittest.main()
