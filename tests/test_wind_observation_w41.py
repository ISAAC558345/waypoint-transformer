"""The 240 s choice has to be the archived W4 decision.

archive choice: the restored request, execution, or reason differs from the
W4 online log.

archive endpoint: the replayed team energy or on-time bits differ from the
saved arm.

forced backhaul: specifying the archived backhaul and continuing the same
policy misses the timely endpoint.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w41 import run  # noqa: E402


class BackhaulAttributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run()

    def test_replay_matches_the_archived_choice_and_endpoint(self):
        self.assertTrue(self.summary["archive_match"])
        states = {row["mode"]: row for row in self.summary["states"]}
        self.assertEqual(states["no_send"]["requested"], "return:None")
        self.assertEqual(states["timely"]["requested"], "backhaul:None")
        self.assertFalse(states["no_send"]["guard_rewrote"])
        self.assertFalse(states["timely"]["guard_rewrote"])
        self.assertLess(abs(self.summary["unexplained_j"]), 1e-6)
        self.assertAlmostEqual(self.summary["message_gap_j"], 0.01038622464, places=9)

    def test_message_changes_admission_not_a_tie(self):
        blocked = next(row for row in self.summary["toggled"]["none"]["rows"] if row["action"] == "backhaul:None")
        admitted = next(row for row in self.summary["toggled"]["message"]["rows"] if row["action"] == "backhaul:None")
        self.assertFalse(blocked["admitted"])
        self.assertLess(blocked["terminal_j"], 4000.0)
        self.assertTrue(admitted["admitted"])
        self.assertGreater(admitted["terminal_j"], 4000.0)
        self.assertEqual(self.summary["toggled"]["none"]["tied"], ["return:None"])
        self.assertEqual(self.summary["toggled"]["message"]["chosen"], "backhaul:None")
        self.assertFalse(self.summary["ranking_opposite"])
        self.assertFalse(self.summary["default_changed"])


if __name__ == "__main__":
    unittest.main()
