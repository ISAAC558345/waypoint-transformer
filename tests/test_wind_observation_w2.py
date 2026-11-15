"""Necessary checks for the W2 rolling wind decision.

opening: a different 180 s candidate, score, or first choice means the rolling
entry changed the W1 quote.

late visibility: a late message that changes the 180 s action, or is still
absent at 210 s, has leaked or missed the registered deadline.

ledger: a second radio charge, or the same cached packet reaching shore twice,
means an interval copied the previous account.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w1 import replay_case  # noqa: E402
from wt.active_model_identification.run_w2 import (  # noqa: E402
    compare_opening,
    load_w1_archive,
    load_w2,
    roll_arm,
    visible,
)
from wt.active_model_identification.wind import observe  # noqa: E402


class WindObservationW2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = replay_case()
        cls.cfg = load_w2()
        cls.archive = load_w1_archive()
        if not cls.replay["ok"]:
            raise AssertionError(cls.replay["mismatches"])

    def _message(self, condition):
        cfg = self.cfg
        rng = random.Random(f"w1-{condition['id']}-{cfg['seed']}")
        sample = self.replay["sample"]
        return observe(
            condition,
            sample["x_m"],
            sample["y_m"],
            sample["t_s"],
            cfg["field"],
            float(cfg["observation_sigma_m_s"]),
            rng,
        )

    def test_opening_matches_archived_w1(self):
        sealed = [uav.consumed_j for uav in self.replay["prefix"].uavs]
        for condition in self.cfg["conditions"]:
            message = self._message(condition)
            for mode in self.cfg["modes"]:
                arm = roll_arm(self.replay, condition, mode, message, self.cfg, stop_after_s=float(self.cfg["decision_time_s"]))
                mismatches = compare_opening(arm["decisions"][0]["rows"], self.archive["candidates"][(condition["id"], mode)])
                self.assertFalse(mismatches, mismatches)
                self.assertEqual(arm["decisions"][0]["used_message"], mode == "timely")
        self.assertEqual(sealed, [uav.consumed_j for uav in self.replay["prefix"].uavs])

    def test_late_message_is_hidden_at_180_and_read_at_210(self):
        self.assertFalse(visible("late", 180.0, self.cfg))
        self.assertTrue(visible("late", 210.0, self.cfg))
        self.assertFalse(visible("no_send", 210.0, self.cfg))
        self.assertTrue(visible("timely", 180.0, self.cfg))
        condition = self.cfg["conditions"][0]
        arm = roll_arm(self.replay, condition, "late", self._message(condition), self.cfg, stop_after_s=210.0)
        self.assertFalse(arm["decisions"][0]["used_message"])
        self.assertEqual(arm["decisions"][0]["t_s"], 180.0)
        self.assertTrue(arm["decisions"][1]["used_message"])
        self.assertEqual(arm["decisions"][1]["t_s"], 210.0)
        self.assertEqual(arm["charge_count"], 1)
        ids = [row["id"] for row in arm["packets"]]
        self.assertEqual(ids, list(dict.fromkeys(ids)))
        self.assertEqual(len(ids), 2)

    def test_rolling_ledger_charges_the_message_once(self):
        condition = self.cfg["conditions"][1]
        arm = roll_arm(self.replay, condition, "timely", self._message(condition), self.cfg, stop_after_s=210.0)
        self.assertEqual(arm["charge_count"], 1)
        self.assertAlmostEqual(arm["message_j"], arm["message_j_nominal"], places=9)
        self.assertEqual(len({row["id"] for row in arm["packets"]}), 2)


if __name__ == "__main__":
    unittest.main()
