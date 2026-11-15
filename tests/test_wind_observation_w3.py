"""Checks for the headwind return-energy audit.

interval buckets: the second-by-second sum misses the archived 270–300 s drop.

duration before wind: the flight is still several seconds longer than the cruise
quote after the truth-wind formula is substituted.

kinematic match: the controller rollout with the executor wind and theta misses
that same drop.

old formula: the analytic return quote changed by a constant residual.
"""

from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w1 import replay_case  # noqa: E402
from wt.active_model_identification.run_w2 import _read_csv, load_w2  # noqa: E402
from wt.active_model_identification.run_w21 import W2_DIR  # noqa: E402
from wt.active_model_identification.run_w3 import (  # noqa: E402
    iter_states,
    load_w3,
    project_terminal,
    reconcile_interval,
)
from wt.active_model_identification.wind import observe  # noqa: E402


class ReturnEnergyAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = replay_case()
        cls.w2 = load_w2()
        cls.cfg = load_w3()
        if not cls.replay["ok"]:
            raise AssertionError(cls.replay["mismatches"])
        cls.condition = next(row for row in cls.w2["conditions"] if row["id"] == "headwind_return")
        sample = cls.replay["sample"]
        rng = random.Random(f"w1-{cls.condition['id']}-{cls.w2['seed']}")
        cls.message = observe(cls.condition, sample["x_m"], sample["y_m"], sample["t_s"], cls.w2["field"], 0.5, rng)
        cls.rows = [row for row in _read_csv(W2_DIR / "decisions.csv") if row["condition"] == "headwind_return" and row["mode"] == "timely"]
        cls.world = None
        cls.beliefs = None
        cls.info = None
        for row, world, beliefs, info in iter_states(cls.replay, cls.condition, "timely", cls.message, cls.rows, cls.w2):
            if float(row["t_s"]) == 270.0:
                cls.world = world
                cls.beliefs = beliefs
                cls.info = info
                break
        cls.ledger = reconcile_interval(cls.world, cls.replay, cls.condition, cls.beliefs, cls.info, cls.w2)

    def test_interval_buckets_match_the_archived_drop(self):
        executed = self.ledger["executed"]
        total = executed["flight_j"] + executed["landing_j"] + executed["ground_j"] + executed["radio_j"]
        archived_300 = next(row for row in self.rows if float(row["t_s"]) == 300.0)
        self.assertAlmostEqual(self.ledger["energy_j"] - total, float(archived_300["energy_j"]), places=4)
        self.assertEqual(executed["flight_s"], 14)
        self.assertEqual(executed["landing_s"], 5)
        self.assertEqual(executed["ground_s"], 11)
        self.assertAlmostEqual(
            self.ledger["flight_gap_j"] + self.ledger["landing_gap_j"] + self.ledger["omitted_ground_and_radio_j"],
            self.ledger["raw_gap_j"],
            places=6,
        )

    def test_duration_gap_remains_after_the_truth_wind_formula(self):
        self.assertGreater(self.ledger["duration_gap_s"], 4.0)
        self.assertGreater(self.ledger["flight_gap_j"], 400.0)
        self.assertLess(abs(self.ledger["truth_wind_formula_shift_j"]), 100.0)
        self.assertLess(self.ledger["landing_gap_j"], 0.0)
        self.assertLess(self.ledger["omitted_ground_and_radio_j"], 20.0)

    def test_truth_rollout_matches_without_a_constant_offset(self):
        rolled = project_terminal(self.world, self.replay, self.condition, self.beliefs, self.info, self.w2, theta_mode="truth", wind_mode="truth")
        scenarios = _read_csv(W2_DIR / "scenarios.csv")
        archived = next(row for row in scenarios if row["condition"] == "headwind_return" and row["mode"] == "timely")
        label = json.loads(archived["uav_energy_j"])[1]
        self.assertAlmostEqual(rolled["terminal_energy_j"], label, delta=1.0)
        analytic = self.ledger["quotes"]["q95_predicted"]["flight_total_j"]
        self.assertLess(analytic, 3600.0)
        self.assertGreater(self.ledger["executed"]["flight_j"] + self.ledger["executed"]["landing_j"], analytic + 400.0)


if __name__ == "__main__":
    unittest.main()
