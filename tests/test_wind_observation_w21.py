"""Checks for the landed-backhaul audit.

archived path: a replay that misses the W2 battery is not the old trajectory.

old guard: a different 300 s request or rewrite means the default guard moved.

airborne beat: the explicit option changed the guard while the vehicle was still flying.

landed quote: cruise or a new landing hover is still inside the ground quote.

terminal gate: a backhaul is released after idle through the deadline already misses the reserve.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w1 import replay_case  # noqa: E402
from wt.active_model_identification.run_w2 import _read_csv, load_w2, roll_arm  # noqa: E402
from wt.active_model_identification.run_w21 import (  # noqa: E402
    W2_DIR,
    audit_landing,
    load_w21,
    trace_archived,
)
from wt.active_model_identification.wind import (  # noqa: E402
    finished_landing,
    make_finished_landing_guard,
    make_return_guard,
    observe,
    predict,
    return_energy_j,
    return_energy_parts,
)


class LandedBackhaulAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = replay_case()
        cls.w2 = load_w2()
        cls.cfg = load_w21()
        if not cls.replay["ok"]:
            raise AssertionError(cls.replay["mismatches"])
        cls.condition = next(row for row in cls.w2["conditions"] if row["id"] == "headwind_return")
        rng = random.Random(f"w1-{cls.condition['id']}-{cls.w2['seed']}")
        sample = cls.replay["sample"]
        cls.message = observe(
            cls.condition,
            sample["x_m"],
            sample["y_m"],
            sample["t_s"],
            cls.w2["field"],
            float(cls.w2["observation_sigma_m_s"]),
            rng,
        )
        cls.rows = [
            row
            for row in _read_csv(W2_DIR / "decisions.csv")
            if row["condition"] == "headwind_return" and row["mode"] == "timely"
        ]
        cls.trace = trace_archived(cls.replay, cls.condition, "timely", cls.message, cls.rows, cls.w2)
        cls.at_270 = roll_arm(
            cls.replay, cls.condition, "timely", cls.message, cls.w2, stop_after_s=270.0, capture_s=270.0
        )
        cls.at_300 = roll_arm(
            cls.replay, cls.condition, "timely", cls.message, cls.w2, stop_after_s=300.0, capture_s=300.0
        )

    def test_archived_replay_matches_w2_energy(self):
        self.assertEqual(len(self.trace), len(self.rows))
        at_300 = next(row for row in self.trace if row["t_s"] == 300.0)
        archived = next(row for row in self.rows if float(row["t_s"]) == 300.0)
        self.assertAlmostEqual(at_300["energy_j"], float(archived["energy_j"]), places=6)
        self.assertLess(at_300["posthoc_idle_upper_j"], float(self.cfg["reserve_j"]))
        at_270 = next(row for row in self.trace if row["t_s"] == 270.0)
        self.assertGreaterEqual(at_270["posthoc_idle_upper_j"], float(self.cfg["reserve_j"]))

    def test_old_guard_reproduces_the_300s_rewrite(self):
        decision = next(row for row in self.at_300["decisions"] if row["t_s"] == 300.0)
        self.assertEqual(decision["chosen"], "backhaul:None")
        self.assertEqual(decision["requested"], "backhaul:None")
        self.assertEqual(decision["executed"], "return:None")
        self.assertEqual(decision["block"], "return_guard")
        self.assertTrue(decision["landed"])
        audit = audit_landing(self.at_300["captured"], self.replay, self.condition, self.w2)
        self.assertTrue(audit["finished_landing"])
        self.assertTrue(audit["old_guard_rewrote"])
        self.assertEqual(audit["old_guard_action"], "return:None")
        self.assertTrue(audit["can_backhaul_now_under_old_admission"])
        self.assertFalse(audit["can_hold_reserve_to_deadline"])

    def test_option_does_not_change_the_airborne_guard(self):
        captured = self.at_270["captured"]
        world = captured["world"]
        spec = self.replay["case"]["spec"]
        uav = world.uavs[1]
        self.assertFalse(finished_landing(uav, spec))
        self.assertTrue(uav.airborne)
        old = make_return_guard(captured["info"], self.w2["field"], 8.0, spec, self.replay["case"]["factor"], 1)
        option = make_finished_landing_guard(captured["info"], self.w2["field"], 8.0, spec, self.replay["case"]["factor"], 1)
        belief = captured["beliefs"][1]
        self.assertEqual(old(uav, ("backhaul", None), belief, world), option(uav, ("backhaul", None), belief, world))

    def test_landed_quote_drops_fictional_flight_and_keeps_the_reserve_gate(self):
        audit = audit_landing(self.at_300["captured"], self.replay, self.condition, self.w2)
        ground = audit["ground"]
        self.assertEqual(ground["cruise_j"], 0.0)
        self.assertEqual(ground["landing_j"], 0.0)
        self.assertAlmostEqual(ground["idle_j"], 1500.0, places=6)
        self.assertGreater(ground["radio_j"], 0.0)
        self.assertGreater(audit["guard_total_j"], 100.0)
        wind = predict(self.at_300["captured"]["info"], audit["x_m"], audit["y_m"], audit["t_s"], self.w2["field"], 8.0)["wind_xy"]
        uav = self.at_300["captured"]["world"].uavs[1]
        theta = self.at_300["captured"]["beliefs"][1].quantile(0.95)
        parts = return_energy_parts(uav, theta, self.replay["case"]["spec"], self.replay["case"]["factor"], wind)
        self.assertAlmostEqual(parts["total_j"], return_energy_j(uav, theta, self.replay["case"]["spec"], self.replay["case"]["factor"], wind), places=9)
        self.assertAlmostEqual(audit["guard_cruise_j"], parts["cruise_j"], places=6)
        self.assertAlmostEqual(audit["guard_landing_j"], parts["landing_j"], places=6)
        self.assertGreater(parts["cruise_j"], 0.0)
        self.assertGreater(parts["landing_j"], 0.0)
        self.assertFalse(ground["holds_reserve_without_radio"])
        self.assertFalse(ground["holds_reserve_with_radio"])
        self.assertTrue(audit["option_guard_rewrote"])
        self.assertEqual(audit["option_guard_action"], "return:None")


if __name__ == "__main__":
    unittest.main()
