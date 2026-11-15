"""Onboard and remote wind samples have to share one field and the W6 arrival rule.

same field: a coincident coordinate is redrawn, or the conditional mean uses the global prior.

remote on time: an onboard sample waits, or a remote sample is usable in the beat it is sent.

remote baseline: the remote-only arm leaves the archived W6 errors.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w61 import (  # noqa: E402
    _conditional_alpha,
    _extend_tables,
    _find_point,
    observation_is_usable,
    run,
)


class AvailabilityTest(unittest.TestCase):
    def test_onboard_is_immediate_and_remote_waits_for_airtime(self):
        airtime = 0.082176
        self.assertTrue(observation_is_usable("onboard", 0.0, 0.0, airtime))
        self.assertFalse(observation_is_usable("remote", 0.0, 0.0, airtime))
        self.assertTrue(observation_is_usable("remote", 0.0, 30.0, airtime))
        self.assertFalse(observation_is_usable("onboard_delayed", 0.0, 0.0, airtime))
        self.assertTrue(observation_is_usable("onboard_delayed", 0.0, 30.0, airtime))


class ConditionalDrawTest(unittest.TestCase):
    def test_conditional_mean_uses_only_the_local_kernel(self):
        kernel = {"prior_std_m_s": 8.0, "sigma_m_s": 0.5, "amplitude_m_s": 2.0, "length_m": 600.0, "time_s": 90.0}
        points = [{"key": (0, 0), "id": 1, "x": 0.0, "y": 0.0, "t": 0.0}]
        alpha, std = _conditional_alpha(points, {"x": 0.0, "y": 0.0, "t": 10.0}, kernel, 1e-9)
        variance = 4.0 + 1e-9
        cross = 4.0 * math.exp(-10.0 / 90.0)
        self.assertAlmostEqual(alpha[0], cross / variance, places=12)
        self.assertAlmostEqual(std, math.sqrt(variance - cross * cross / variance), places=12)

    def test_coincident_coordinate_reuses_the_archived_point(self):
        points = [{"key": (0, 0), "x": 5.0, "y": 7.0, "t": 0.0}]
        status, point = _find_point(points, {"t": 0.0, "x": 5.0, "y": 7.0})
        self.assertEqual(status, "coincident")
        self.assertEqual(point["key"], (0, 0))
        archived, exact = _find_point([{"key": (0, 1), "x": 5.0, "y": 7.0, "t": 0.0}], {"t": 0.0, "x": 5.0, "y": 7.0})
        self.assertEqual(archived, "archived")
        self.assertEqual(exact["key"], (0, 1))
        with self.assertRaises(RuntimeError):
            _find_point([{"key": (0, 1), "x": 5.0, "y": 7.0, "t": 0.0}], {"t": 0.0, "x": 80.0, "y": 7.0})

    def test_a_missing_point_keeps_the_archived_value_and_is_deterministic(self):
        kernel = {"prior_std_m_s": 8.0, "sigma_m_s": 0.5, "amplitude_m_s": 2.0, "length_m": 600.0, "time_s": 90.0}
        points = [{"key": (0, 0), "id": 1, "x": 0.0, "y": 0.0, "t": 0.0}]
        tables = {11031: {(0, 0): (1.5, -0.25)}}
        sites = [{"t": 10.0, "x": 0.0, "y": 0.0}]
        extended, rows = _extend_tables(points, tables, sites, kernel, 1e-9, "unit")
        again, _rows = _extend_tables(points, tables, sites, kernel, 1e-9, "unit")
        self.assertEqual(rows[0]["status"], "drawn")
        self.assertEqual(tables[11031][(0, 0)], (1.5, -0.25))
        self.assertEqual(extended[11031][(10, 1)], again[11031][(10, 1)])
        alpha, std = _conditional_alpha(points, {"x": 0.0, "y": 0.0, "t": 10.0}, kernel, 1e-9)
        east = random.Random("unit-11031-x-10").gauss(0.0, 1.0)
        north = random.Random("unit-11031-y-10").gauss(0.0, 1.0)
        self.assertAlmostEqual(extended[11031][(10, 1)][0], alpha[0] * 1.5 + std * east, places=12)
        self.assertAlmostEqual(extended[11031][(10, 1)][1], alpha[0] * -0.25 + std * north, places=12)


class WindComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run()

    def test_remote_arm_matches_w6_and_the_samples_stay_paired(self):
        summary = self.summary
        self.assertTrue(summary["same_field"])
        self.assertTrue(summary["remote_on_time"])
        self.assertTrue(summary["archive_match"])
        self.assertFalse(summary["default_changed"])
        self.assertEqual(summary["n_remote_messages"], 30)
        self.assertEqual(summary["n_onboard_samples"], 30)
        self.assertEqual(summary["payload_bytes"], 32)
        self.assertEqual(summary["extra_bytes_vs_onboard"], 960)
        self.assertEqual(summary["n_flight_decisions"], 8)
        self.assertEqual(summary["n_drawn_points"] + summary["n_reused_points"], 30)
        self.assertGreater(summary["separation_max_m"], 1.0)
        self.assertEqual(summary["config_status"], "registered_before_run")
        config_path = ROOT / "configs" / "experiments" / "wind_observation_w61.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "registered_before_run")
        self.assertEqual(summary["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())
        self.assertEqual(config["prior_weight_global"], 0.5)
        self.assertEqual(config["prior_weight_local"], 0.5)
        self.assertEqual(config["onboard_noise_token"], "w61-onboard-11030")
        flight = [row for row in summary["scores"] if row["subset"] == "flight"]
        self.assertEqual(len(flight), 10)
        for row in flight:
            self.assertEqual(row["n"], 8)


if __name__ == "__main__":
    unittest.main()
