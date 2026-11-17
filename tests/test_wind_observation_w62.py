"""A second wind sample has to share the receiver latent and the remote arrival time.

archive ab: onboard and onboard-plus-remote forecasts leave W6.1.

same latent once: the colocated measurement carries a different latent value, or a repeated id is accepted.

second on time: the second sample is already in the forecast at the sampling decision.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w6 import _predict  # noqa: E402
from wt.active_model_identification.run_w62 import _colocated, _same_receiver_latent, _schedule, run  # noqa: E402


class SecondSampleTest(unittest.TestCase):
    def test_the_second_sample_waits_and_reuses_the_receiver_latent(self):
        decisions = [
            {"t": 0.0, "actual": {30: (0.0, 0.0)}, "predicted": {30: (0.0, 0.0)}},
            {"t": 30.0, "actual": {30: (1.0, 1.0)}, "predicted": {30: (1.0, 1.0)}},
        ]
        onboard = [{"id": 100000, "source": "onboard", "t": 0.0, "x": 10.0, "y": 20.0, "rx": 6.1, "ry": 0.2}]
        second = [{"id": 200000, "source": "remote", "t": 0.0, "x": 10.0, "y": 20.0, "rx": 6.3, "ry": -0.4}]
        frames = _schedule(decisions, onboard, second, 0.082176)
        self.assertEqual([item["id"] for item in frames[0]], [100000])
        self.assertEqual([item["id"] for item in frames[1]], [100000, 200000])
        with self.assertRaises(RuntimeError):
            _schedule(decisions, onboard, [second[0], dict(second[0])], 0.082176)
        site = {"t": 0.0, "x": 10.0, "y": 20.0}
        sample = {"id": 7, "t": 0.0, "x": 80.0, "y": 90.0, "rx": 6.3, "ry": -0.4}
        table = {(0, 1): (1.5, -2.0), (0, 0): (4.0, 5.0)}
        colocated = _colocated(site, sample, table, 6.0, 0.0, 200000, (0.3, -0.4))
        self.assertEqual(colocated["id"], 200000)
        self.assertEqual((colocated["x"], colocated["y"], colocated["t"]), (10.0, 20.0, 0.0))
        self.assertAlmostEqual(colocated["rx"], 6.0 + 0.3 + 1.5, places=12)
        self.assertAlmostEqual(colocated["ry"], 0.0 + -0.4 + -2.0, places=12)
        receiver = {"id": 100000, "source": "onboard", "t": 0.0, "x": 10.0, "y": 20.0, "rx": 6.0 + 0.1 + 1.5, "ry": -2.0}
        _same_receiver_latent([receiver], [colocated], [(0.1, 0.0)], [(0.3, -0.4)], 6.0, 0.0)
        kernel = {"prior_std_m_s": 8.0, "sigma_m_s": 0.5, "amplitude_m_s": 0.0, "length_m": 600.0, "time_s": 90.0}
        with self.assertRaises(RuntimeError):
            _predict([colocated], colocated, kernel)


class LocationComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run()

    def test_archived_arms_match_and_the_second_sample_stays_paired(self):
        summary = self.summary
        self.assertTrue(summary["archive_match"])
        self.assertTrue(summary["same_latent_once"])
        self.assertTrue(summary["second_on_time"])
        self.assertTrue(summary["fixed_global_match"])
        self.assertFalse(summary["default_changed"])
        self.assertLess(summary["fixed_global_max_abs_m_s"], 1e-6)
        self.assertEqual(summary["n_holdout_fields"], 4)
        self.assertEqual(summary["holdout_seeds"], [11041, 11042, 11043, 11044])
        self.assertEqual(summary["n_flight_decisions"], 8)
        self.assertEqual(summary["config_status"], "registered_before_run")
        config_path = ROOT / "configs" / "experiments" / "wind_observation_w62.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "registered_before_run")
        self.assertEqual(summary["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())
        flight = [row for row in summary["scores"] if row["subset"] == "flight"]
        self.assertEqual(len(flight), 18)
        for row in flight:
            self.assertEqual(row["n"], 8)
            self.assertIsNotNone(row["mean_paired_sq_b_minus_d"])


if __name__ == "__main__":
    unittest.main()
