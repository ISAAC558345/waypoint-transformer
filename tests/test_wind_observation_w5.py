"""Route-aware sample selection has to stay inside the registered information.

future isolation: a sample taken after the send time is accepted.

zero amplitude: amplitude 0 prefers a closer sample, or the posterior depends on location.

equal budget: the sending methods use different payload counts on the frozen trajectory.

frozen trajectory: the replay leaves the archived W4 bits or energy.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w5 import (  # noqa: E402
    global_mean,
    posterior_mean,
    run,
    select_sample,
)

KERNEL = {
    "prior_std_m_s": 8.0,
    "sigma_m_s": 0.5,
    "amplitude_m_s": 0.0,
    "length_m": 600.0,
    "time_s": 90.0,
}


def _samples():
    return [
        {"id": 0, "t": 0.0, "x": 0.0, "y": 0.0, "rx": 100.0, "ry": -100.0},
        {"id": 1, "t": 10.0, "x": 5000.0, "y": 0.0, "rx": 0.0, "ry": 0.0},
    ]


class SelectorRuleTest(unittest.TestCase):
    def test_zero_amplitude_ties_and_keeps_the_latest_sample(self):
        picked = select_sample(_samples(), set(), [{"x": 0.0, "y": 0.0, "t": 30.0, "id": -1}], 10.0, KERNEL)
        self.assertEqual(picked["chosen"]["id"], 1)
        self.assertEqual(picked["latest"]["id"], 1)
        self.assertEqual(picked["available"], 2)

    def test_local_kernel_prefers_the_closer_sample_without_reading_the_measurement(self):
        kernel = {**KERNEL, "amplitude_m_s": 2.0}
        picked = select_sample(_samples(), set(), [{"x": 0.0, "y": 0.0, "t": 30.0, "id": -1}], 10.0, kernel)
        self.assertEqual(picked["chosen"]["id"], 0)

    def test_future_sample_is_rejected(self):
        samples = _samples() + [{"id": 2, "t": 40.0, "x": 0.0, "y": 0.0}]
        with self.assertRaises(ValueError):
            select_sample(samples, set(), [{"x": 0.0, "y": 0.0, "t": 30.0, "id": -1}], 10.0, KERNEL)

    def test_zero_amplitude_posterior_is_the_location_free_mean(self):
        observations = [
            {"id": 0, "t": 0.0, "x": 0.0, "y": 0.0, "rx": 1.0, "ry": -2.0},
            {"id": 1, "t": 30.0, "x": 1000.0, "y": -400.0, "rx": 3.0, "ry": 0.5},
        ]
        first = posterior_mean(observations, {"x": 10.0, "y": 10.0, "t": 60.0, "id": -1}, KERNEL)
        second = posterior_mean(observations, {"x": -800.0, "y": 2000.0, "t": 90.0, "id": -1}, KERNEL)
        self.assertAlmostEqual(first[0], second[0], places=6)
        self.assertAlmostEqual(first[1], second[1], places=6)
        self.assertAlmostEqual(first[0], global_mean([1.0, 3.0], 8.0, 0.5), places=6)
        self.assertAlmostEqual(first[1], global_mean([-2.0, 0.5], 8.0, 0.5), places=6)

    def test_duplicate_sample_is_not_a_second_observation(self):
        observations = [
            {"id": 0, "t": 0.0, "x": 0.0, "y": 0.0, "rx": 1.0, "ry": 1.0},
            {"id": 0, "t": 0.0, "x": 0.0, "y": 0.0, "rx": 5.0, "ry": 5.0},
        ]
        with self.assertRaises(ValueError):
            posterior_mean(observations, {"x": 0.0, "y": 0.0, "t": 30.0, "id": -1}, KERNEL)


class FrozenTrajectoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run()

    def test_frozen_path_equal_budget_and_control_tie(self):
        summary = self.summary
        self.assertTrue(summary["trajectory_match"])
        self.assertAlmostEqual(summary["on_time_bits"], 140000.0, places=6)
        self.assertAlmostEqual(summary["energy_j"], 111310.32858681791, places=4)
        self.assertGreater(summary["n_samples"], summary["n_decisions"])
        self.assertEqual(summary["send_count"]["latest"], summary["n_decisions"])
        self.assertEqual(summary["send_count"]["relevant"], summary["n_decisions"])
        self.assertEqual(summary["send_count"]["no_message"], 0)
        self.assertEqual(summary["payload_bytes"], 32)
        self.assertLess(summary["airtime_s"], 30.0)
        self.assertEqual(summary["future_choice_count"], 0)
        self.assertEqual(summary["choices"]["control"]["differed"], 0)
        self.assertGreater(summary["choices"]["control"]["possible"], 0)
        self.assertEqual(summary["choices"]["control"]["possible"], summary["choices"]["local"]["possible"])
        self.assertEqual(summary["single_sample_times_s"], [0.0])
        self.assertFalse(summary["default_changed"])
        self.assertEqual(summary["config_status"], "registered_before_run")
        for row in summary["scores"]:
            if row["field"] == "control":
                self.assertAlmostEqual(row["paired_relevant_minus_latest"], 0.0, places=9)
        config = json.loads((ROOT / "configs" / "experiments" / "wind_observation_w5.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256((ROOT / "configs" / "experiments" / "wind_observation_w5.json").read_bytes()).hexdigest()
        self.assertEqual(config["status"], "registered_before_run")
        self.assertEqual(summary["config_sha256"], digest)


if __name__ == "__main__":
    unittest.main()
