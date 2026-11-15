"""Model weights have to use each arrived sample once and keep the fixed-model archive.

fixed model archive: the global-only or local-only forecast leaves the W5.1 early errors.

message once: an assimilated sample is still accepted as a new observation.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.run_w51 import message_is_usable  # noqa: E402
from wt.active_model_identification.run_w6 import _predict, run  # noqa: E402


class MessageOnceTest(unittest.TestCase):
    def test_a_message_is_not_usable_at_its_send_decision(self):
        self.assertFalse(message_is_usable(0.0, 0.0, 0.082176))
        self.assertTrue(message_is_usable(0.0, 30.0, 0.082176))

    def test_an_assimilated_sample_cannot_predict_itself(self):
        kernel = {"prior_std_m_s": 8.0, "sigma_m_s": 0.5, "amplitude_m_s": 0.0, "length_m": 600.0, "time_s": 90.0}
        sample = {"id": 3, "x": 10.0, "y": 20.0, "t": 30.0, "rx": 6.0, "ry": 0.1}
        with self.assertRaises(RuntimeError):
            _predict([sample], sample, kernel)


class ModelWeightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run()

    def test_fixed_models_match_the_archive_and_each_message_is_used_once(self):
        summary = self.summary
        self.assertTrue(summary["archive_match"])
        self.assertTrue(summary["message_once"])
        self.assertFalse(summary["default_changed"])
        self.assertEqual(summary["n_messages"], 30)
        self.assertEqual(summary["payload_bytes"], 32)
        self.assertEqual(summary["n_flight_decisions"], 8)
        self.assertEqual(summary["config_status"], "registered_before_run")
        config_path = ROOT / "configs" / "experiments" / "wind_observation_w6.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "registered_before_run")
        self.assertEqual(summary["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())
        self.assertEqual(config["prior_weight_global"], 0.5)
        self.assertEqual(config["prior_weight_local"], 0.5)
        for snap in summary["snapshots"]:
            self.assertLessEqual(snap["last_flight"]["n_received"], snap["mission_end"]["n_received"])
            self.assertAlmostEqual(snap["last_flight"]["weight_global"] + snap["last_flight"]["weight_local"], 1.0, places=9)
            self.assertAlmostEqual(snap["mission_end"]["weight_global"] + snap["mission_end"]["weight_local"], 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
