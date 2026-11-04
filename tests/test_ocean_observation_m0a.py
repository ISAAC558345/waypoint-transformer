"""M0A checks the horizon, the existing packet, and the saved same-task table."""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from wt.ocean_observation_m0.codec import packet_sizes
from wt.ocean_observation_m0.radio import lora_time_on_air_s, radio_horizon_m
from wt.ocean_observation_m0a.compare import CONCLUSIONS
from wt.ocean_observation_m0a.horizon import geometric_horizon_m

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "results" / "ocean_observation_m0a" / "comparison.json"


class OceanObservationM0ATests(unittest.TestCase):
    def test_hand_horizon_for_one_and_ten_metres(self) -> None:
        # sqrt(2*6371000*1) + sqrt(2*6371000*10) and the same product with k=4/3.
        self.assertAlmostEqual(geometric_horizon_m(1.0, 10.0, k=1.0), 14857.640589667357, places=6)
        self.assertAlmostEqual(geometric_horizon_m(1.0, 10.0, k=4.0 / 3.0), 17156.125587934315, places=6)
        historical = 3570.0 * (math.sqrt(1.0) + math.sqrt(10.0))
        self.assertAlmostEqual(radio_horizon_m(1.0, 10.0), historical, places=6)

    def test_product_length_and_airtime_stay_on_the_existing_codec(self) -> None:
        sizes = packet_sizes()
        self.assertEqual(sizes["product"]["framed_bytes"], 52)
        self.assertEqual(sizes["product"]["application_bytes"], 44)
        airtime = lora_time_on_air_s(sizes["product"]["framed_bytes"])
        self.assertAlmostEqual(airtime, 0.11904, places=5)

    def test_saved_comparison_is_the_same_eight_products(self) -> None:
        self.assertTrue(COMPARISON.is_file(), "the analytical comparison is missing")
        geometry = json.loads((ROOT / "results" / "ocean_observation_m0a" / "geometry_manifest.json").read_text(encoding="utf-8"))
        for context in geometry["geometries"]:
            self.assertEqual(len(context["buoys_xy_m"]), 4)
        table = json.loads(COMPARISON.read_text(encoding="utf-8"))
        self.assertEqual(len(table["rows"]), 18)
        self.assertIn(table["conclusion"], CONCLUSIONS)
        seen = {(row["geometry"], row["wind"], row["method"]) for row in table["rows"]}
        self.assertEqual(len(seen), 18)
        for row in table["rows"]:
            self.assertEqual(row["product_count"], 8)
            self.assertGreaterEqual(row["undelivered_count"], 0)
            self.assertLessEqual(row["undelivered_count"], 8)
            self.assertIn("uav_0", row["nodes_J"])
            self.assertIn("uav_1", row["nodes_J"])
            self.assertIn("shore", row["nodes_J"])


if __name__ == "__main__":
    unittest.main()
