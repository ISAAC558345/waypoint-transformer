"""Fairness checks for the derived B2 view. They do not rerun the 18-row matrix."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from wt.ocean_observation_m0b.contract import EQUIVALENCE_LABEL
from wt.ocean_observation_m0b.derive import ROOT, build_view

SOURCE = ROOT / "results" / "ocean_observation_m0a"
WATCHED = (
    SOURCE / "comparison.json",
    SOURCE / "schedule.json",
    SOURCE / "geometry_manifest.json",
    ROOT / "docs" / "research" / "Ocean_Observation_M0A_Report_CN.md",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OceanObservationM0BTests(unittest.TestCase):
    def test_shared_rule_keeps_b2_equal_to_c_and_the_saved_bytes(self) -> None:
        package = ROOT / "src" / "wt" / "ocean_observation_m0b"
        for path in package.glob("*.py"):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                self.assertFalse(stripped.startswith("import torch") or stripped.startswith("from torch"))
        before = {path: _sha256(path) for path in WATCHED}
        view = build_view()
        self.assertEqual({path: _sha256(path) for path in WATCHED}, before)
        self.assertEqual(len(json.loads((SOURCE / "comparison.json").read_text(encoding="utf-8"))["rows"]), 18)
        self.assertEqual(view["environment_steps"], 0)
        self.assertEqual(len(view["b2_rows"]), 6)
        self.assertEqual(len(view["revised_c_rows"]), 2)
        self.assertLessEqual(view["derived_method_row_count"], 12)
        self.assertTrue(all(row["relation"] == EQUIVALENCE_LABEL and row["independent_experiment"] is False for row in view["b2_rows"]))
        self.assertEqual(view["scheduled_overlaps"], [])
        self.assertEqual(len(view["proposed_cross_uav_overlaps"]), 12)
        self.assertTrue(all(item["geometry"] == "separated_regions" and item["generation_s"] == 1200.0 and item["method"] == "moving" for item in view["proposed_cross_uav_overlaps"]))
        self.assertTrue(all(row["same_under_shared_rule"] for row in view["difference_table"]))
        self.assertTrue(all(row["delivered_count_difference"] == 0 for row in view["difference_table"]))
        self.assertTrue(all(row["mean_delay_difference_s"] == 0.0 for row in view["difference_table"]))
        self.assertTrue(all(all(value == 0.0 for value in row["node_energy_difference_J"].values()) for row in view["difference_table"]))

        separated = [row for row in view["identity"] if row["geometry"] == "separated_regions" and row["method"] == "moving"]
        self.assertEqual([row["recorded_uav"] for row in separated], [0, 0, 0, 0])
        self.assertEqual([row["assigned_uav"] for row in separated], [0, 1, 0, 1])
        self.assertTrue(all(row["assigned_uav"] == row["list_index"] for row in view["identity"]))
        active = {}
        for row in view["identity"]:
            if row["method"] != "moving" or not row["active"]:
                continue
            active.setdefault((row["geometry"], row["wind"]), []).append(row["assigned_uav"])
        self.assertEqual(active[("original_domain", "calm")], [0])
        self.assertEqual(active[("single_region_beyond_direct", "calm")], [0])
        self.assertEqual(active[("separated_regions", "calm")], [0, 1])
        self.assertEqual(active[("separated_regions", "wind_8mps_east")], [0, 1])
        self.assertTrue(view["fairness"]["points_match"])
        schedule = json.loads((SOURCE / "schedule.json").read_text(encoding="utf-8"))
        for context in schedule["contexts"]:
            for wind in context["winds"]:
                saved_uavs = wind["methods"]["moving"]["uavs"]
                rows = [
                    row
                    for row in view["identity"]
                    if row["geometry"] == context["geometry"] and row["wind"] == wind["wind"] and row["method"] == "moving"
                ]
                self.assertEqual([row["proposed_windows"] for row in rows], [uav["windows"] for uav in saved_uavs])
                self.assertEqual(len(wind["methods"]["moving"]["products"]), 8)
                self.assertEqual(len(wind["methods"]["fixed"]["products"]), 8)

        limit = view["energy_limit_J"]
        self.assertEqual(len(view["timing_corrections"]), 2)
        for correction in view["timing_corrections"]:
            self.assertEqual(correction["waited_uavs"], [1])
            self.assertGreater(correction["wait_s"]["1"], 0.0)
            self.assertEqual(correction["revised_uav_energy_J"][0], correction["original_uav_energy_J"][0])
            self.assertGreater(correction["revised_uav_energy_J"][1], correction["original_uav_energy_J"][1])
            self.assertLessEqual(correction["revised_uav_energy_J"][0], limit)
            self.assertLessEqual(correction["revised_uav_energy_J"][1], limit)
        wind = next(row for row in view["revised_c_rows"] if row["wind"] == "wind_8mps_east")
        self.assertGreater(wind["system_energy_J"], limit)
        self.assertTrue(wind["acceptable"])
        self.assertTrue(all(flag for flag in wind["uav_within_reserve"]))


if __name__ == "__main__":
    unittest.main()
