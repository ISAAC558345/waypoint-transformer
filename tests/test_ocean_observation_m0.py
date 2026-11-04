"""Checks for the ocean-observation M0 accounting."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.ocean_observation_m0.accounting import (  # noqa: E402
    build_cases,
    case_catalog,
    conclude,
    evaluate_direct,
)
from wt.ocean_observation_m0.codec import (  # noqa: E402
    encode_product,
    packet_sizes,
    parse_frame,
)
from wt.ocean_observation_m0.flight import (  # noqa: E402
    energy_limit_j,
    ground_speed_m_s,
    hover_power_w,
    induced_power_w,
    leg_budget,
    profile_power_w,
    propulsion_power_w,
    select_cruise_airspeed_m_s,
)
from wt.ocean_observation_m0.information import (  # noqa: E402
    ShoreBelief,
    advance,
    command_outcome,
    worked_product,
)
from wt.ocean_observation_m0.parameters import (  # noqa: E402
    ADOPTED,
    AIRSPEED_GRID_M_S,
    PERFORMANCE_PASS,
    RAW_RECORD_BITS,
    RELEASE_ELIGIBLE,
    assert_registered_layouts,
    parameter_records,
)
from wt.ocean_observation_m0.radio import (  # noqa: E402
    dbm_to_w,
    evaluate_link,
    j_to_wh,
    lora_time_on_air_s,
    tx_power_w,
)


class OceanObservationM0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = build_cases()
        cls.sizes = packet_sizes()
        cls.cruise = select_cruise_airspeed_m_s()

    def test_units_and_packet_dimensions(self) -> None:
        self.assertAlmostEqual(dbm_to_w(0.0), 0.001)
        self.assertAlmostEqual(dbm_to_w(30.0), 1.0)
        self.assertAlmostEqual(j_to_wh(3600.0), 1.0)
        self.assertEqual(RAW_RECORD_BITS, 3 * 4 * 16 * 1200)
        self.assertEqual(RAW_RECORD_BITS, 230400)
        product = self.sizes["product"]
        self.assertEqual(product["framed_bytes"], product["application_bytes"] + 8)
        self.assertEqual(product["framed_bits"], product["framed_bytes"] * 8)
        self.assertLess(product["framed_bits"], 10**4)
        self.assertNotEqual(product["framed_bits"], 10**6)
        self.assertEqual(product["framed_bytes"], 52)

    def test_reference_airtime_and_single_threshold(self) -> None:
        # SF7, 125 kHz, CR 4/6, 8 preamble symbols, CRC on, 10-byte payload.
        self.assertAlmostEqual(lora_time_on_air_s(10), 0.045312, places=9)
        link = evaluate_link(1000.0, 1.0, 10.0, 2.0, 6.0, 52)
        self.assertIsNone(link["packet_error_model"])
        self.assertAlmostEqual(link["rate_bps"], link["payload_bits"] / link["airtime_s"])
        self.assertAlmostEqual(link["tx_energy_J"], tx_power_w() * link["airtime_s"])

    def test_cruise_is_the_zero_wind_minimum_and_hover_matches_parts(self) -> None:
        costs = [propulsion_power_w(speed) / speed for speed in AIRSPEED_GRID_M_S]
        self.assertEqual(self.cruise, AIRSPEED_GRID_M_S[costs.index(min(costs))])
        self.assertGreater(self.cruise, AIRSPEED_GRID_M_S[0])
        self.assertLess(self.cruise, AIRSPEED_GRID_M_S[-1])
        self.assertAlmostEqual(hover_power_w(), profile_power_w() + induced_power_w())
        self.assertAlmostEqual(propulsion_power_w(0.0), hover_power_w())

    def test_headwind_costs_more_than_tailwind_and_round_trip_is_not_double_headwind(self) -> None:
        calm = leg_budget((0.0, 0.0), (1000.0, 0.0), (0.0, 0.0), self.cruise)
        tail = leg_budget((0.0, 0.0), (1000.0, 0.0), (8.0, 0.0), self.cruise)
        head = leg_budget((1000.0, 0.0), (0.0, 0.0), (8.0, 0.0), self.cruise)
        self.assertAlmostEqual(calm["ground_speed_m_s"], self.cruise)
        self.assertGreater(head["time_s"], tail["time_s"])
        self.assertGreater(head["propulsion_J"], tail["propulsion_J"])
        self.assertGreater(head["propulsion_J"], 0.0)
        self.assertLess(tail["propulsion_J"] + head["propulsion_J"], 2.0 * head["propulsion_J"])
        expected = propulsion_power_w(self.cruise) * head["cruise_s"] + hover_power_w() * head["transient_s"]
        self.assertAlmostEqual(head["propulsion_J"], expected)

    def test_crosswind_above_airspeed_has_no_ground_speed_and_no_invented_energy(self) -> None:
        self.assertIsNone(ground_speed_m_s(5.0, (0.0, 8.0), (1.0, 0.0)))
        blocked = leg_budget((0.0, 0.0), (1000.0, 0.0), (0.0, 8.0), 5.0)
        self.assertFalse(blocked["feasible"])
        self.assertIsNone(blocked["propulsion_J"])

    def test_long_leg_exceeds_the_registered_reserve(self) -> None:
        long_leg = leg_budget((0.0, 0.0), (100000.0, 0.0), (0.0, 0.0), self.cruise)
        spent = long_leg["propulsion_J"] + long_leg["compute_J"]
        self.assertGreater(spent, energy_limit_j())
        self.assertGreater(long_leg["propulsion_J"], 0.0)

    def test_full_packet_duplicate_stale_and_unacked_command(self) -> None:
        blob = encode_product(3, 9, -1200.0, 0.0, 10.0, 20.0, 1.4, 0.01, 0, 21)
        belief = ShoreBelief()
        self.assertEqual(belief.mean_m, 1.2)
        fused = belief.receive(blob, 10.0)
        self.assertEqual(fused["status"], "fused")
        self.assertEqual(fused["window_end_s"], 0.0)
        self.assertFalse(fused["sampling_time_rewritten"])
        self.assertNotEqual(belief.mean_m, 1.2)
        duplicate = belief.receive(blob, 11.0)
        self.assertEqual(duplicate["status"], "duplicate_no_fusion")
        self.assertEqual(belief.information_updates, 1)
        stale = ShoreBelief()
        stale_event = stale.receive(blob, 3601.0)
        self.assertEqual(stale_event["status"], "stale_recorded_not_fused")
        self.assertEqual(stale.mean_m, 1.2)
        self.assertEqual(stale.information_updates, 0)
        partial = ShoreBelief().receive(blob, 1.0, complete=False)
        self.assertEqual(partial["status"], "partial_no_fusion")
        broken = bytearray(blob)
        broken[10] ^= 0xFF
        self.assertFalse(parse_frame(bytes(broken))["ok"])
        self.assertEqual(ShoreBelief().receive(bytes(broken), 1.0)["map_updated"], False)
        unacked = command_outcome(True, False, 0.0, 1800.0)
        self.assertEqual(unacked["uav"], "activated")
        self.assertEqual(unacked["shore"], "pending_not_assumed_unexecuted")
        self.assertFalse(unacked["assumed_unexecuted"])
        lost = command_outcome(False, False, 0.0, 1800.0)
        self.assertEqual(lost["uav"], "keeps_last_activated_task")
        self.assertFalse(lost["shore_knows_execution"])

    def test_drift_is_not_reflected_and_example_is_hidden_before_receipt(self) -> None:
        ended = advance((1990.0, 1000.0), (8.0, 0.0), 1800.0)
        self.assertGreater(ended[0], 2000.0)
        example = worked_product((500.0, 500.0), (0.0, 0.0))
        self.assertEqual(example["n_samples"], 21)
        self.assertAlmostEqual(example["measurement_variance_m2"], 0.05**2 / 21)
        self.assertFalse(example["readable_before_receive"])
        self.assertEqual(example["belief_before"]["mean_m"], 1.2)
        self.assertEqual(example["fusion"]["status"], "fused")
        self.assertEqual(example["window_end_s"], 0.0)

    def test_registered_matrix_and_direct_margin(self) -> None:
        assert_registered_layouts()
        catalog = case_catalog()
        self.assertEqual(len(catalog), 24)
        self.assertEqual(len({row["case_id"] for row in catalog}), 24)
        self.assertEqual(len(self.cases), 24)
        regimes = {row["regime"] for row in catalog}
        self.assertEqual(len(regimes), 3)
        self.assertTrue(all(case["analytical_case"] and not case["environment_episode"] for case in self.cases))
        self.assertTrue(all(case["direct"]["delivery_feasible"] for case in self.cases))
        self.assertGreater(min(case["direct"]["min_margin_db"] for case in self.cases), 20.0)
        radios = {case["direct"]["radio"]["tx_dbm"] for case in self.cases}
        radios.update(case["farthest"]["radio"]["tx_dbm"] for case in self.cases)
        self.assertEqual(radios, {13.0})
        self.assertTrue(all(case["direct"]["radio"]["packet_error_model"] is None for case in self.cases))

    def test_infeasible_results_stay_in_the_matrix(self) -> None:
        abundant = [case for case in self.cases if case["regime"] == "communication_abundant"]
        self.assertTrue(all(case["farthest"]["delivery_feasible"] and case["tour"]["delivery_feasible"] for case in abundant))
        intermittent = [case for case in self.cases if case["regime"] == "intermittent_backhaul"]
        self.assertEqual(len(intermittent), 8)
        self.assertTrue(all("timescale" in case["farthest"]["reasons"] for case in intermittent))
        self.assertTrue(all(not case["farthest"]["delivery_feasible"] for case in intermittent))
        reserve = [case for case in self.cases if "energy_reserve" in case["tour"]["reasons"]]
        self.assertEqual(
            [case["case_id"] for case in reserve],
            ["u6_b12_joint_access_and_communication_constraint_wind_8mps_east_dispersed"],
        )
        self.assertTrue(reserve[0]["farthest"]["delivery_feasible"])
        spent = max(vehicle["uav_J"] for vehicle in reserve[0]["tour"]["vehicles"]) * 2.0
        self.assertGreater(spent, energy_limit_j())
        self.assertGreater(spent, 0.0)
        self.assertEqual(sum(case["drift"]["exited"] for case in self.cases), 0)
        self.assertTrue(all(case["drift"]["reflected"] is False for case in self.cases))

    def test_cost_buckets_partition_the_total_and_flight_dominates_the_ferry(self) -> None:
        sample = next(
            case
            for case in self.cases
            if case["regime"] == "communication_abundant"
            and case["wind_name"] == "calm"
            and case["layout"] == "dispersed"
            and case["n_uav"] == 2
        )
        episode = sample["farthest"]["episode"]
        keys = ("flight_J", "return_J", "wait_J", "handshake_J", "payload_radio_J", "telemetry_J")
        self.assertAlmostEqual(sum(episode[key] for key in keys), episode["total_J"])
        self.assertLessEqual(episode["uav_J"], episode["total_J"] + 1e-6)
        self.assertEqual(sample["farthest_dominant"], "flight_J")
        self.assertLess(sample["direct"]["episode"]["total_J"], 10.0)
        self.assertGreater(episode["uav_J"], 10**4)
        self.assertGreater(episode["flight_J"] + episode["return_J"], episode["wait_J"])
        self.assertGreater(episode["wait_J"], episode["payload_radio_J"])

    def test_conclusion_rule_and_release_flags(self) -> None:
        self.assertFalse(PERFORMANCE_PASS)
        self.assertFalse(ADOPTED)
        self.assertFalse(RELEASE_ELIGIBLE)
        records = parameter_records(
            {
                "P0_W": 1.0,
                "Pi_W": 1.0,
                "cruise_airspeed_m_s": self.cruise,
                "usable_energy_J": 1.0,
            }
        )
        self.assertTrue(all(row["missing"] is False for row in records))
        self.assertEqual(conclude(self.cases, records)["conclusion"], "REFRAME_REQUIRED")
        forced = [dict(case) for case in self.cases]
        for case in forced:
            case["direct"] = {"delivery_feasible": False}
            case["farthest"] = {"delivery_feasible": True}
        self.assertEqual(conclude(forced, records)["conclusion"], "DESIGN_READY")

    def test_package_does_not_load_a_policy(self) -> None:
        package = ROOT / "src" / "wt" / "ocean_observation_m0"
        text = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
        self.assertNotIn("import torch", text)
        self.assertNotIn("optimizer.step", text)

    def test_direct_session_uses_the_same_packet_airtime(self) -> None:
        buoys = [tuple(point) for point in self.cases[0]["buoys_xy_m"]]
        direct = evaluate_direct(buoys, "communication_abundant", self.sizes)
        self.assertTrue(direct["delivery_feasible"])
        self.assertTrue(math.isfinite(direct["episode"]["total_J"]))


if __name__ == "__main__":
    unittest.main()
