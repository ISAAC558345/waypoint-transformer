"""Necessary checks for the M1.2 completion quote, quadrature, and budget window."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.belief import (  # noqa: E402
    Belief,
    equal_width_quadrature,
    expectation,
    gauss_legendre_unit,
    prior_belief,
)
from wt.active_model_identification.completion import accounting_cases, admits, completion_quote  # noqa: E402
from wt.active_model_identification.integral import select_integral  # noqa: E402
from wt.active_model_identification.policies import candidates, select_lookahead, terminal_value  # noqa: E402
from wt.active_model_identification.scenarios import ablate_beliefs, assess_m12  # noqa: E402
from wt.active_model_identification.spec import load_m12_spec  # noqa: E402
from wt.active_model_identification.world import advance, make_world  # noqa: E402


def _net(quote, spec: dict) -> float:
    block = spec["utility"]
    return quote.on_time_bits / float(block["b_ref_bits"]) - float(block["lambda_e"]) * quote.energy_j / float(block["e_ref_j"])


class QuadratureTests(unittest.TestCase):
    def test_gauss_legendre_matches_the_first_polynomials(self) -> None:
        nodes = gauss_legendre_unit(3)
        self.assertEqual(len(nodes), 3)
        self.assertAlmostEqual(sum(weight for _u, weight in nodes), 1.0)
        self.assertAlmostEqual(sum(weight * probability for probability, weight in nodes), 0.5)
        self.assertAlmostEqual(sum(weight * probability * probability for probability, weight in nodes), 1.0 / 3.0)

    def test_registered_moment_tolerance_holds_for_both_candidates(self) -> None:
        spec = load_m12_spec()
        rule = spec["integral"]
        mean_tol = float(rule["theta_mean_abs_tol"])
        variance_tol = float(rule["theta_variance_abs_tol"])
        for nodes in rule["theta_candidates"]:
            for case in rule["moment_cases"]:
                belief = Belief(float(case["mu"]), 1.0 / float(case["std"]) ** 2, 0.8, 1.4)
                mean, variance = belief.moments()
                mean_error = abs(expectation(belief, int(nodes), lambda theta: theta) - mean)
                variance_error = abs(expectation(belief, int(nodes), lambda theta, mean=mean: (theta - mean) ** 2) - variance)
                self.assertLess(mean_error, mean_tol)
                self.assertLess(variance_error, variance_tol)
        narrow = Belief(1.25, 1.0 / (0.02 ** 2), 0.8, 1.4)
        width = sum(weight * theta for theta, weight in equal_width_quadrature(narrow, 7))
        self.assertGreater(abs(width - narrow.mean()), 0.01)

    def test_failed_candidates_are_not_met(self) -> None:
        spec = copy.deepcopy(load_m12_spec())
        spec["integral"]["theta_variance_abs_tol"] = -1.0
        selected = select_integral(spec)
        self.assertEqual(selected["numerical_status"], "NOT_MET")
        self.assertIsNone(selected["theta_nodes"])
        self.assertIsNone(selected["noise_nodes"])
        self.assertTrue(all(not row["passed"] for row in selected["moment_checks"]))


class AccountingTests(unittest.TestCase):
    def test_quote_matches_execution_and_is_affine_in_theta(self) -> None:
        spec = load_m12_spec()
        rows = accounting_cases(spec)
        tol = float(spec["energy"]["execution_abs_tol_j"])
        by_name: dict[str, list[dict]] = {}
        for row in rows:
            self.assertLess(abs(row["group_errors_j"]["total"]), tol)
            self.assertEqual(row["interventions"], 0)
            for value in row["group_errors_j"].values():
                self.assertLess(abs(value), tol)
            by_name.setdefault(row["name"], []).append(row)
        pad = by_name["pad_far"]
        self.assertTrue(all(row["on_time"] for row in pad))
        self.assertTrue(all(not row["on_time"] for row in by_name["near_deadline_cache"]))
        left, right = pad
        self.assertAlmostEqual(left["affine_z_j"], right["affine_z_j"], places=6)
        self.assertAlmostEqual(left["quote_j"], left["theta"] * left["affine_z_j"] + left["affine_aux_j"], places=4)
        self.assertAlmostEqual(right["quote_j"], right["theta"] * right["affine_z_j"] + right["affine_aux_j"], places=4)

    def test_original_battery_stays_put_when_the_posterior_quantile_does_not_fit(self) -> None:
        spec = load_m12_spec()
        assessed = assess_m12(spec)
        candidate = float(spec["energy"]["candidate_sensitive_battery_j"])
        self.assertEqual(assessed["candidate_battery_j"], candidate)
        self.assertEqual(assessed["sensitive_battery_j"], [candidate, candidate])
        self.assertFalse(assessed["battery_replaced"])
        self.assertTrue(assessed["ideal_script_window"])
        self.assertFalse(assessed["posterior_zero_admits"])
        self.assertFalse(assessed["window_valid"])
        self.assertEqual(assessed["scenario_status"], "NOT_A_WINDOW")
        self.assertTrue(assessed["near_admitted_under_prior"])
        for row in assessed["uavs"]:
            self.assertTrue(row["physical_at_true_theta_1"])
            self.assertFalse(row["admitted_under_prior"])
            self.assertFalse(row["information_changes_admission"])
            self.assertGreater(row["prior_quote_j"] - row["physical_quote_j"], 1000.0)
            self.assertLess(row["zero_noise_posterior"]["slack_j"], 0.0)

    def test_airborne_wait_must_keep_a_return_margin(self) -> None:
        spec = load_m12_spec()
        remote = make_world(spec, [1.0e7, 1.0e7], "remote-wait")
        remote.uavs[0].x = 800.0
        remote.uavs[0].airborne = True
        remote.uavs[0].energy_j = 100.0
        beliefs = [prior_belief(spec), prior_belief(spec)]
        observations = advance(
            remote,
            [("wait", None), ("wait", None)],
            [1.0, 1.0],
            beliefs,
            spec,
            1.0,
            None,
            planning=True,
            interval_index=0,
            noise_j=lambda *_: 0.0,
        )
        self.assertEqual(observations[0]["executed"][0], "return")
        self.assertTrue(observations[0]["intervened"])
        home = make_world(spec, [1.0e7, 1.0e7], "landed-wait")
        home.uavs[0].energy_j = 100.0
        home.uavs[0].airborne = False
        observations = advance(
            home,
            [("wait", None), ("wait", None)],
            [1.0, 1.0],
            [prior_belief(spec), prior_belief(spec)],
            spec,
            1.0,
            None,
            planning=True,
            interval_index=0,
            noise_j=lambda *_: 0.0,
        )
        self.assertEqual(observations[0]["executed"][0], "wait")
        self.assertFalse(home.uavs[0].airborne)
        self.assertEqual(home.uavs[0].z_j, 0.0)

    def test_terminal_picks_one_business_and_charges_its_cost(self) -> None:
        spec = load_m12_spec()
        world = make_world(spec, [1.0e7, 1.0e7], "terminal")
        packet = next(item for item in world.packets if item.buoy == 1)
        packet.holder = "uav:0"
        world.uavs[0].cache_ids = [packet.id]
        beliefs = [prior_belief(spec), prior_belief(spec)]
        theta = beliefs[0].quantile(float(spec["belief"]["guard_probability"]))
        nets = []
        cached = completion_quote(world, 0, None, theta, spec)
        if admits(world.uavs[0], cached, spec):
            nets.append(_net(cached, spec))
        for buoy in spec["geometry"]["buoys"]:
            buoy_id = int(buoy["id"])
            if not any(packet.buoy == buoy_id and packet.holder == "buoy" for packet in world.packets):
                continue
            quote = completion_quote(world, 0, buoy_id, theta, spec)
            if admits(world.uavs[0], quote, spec):
                nets.append(_net(quote, spec))
        value = terminal_value(world, beliefs, spec, 1.0, 0)
        self.assertGreater(len([item for item in nets if item > 0.0]), 1)
        self.assertAlmostEqual(value, max(nets), places=6)
        self.assertLess(value, sum(item for item in nets if item > 0.0) - 0.1)
        world.t = float(spec["time"]["mission_s"]) - 0.1
        self.assertEqual(terminal_value(world, beliefs, spec, 1.0, 0), 0.0)


class PlannerTests(unittest.TestCase):
    def test_calibration_switch_and_ablation_keep_the_other_update(self) -> None:
        spec = copy.deepcopy(load_m12_spec())
        spec["integral"]["noise_nodes"] = 1
        spec["belief"]["quadrature_nodes"] = 7
        world = make_world(spec, [1.0e7, 1.0e7], "switch")
        beliefs = [prior_belief(spec), prior_belief(spec)]
        self.assertTrue(any(action[0] == "calibrate" for action in candidates(spec, world, 0, include_calibrate=True)))
        off, off_tables = select_lookahead("lookahead_calibration_off", world, beliefs, spec, 1.0, None)
        on_tables = select_lookahead("lookahead_calibration_on", world, [item.copy() for item in beliefs], spec, 1.0, None)[1]
        self.assertTrue(all(action[0] != "calibrate" for action in off))
        self.assertTrue(all(item[1][0] != "calibrate" for table in off_tables for item in table))
        self.assertTrue(any(item[1][0] == "calibrate" for table in on_tables for item in table))
        before = prior_belief(spec)
        updated = before.update(4000.0, 4200.0, 180.0)
        other = before.update(1000.0, 900.0, 180.0)
        kept, removed = ablate_beliefs([before, before], [updated, other], [("calibrate", None), ("service", 0)])
        self.assertEqual(removed, [0])
        self.assertEqual(kept[0].mu, before.mu)
        self.assertEqual(kept[1].mu, other.mu)


if __name__ == "__main__":
    unittest.main()
