"""Necessary checks for the M1.4 attribution. They do not rerun the M1.3 pilot."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.attribution import (  # noqa: E402
    attribute,
    choose_alternative,
    classify_requests,
    duration_of,
    identity_terms,
    judge_sign,
    needs_continuation,
    pair_rows,
    split_action,
)
from wt.active_model_identification.belief import prior_belief  # noqa: E402
from wt.active_model_identification.spec import load_m13_spec  # noqa: E402
from wt.active_model_identification.world import advance, make_world  # noqa: E402


def _episode(scenario_id: str, condition: str) -> dict:
    return {
        "base": scenario_id.split("_", 1)[0],
        "condition": condition,
        "scenario_id": scenario_id,
        "method": "lookahead_calibration_on",
        "branch": "core",
        "utility": 1.0,
        "on_time_bits": 0.0,
        "energy_j": 0.0,
        "calibration_s": 0.0,
        "returned": 0,
        "not_returned": 0,
        "not_landed": 0,
        "below_reserve": 0,
        "energy_exhausted": 0,
        "return_failures": 0,
    }


def _request(scenario_id: str, epoch: int, uav: int, table: str, executed: str, intervened: str) -> dict:
    return {
        "branch": "core",
        "method": "lookahead_calibration_on",
        "scenario_id": scenario_id,
        "epoch": str(epoch),
        "uav": str(uav),
        "table_choice": table,
        "requested": "calibrate:None",
        "executed": executed,
        "intervened": intervened,
        "calibrate_q": "-0.02",
        "best_execute_q": "-0.021",
        "q_gap_calibrate_minus_execute": "0.001",
    }


class AttributionTests(unittest.TestCase):
    def test_identity_duration_and_pair_keys(self) -> None:
        terms = identity_terms(1.0, 0.4, 0.7, 0.2)
        self.assertAlmostEqual(terms["residual"], 0.0, places=12)
        self.assertAlmostEqual(terms["state_cost"] + terms["info_premium_c"] - terms["info_premium_e"], terms["gap_plus"], places=12)
        self.assertEqual(duration_of("calibrate:None"), (30.0, "final_executed_action_is_calibrate"))
        self.assertIsNone(duration_of("return:None")[0])
        labeled = classify_requests(
            [
                _request("b1_tight", 1, 0, "service:0", "calibrate:None", "False"),
                _request("b1_tight", 1, 1, "service:0", "service:0", "False"),
                _request("b1_window", 3, 0, "calibrate:None", "return:None", "True"),
            ]
        )
        by_uav = {(row["scenario_id"], row["uav"]): row for row in labeled}
        self.assertEqual(by_uav[("b1_tight", 0)]["origin"], "conflict_repair")
        self.assertEqual(by_uav[("b1_tight", 0)]["conflict_partner"], 1)
        self.assertEqual(by_uav[("b1_tight", 0)]["duration_s"], 30.0)
        self.assertFalse(by_uav[("b1_tight", 0)]["intervention"])
        self.assertEqual(by_uav[("b1_window", 0)]["origin"], "single_score")
        self.assertTrue(by_uav[("b1_window", 0)]["intervention"])
        self.assertIsNone(by_uav[("b1_window", 0)]["duration_s"])
        rows = pair_rows([_episode("b1_tight", "tight"), _episode("b1_window", "window")])
        self.assertEqual({row["scenario_id"] for row in rows}, {"b1_tight", "b1_window"})
        self.assertEqual({row["condition"] for row in rows}, {"tight", "window"})
        self.assertTrue(all(row["method"] and row["branch"] for row in rows))

    def test_sign_and_continuation_gate(self) -> None:
        tie = 0.0001
        residual = 0.000290380546953628
        self.assertEqual(judge_sign(0.00005, tie, residual, 0.001), "unresolved")
        self.assertEqual(judge_sign(0.01, tie, residual, None), "resolved")
        self.assertEqual(judge_sign(0.0002, tie, residual, 0.0003), "resolved")
        self.assertEqual(judge_sign(0.0002, tie, residual, -0.0003), "unresolved")
        terms = identity_terms(0.2, 0.1, 0.0, -0.1)
        self.assertEqual(attribute(terms, 0.0, tie, "resolved"), "immediate_motion_cost")
        self.assertFalse(needs_continuation(["immediate_motion_cost", "sign_unresolved"]))
        self.assertTrue(needs_continuation(["mixed"]))
        self.assertTrue(needs_continuation(["state_cost_without_update"]))

    def test_legal_alternative_skips_the_contested_buoy_and_calibrate(self) -> None:
        spec = load_m13_spec()
        world = make_world(spec, [1.0e7, 1.0e7], "m14-legal")
        belief = prior_belief(spec)
        table = [
            (5.0, ("service", 0)),
            (4.0, ("calibrate", None)),
            (3.0, ("service", 1)),
            (1.0, ("wait", None)),
            (0.5, ("return", None)),
        ]
        _score, action, notes = choose_alternative(table, ("service", 0), world, belief, spec, 1.0, 0)
        self.assertEqual(action, ("service", 1))
        reasons = {item["action"]: item["reason"] for item in notes}
        self.assertEqual(reasons["service:0"], "conflict_same_buoy")
        self.assertEqual(reasons["calibrate:None"], "calibrate_not_an_alternative")
        world.uavs[0].energy_j = 100.0
        _score, action, notes = choose_alternative(table, ("service", 0), world, belief, spec, 1.0, 0)
        self.assertEqual(action, ("wait", None))
        rejected = {item["action"] for item in notes if item["reason"] == "admission_rejected"}
        self.assertIn("service:1", rejected)

    def test_removing_the_focus_update_does_not_change_interval_cost(self) -> None:
        spec = load_m13_spec()
        world = make_world(spec, [1.0e7, 1.0e7], "m14-remove")
        for uav in world.uavs:
            uav.airborne = True
        beliefs = [prior_belief(spec), prior_belief(spec)]
        actions = [("calibrate", None), ("wait", None)]

        def run(remove: bool):
            snap = world.snapshot()
            copied = [item.copy() for item in beliefs]
            pre_focus = copied[0].copy()
            advance(
                snap,
                actions,
                [1.0, 1.1],
                copied,
                spec,
                1.0,
                None,
                planning=True,
                interval_index=0,
                noise_j=lambda index, _interval: 10.0 if index == 0 else -5.0,
            )
            focus = snap.uavs[0]
            partner = snap.uavs[1]
            cost = (
                focus.x,
                focus.y,
                focus.z_j,
                focus.aux_j,
                focus.consumed_j,
                partner.z_j,
                partner.consumed_j,
                snap.t,
            )
            if remove:
                copied[0] = pre_focus
            return cost, copied[0].mean(), copied[1].mean(), pre_focus.mean(), focus.aux_j, getattr(snap, "interval_uav_radio_j")[0]

        kept = run(False)
        removed = run(True)
        self.assertEqual(kept[0], removed[0])
        self.assertNotAlmostEqual(kept[1], kept[3], places=6)
        self.assertAlmostEqual(removed[1], removed[3], places=12)
        self.assertAlmostEqual(kept[2], removed[2], places=12)
        self.assertGreater(kept[0][2], 100.0)
        self.assertGreater(kept[5], 0.0)
        self.assertLessEqual(kept[5], kept[4] + 1e-9)
        self.assertAlmostEqual(kept[0][4], 1.0 * kept[0][2] + kept[0][3], places=6)

    def test_ordinary_execution_keeps_its_own_information_term(self) -> None:
        spec = copy.deepcopy(load_m13_spec())
        spec["belief"]["quadrature_nodes"] = 7
        spec["integral"]["segment_inner_nodes"] = 7
        world = make_world(spec, [1.0e7, 1.0e7], "m14-split")
        world.uavs[0].airborne = True
        beliefs = [prior_belief(spec), prior_belief(spec)]
        stats = {"rollouts": 0, "cache_hits": 0, "algebraic_integrals": 0, "algebraic_seconds": 0.0, "root_solves": 0}
        wait = split_action(world, beliefs, spec, 1.0, 0, ("wait", None), stats)
        calibrate = split_action(world, beliefs, spec, 1.0, 0, ("calibrate", None), stats)
        for item in (wait, calibrate):
            self.assertTrue(item["detail"]["telemetry"])
            self.assertGreater(item["detail"]["z_j"], 100.0)
            self.assertNotAlmostEqual(item["detail"]["prior_q95"], item["detail"]["posterior_q95_eps0"], places=4)
            self.assertAlmostEqual(item["q_plus"] - item["r30"], item["terminal_plus"], places=9)
            self.assertAlmostEqual(item["q_minus"] - item["r30"], item["terminal_minus"], places=9)
            self.assertAlmostEqual(item["info_premium"], item["terminal_plus"] - item["terminal_minus"], places=9)
            terms = identity_terms(item["q_plus"], item["q_minus"], item["q_plus"], item["q_minus"])
            self.assertAlmostEqual(terms["residual"], 0.0, places=9)
        crossed = identity_terms(calibrate["q_plus"], calibrate["q_minus"], wait["q_plus"], wait["q_minus"])
        self.assertAlmostEqual(crossed["residual"], 0.0, places=9)
        self.assertIn("info_premium", wait)
        self.assertIn("info_premium", calibrate)


if __name__ == "__main__":
    unittest.main()
