"""Necessary checks for the M1.5 return tail.

One airborne state and one landed state share the controller fixture. The
second test checks that a positive business is left unchanged and that the
negative tail enters the segmented score.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.belief import prior_belief  # noqa: E402
from wt.active_model_identification.completion import (  # noqa: E402
    TAIL_MODE,
    clear_return_tail_cache,
    completion_quote,
    return_tail_nominal,
    return_tail_value,
    terminal_mode,
)
from wt.active_model_identification.policies import score_actions, terminal_value  # noqa: E402
from wt.active_model_identification.segment import (  # noqa: E402
    clear_rollout_cache,
    extract_plans,
    reference_expectation,
    roll_action,
    segment_expectation,
    value_at,
)
from wt.active_model_identification.spec import load_m13_spec  # noqa: E402
from wt.active_model_identification.world import advance, make_world  # noqa: E402


def _segment_spec() -> dict:
    spec = copy.deepcopy(load_m13_spec())
    spec["integral"]["scoring"] = "segment"
    spec["integral"]["segment_inner_nodes"] = 7
    spec["belief"]["quadrature_nodes"] = 7
    return spec


def _boost(world):
    sim = world.snapshot()
    for uav in sim.uavs:
        uav.energy_j = 1.0e8
        uav.initial_energy_j = 1.0e8
        uav.failed = False
        uav.failure_recorded = False
        uav.pending_failure = False
        uav.z_j = 0.0
        uav.aux_j = 0.0
        uav.true_drop_j = 0.0
        uav.consumed_j = 0.0
    return sim


def _script(world, spec, theta: float, focus: int, first_action=None) -> dict:
    sim = _boost(world)
    beliefs = [prior_belief(spec), prior_belief(spec)]
    mission = float(spec["time"]["mission_s"])
    interval_z = []
    airborne_after = []
    epoch = 0
    while sim.t < mission - 1e-9:
        actions = [("wait", None), ("wait", None)]
        actions[focus] = first_action if epoch == 0 and first_action is not None else ("return", None)
        advance(
            sim,
            actions,
            [theta, theta],
            beliefs,
            spec,
            1.0,
            None,
            planning=True,
            interval_index=epoch,
            noise_j=lambda _index, _interval: 0.0,
        )
        interval_z.append(sim.uavs[focus].z_j)
        airborne_after.append(bool(sim.uavs[focus].airborne))
        epoch += 1
    focus_uav = sim.uavs[focus]
    return {
        "consumed_j": focus_uav.consumed_j,
        "z_j": sum(interval_z),
        "interval_z": interval_z,
        "airborne_after": airborne_after,
        "interventions": sim.interventions,
        "deliveries": len(sim.deliveries),
        "t": sim.t,
    }


def _airborne(world) -> None:
    uav = world.uavs[0]
    uav.x = 800.0
    uav.y = 400.0
    uav.vx = 18.0
    uav.vy = 0.0
    uav.airborne = True
    uav.land_timer_s = 0.0


def _hold_cache(world) -> None:
    uav = world.uavs[0]
    packet = world.packets[0]
    packet.holder = "uav:0"
    packet.remaining_bits = packet.bits
    if packet.id not in uav.cache_ids:
        uav.cache_ids.append(packet.id)


def _clear_business(world) -> None:
    for packet in world.packets:
        packet.holder = "shore"
        packet.remaining_bits = 0.0
        packet.delivered_s = 0.0
    for uav in world.uavs:
        uav.cache_ids.clear()


class ReturnTailTests(unittest.TestCase):
    def test_return_tail_matches_the_fixed_controller(self) -> None:
        spec = load_m13_spec()
        clear_return_tail_cache()
        mission = float(spec["time"]["mission_s"])
        base = make_world(spec, [1.0e8, 1.0e8], "m15-tail")

        landed = base.snapshot()
        landed_quote = return_tail_nominal(landed, 0, spec)
        landed_ref = _script(landed, spec, 1.0, 0)
        self.assertEqual(landed_ref["interventions"], 0)
        self.assertAlmostEqual(landed_quote.z_j + landed_quote.aux_j, landed_ref["consumed_j"], places=3)
        self.assertAlmostEqual(landed_quote.flight_z_j, 0.0, places=6)
        self.assertAlmostEqual(landed_quote.ground_z_j, 0.0, places=6)
        self.assertGreater(landed_quote.ground_aux_j, 0.0)
        self.assertGreater(landed_quote.radio_j, 0.0)
        self.assertLess(landed_quote.radio_j, landed_quote.aux_j)
        self.assertTrue(landed_quote.landed)
        self.assertEqual(landed_quote.landing_s, 0.0)
        self.assertEqual(landed_quote.seconds, mission)
        self.assertEqual(landed_ref["t"], mission)
        self.assertTrue(all(abs(value) < 1e-6 for value in landed_ref["interval_z"]))

        airborne = base.snapshot()
        _airborne(airborne)
        _hold_cache(airborne)
        deliveries_before = len(airborne.deliveries)
        air_quote = return_tail_nominal(airborne, 0, spec)
        air_ref = _script(airborne, spec, 1.0, 0)
        backhaul = completion_quote(airborne, 0, None, 1.0, spec)
        self.assertEqual(air_ref["interventions"], 0)
        self.assertEqual(air_ref["deliveries"], deliveries_before)
        self.assertAlmostEqual(air_quote.z_j + air_quote.aux_j, air_ref["consumed_j"], places=3)
        self.assertAlmostEqual(air_quote.z_j, air_ref["z_j"], places=3)
        self.assertGreater(air_quote.flight_z_j, 0.0)
        self.assertAlmostEqual(air_quote.ground_z_j, 0.0, places=6)
        self.assertTrue(air_quote.landed)
        self.assertGreater(air_quote.landing_s, 0.0)
        self.assertLess(air_quote.landing_s, mission)
        self.assertNotAlmostEqual(air_quote.z_j + air_quote.aux_j, backhaul.energy_j, places=0)
        touched = air_ref["airborne_after"].index(False)
        self.assertTrue(all(abs(value) < 1e-6 for value in air_ref["interval_z"][touched + 1 :]))

        scaled = _script(airborne, spec, 1.25, 0)
        self.assertAlmostEqual(1.25 * air_quote.z_j + air_quote.aux_j, scaled["consumed_j"], places=3)

        started = base.snapshot()
        _airborne(started)
        _hold_cache(started)
        first = _boost(started)
        beliefs = [prior_belief(spec), prior_belief(spec)]
        advance(
            first,
            [("calibrate", None), ("wait", None)],
            [1.0, 1.0],
            beliefs,
            spec,
            1.0,
            None,
            planning=True,
            interval_index=0,
            noise_j=lambda _index, _interval: 0.0,
        )
        first_energy = first.uavs[0].consumed_j
        post = first.snapshot()
        post.uavs[0].consumed_j = 12345.0
        tail = return_tail_nominal(post, 0, spec)
        whole = _script(started, spec, 1.0, 0, first_action=("calibrate", None))
        self.assertGreater(first_energy, 0.0)
        self.assertAlmostEqual(first_energy + tail.z_j + tail.aux_j, whole["consumed_j"], places=3)
        self.assertLess(tail.z_j + tail.aux_j, whole["consumed_j"] - 1.0)

        late = base.snapshot()
        _airborne(late)
        late.t = mission - 30.0
        late.channel_busy_until = late.t
        late_quote = return_tail_nominal(late, 0, spec)
        late_ref = _script(late, spec, 1.0, 0)
        self.assertEqual(late_ref["t"], mission)
        self.assertEqual(late_quote.seconds, 30.0)
        self.assertFalse(late_quote.landed)
        self.assertTrue(late_quote.clock_truncated)
        self.assertAlmostEqual(late_quote.z_j + late_quote.aux_j, late_ref["consumed_j"], places=3)

        done = base.snapshot()
        done.t = mission
        done.channel_busy_until = mission
        empty = return_tail_nominal(done, 0, spec)
        self.assertEqual(empty.seconds, 0.0)
        self.assertEqual(empty.z_j, 0.0)
        self.assertEqual(empty.aux_j, 0.0)

        short = base.snapshot()
        _airborne(short)
        short.uavs[0].energy_j = 100.0
        coef = float(spec["utility"]["lambda_e"]) / float(spec["utility"]["e_ref_j"])
        clipped = short.snapshot()
        clipped.uavs[0].energy_j = 100.0
        clipped.uavs[0].consumed_j = 0.0
        beliefs = [prior_belief(spec), prior_belief(spec)]
        advance(
            clipped,
            [("return", None), ("wait", None)],
            [1.0, 1.0],
            beliefs,
            spec,
            1.0,
            None,
            planning=True,
            interval_index=0,
            noise_j=lambda _index, _interval: 0.0,
        )
        self.assertAlmostEqual(clipped.uavs[0].consumed_j, 100.0, places=6)
        self.assertAlmostEqual(return_tail_value(short, 0, 1.0, spec), -coef * 100.0, places=9)
        self.assertNotAlmostEqual(return_tail_value(short, 0, 1.0, spec), -coef * (100.0 + float(spec["energy"]["reserve_j"])), places=3)

    def test_tail_replaces_only_the_none_branch_and_enters_the_integral(self) -> None:
        spec = _segment_spec()
        legacy = spec
        tail_spec = copy.deepcopy(spec)
        tail_spec["terminal"] = {"mode": TAIL_MODE}
        self.assertEqual(terminal_mode(legacy), "legacy")
        self.assertEqual(terminal_mode(tail_spec), TAIL_MODE)
        clear_rollout_cache()
        clear_return_tail_cache()

        positive = make_world(spec, [1.0e7, 1.0e7], "m15-positive")
        beliefs = [prior_belief(spec), prior_belief(spec)]
        theta = beliefs[0].quantile(float(spec["belief"]["guard_probability"]))
        legacy_terminal = terminal_value(positive, beliefs, legacy, 1.0, 0)
        tail_terminal = terminal_value(positive, beliefs, tail_spec, 1.0, 0)
        self.assertGreater(legacy_terminal, 0.0)
        self.assertAlmostEqual(tail_terminal, legacy_terminal, places=9)
        plans = extract_plans(positive, 0, tail_spec)
        self.assertAlmostEqual(value_at(theta, positive.uavs[0].energy_j, plans, tail_spec, False), tail_terminal, places=9)
        business = tuple(plan for plan in plans if plan.label != "return_tail")
        self.assertAlmostEqual(value_at(theta, positive.uavs[0].energy_j, business, legacy, False), legacy_terminal, places=9)

        none = make_world(spec, [1.0e7, 1.0e7], "m15-none")
        _clear_business(none)
        _airborne(none)
        none_beliefs = [prior_belief(spec), prior_belief(spec)]
        none_theta = none_beliefs[0].quantile(float(spec["belief"]["guard_probability"]))
        self.assertEqual(terminal_value(none, none_beliefs, legacy, 1.0, 0), 0.0)
        none_tail = terminal_value(none, none_beliefs, tail_spec, 1.0, 0)
        none_plans = extract_plans(none, 0, tail_spec)
        self.assertLess(none_tail, 0.0)
        self.assertAlmostEqual(value_at(none_theta, none.uavs[0].energy_j, none_plans, tail_spec, False), none_tail, places=9)
        self.assertTrue(any(plan.label == "return_tail" for plan in none_plans))
        self.assertFalse(any(plan.label == "return_tail" for plan in extract_plans(none, 0, legacy)))

        stats = {"rollouts": 0, "algebraic_integrals": 0, "tail_piece_evals": 0}
        roll = roll_action(none, none_beliefs, ("calibrate", None), 0, 1.1, tail_spec, 1.0, stats)
        segmented = segment_expectation(roll, tail_spec, 7, stats)
        reference = reference_expectation(roll, tail_spec, None)
        self.assertLess(segmented, 0.0)
        self.assertLess(abs(segmented - reference), 1e-4)
        self.assertGreater(int(stats["tail_piece_evals"]), 0)
        legacy_roll = roll_action(none, none_beliefs, ("calibrate", None), 0, 1.1, legacy, 1.0, stats)
        legacy_terminal_expectation = segment_expectation(legacy_roll, legacy, 7, stats)
        self.assertAlmostEqual(legacy_terminal_expectation, 0.0, places=8)
        self.assertLess(segmented, legacy_terminal_expectation - 1e-6)

        legacy_scores = {
            action: score
            for score, action in score_actions(none, none_beliefs, legacy, 1.0, 0, include_calibrate=True, stats=stats)
        }
        tail_scores = {
            action: score
            for score, action in score_actions(none, none_beliefs, tail_spec, 1.0, 0, include_calibrate=True, stats=stats)
        }
        self.assertEqual(set(legacy_scores), set(tail_scores))
        for action, score in legacy_scores.items():
            self.assertLess(tail_scores[action], score - 1e-6)


if __name__ == "__main__":
    unittest.main()
