"""Necessary checks for the M1.3 segmented integral and three-budget pilot."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.belief import prior_belief  # noqa: E402
from wt.active_model_identification.completion import standard_quantile  # noqa: E402
from wt.active_model_identification.policies import score_actions, terminal_value  # noqa: E402
from wt.active_model_identification.scenarios import assess_m13  # noqa: E402
from wt.active_model_identification.segment import (  # noqa: E402
    action_values,
    clear_rollout_cache,
    extract_plans,
    fast_truncated_quantile,
    inv_norm,
    roll_action,
    select_integral_m13,
    value_at,
)
from wt.active_model_identification.spec import load_m13_spec  # noqa: E402
from wt.active_model_identification.world import advance, make_world, mission_terminal  # noqa: E402


def _segment_spec() -> dict:
    spec = copy.deepcopy(load_m13_spec())
    spec["integral"]["scoring"] = "segment"
    spec["integral"]["segment_inner_nodes"] = 7
    spec["belief"]["quadrature_nodes"] = 7
    return spec


class SegmentTests(unittest.TestCase):
    def test_inverse_normal_matches_the_erf_bisection(self) -> None:
        for probability in (0.05, 0.5, 0.95, 0.975):
            self.assertAlmostEqual(inv_norm(probability), standard_quantile(probability), places=6)

    def test_fast_quantile_matches_the_belief_bisection(self) -> None:
        belief = prior_belief(load_m13_spec())
        for probability in (0.05, 0.5, 0.95):
            fast = fast_truncated_quantile(belief.mu, belief.lam, belief.lo, belief.hi, probability)
            self.assertAlmostEqual(fast, belief.quantile(probability), places=6)

    def test_physical_interval_does_not_depend_on_epsilon(self) -> None:
        spec = load_m13_spec()
        world = make_world(spec, [1.0e7, 1.0e7], "m13-eps")
        beliefs = [prior_belief(spec), prior_belief(spec)]

        def run(epsilon: float):
            snap = world.snapshot()
            copied = [item.copy() for item in beliefs]
            observations = advance(
                snap,
                [("service", 1), ("wait", None)],
                [1.0, 1.1],
                copied,
                spec,
                1.0,
                None,
                planning=True,
                interval_index=0,
                noise_j=lambda index, _interval, _epsilon=epsilon: _epsilon if index == 0 else 0.0,
            )
            uav = snap.uavs[0]
            return (
                uav.x,
                uav.y,
                uav.vx,
                uav.vy,
                uav.z_j,
                uav.aux_j,
                uav.true_drop_j,
                uav.consumed_j,
                uav.energy_j,
                tuple(observations[0]["executed"]),
                observations[0]["y_j"],
            )

        quiet = run(0.0)
        loud = run(250.0)
        self.assertEqual(quiet[:-1], loud[:-1])
        self.assertNotEqual(quiet[-1], loud[-1])
        clear_rollout_cache()
        stats = {"rollouts": 0, "cache_hits": 0}
        first = roll_action(world, beliefs, ("wait", None), 0, 1.1, spec, 1.0, stats)
        second = roll_action(world, beliefs, ("wait", None), 0, 1.1, spec, 1.0, stats)
        self.assertEqual(stats["rollouts"], 1)
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(first.immediate, second.immediate)
        self.assertEqual(first.z_j, second.z_j)
        self.assertEqual(first.energy_j, second.energy_j)

    def test_plan_value_matches_terminal_value(self) -> None:
        spec = load_m13_spec()
        world = make_world(spec, [80000.0, 80000.0], "m13-terminal")
        beliefs = [prior_belief(spec), prior_belief(spec)]
        snap = world.snapshot()
        advance(
            snap,
            [("service", 1), ("wait", None)],
            [1.1, 1.1],
            beliefs,
            spec,
            1.0,
            None,
            planning=True,
            interval_index=0,
            noise_j=lambda _index, _interval: 0.0,
        )
        theta = beliefs[0].quantile(float(spec["belief"]["guard_probability"]))
        direct = terminal_value(snap, beliefs, spec, 1.0, 0)
        plans = extract_plans(snap, 0, spec)
        self.assertAlmostEqual(value_at(theta, snap.uavs[0].energy_j, plans, spec, snap.uavs[0].failed), direct, places=9)

    def test_segment_matches_independent_reference(self) -> None:
        spec = _segment_spec()
        world = make_world(spec, [1.0e7, 1.0e7], "m13-reference")
        beliefs = [prior_belief(spec), prior_belief(spec)]
        clear_rollout_cache()
        segment = action_values(
            world, beliefs, spec, 1.0, 0, include_calibrate=True, theta_nodes=7, noise="segment", stats={"rollouts": 0}
        )
        reference = action_values(
            world, beliefs, spec, 1.0, 0, include_calibrate=True, theta_nodes=7, noise="reference", stats={"rollouts": 0}
        )
        left = {action: score for score, action in segment}
        right = {action: score for score, action in reference}
        self.assertEqual(set(left), set(right))
        self.assertLess(max(abs(left[action] - right[action]) for action in left), 0.01)

    def test_permission_switch_keeps_shared_action_values(self) -> None:
        spec = _segment_spec()
        world = make_world(spec, [1.0e7, 1.0e7], "m13-share")
        beliefs = [prior_belief(spec), prior_belief(spec)]
        clear_rollout_cache()
        stats = {"rollouts": 0}
        off = score_actions(world, beliefs, spec, 1.0, 0, include_calibrate=False, stats=stats)
        after_off = int(stats["rollouts"])
        on = score_actions(world, beliefs, spec, 1.0, 0, include_calibrate=True, stats=stats)
        self.assertEqual(int(stats["rollouts"]) - after_off, 7)
        off_map = {action: score for score, action in off}
        on_map = {action: score for score, action in on}
        for action, score in off_map.items():
            self.assertEqual(on_map[action], score)
        self.assertIn(("calibrate", None), on_map)
        self.assertNotIn(("calibrate", None), off_map)

    def test_window_formula_and_tight_missions_stay_scheduled(self) -> None:
        assessed = assess_m13(load_m13_spec())
        self.assertFalse(assessed["posterior_zero_admits"])
        self.assertEqual(assessed["tight_battery_j"], [52156.035109573495, 52156.035109573495])
        self.assertEqual(len(assessed["loose_battery_j"]), 2)
        for value in assessed["loose_battery_j"]:
            self.assertAlmostEqual(value, 70355.51783623741, places=9)
        self.assertTrue(assessed["window_valid"])
        for row in assessed["uavs"]:
            self.assertLess(row["L_j"], row["U_j"])
            self.assertAlmostEqual(row["window_battery_j"], 0.5 * (row["L_j"] + row["U_j"]))
        tight = [row for row in assessed["scenarios"] if row["condition"] == "tight"]
        self.assertEqual(len(tight), 4)
        self.assertTrue(all(row["scheduled"] for row in tight))
        window = {tuple(row["battery_j"]) for row in assessed["scenarios"] if row["condition"] == "window"}
        self.assertEqual(len(window), 1)
        self.assertGreater(assessed["uavs"][0]["window_battery_j"], assessed["tight_battery_j"][0])
        self.assertLess(assessed["uavs"][0]["window_battery_j"], assessed["loose_battery_j"][0])

    def test_terminal_flags_separate_landing_from_exhaustion(self) -> None:
        spec = load_m13_spec()
        world = make_world(spec, [50000.0, 50000.0], "m13-flags")
        landed = mission_terminal(world, spec)[0]
        self.assertTrue(landed["returned"])
        self.assertFalse(landed["energy_exhausted"])
        self.assertFalse(landed["not_returned"])
        world.uavs[0].x = 800.0
        world.uavs[0].airborne = False
        world.uavs[0].failed = True
        world.uavs[0].energy_j = 0.0
        exhausted = mission_terminal(world, spec)[0]
        self.assertTrue(exhausted["energy_exhausted"])
        self.assertTrue(exhausted["not_returned"])
        self.assertFalse(exhausted["returned"])

    def test_rollout_budget_does_not_promote_a_grid(self) -> None:
        spec = copy.deepcopy(load_m13_spec())
        spec["integral"]["development_rollout_budget"] = 2
        result = select_integral_m13(spec, 1.0)
        self.assertEqual(result["numerical_status"], "NOT_MET")
        self.assertIsNone(result["theta_nodes"])
        self.assertIsNone(result["segment_inner_nodes"])
        self.assertTrue(result["budget_exhausted"])
        self.assertLessEqual(result["development_rollouts"], 2)
        self.assertEqual(result["scoring"], "frozen")


if __name__ == "__main__":
    unittest.main()
