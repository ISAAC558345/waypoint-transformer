"""Necessary checks for the active-identification M1 loop."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.belief import (  # noqa: E402
    Belief,
    normal_cdf,
    normal_pdf,
    prior_belief,
    quadrature,
    telemetry_y,
)
from wt.active_model_identification.policies import select_actions  # noqa: E402
from wt.active_model_identification.scenarios import build_scenarios  # noqa: E402
from wt.active_model_identification.spec import load_spec  # noqa: E402
from wt.active_model_identification.world import advance, make_world  # noqa: E402


def _simpson_mean(belief: Belief, panels: int = 800) -> float:
    sigma = belief.kernel_std
    alpha = (belief.lo - belief.mu) / sigma
    beta = (belief.hi - belief.mu) / sigma
    mass = normal_cdf(beta) - normal_cdf(alpha)

    def density(theta: float) -> float:
        if theta < belief.lo or theta > belief.hi:
            return 0.0
        return normal_pdf((theta - belief.mu) / sigma) / sigma / mass

    step = (belief.hi - belief.lo) / panels
    total = 0.0
    weight = 0.0
    for index in range(panels + 1):
        theta = belief.lo + index * step
        coefficient = 1 if index in (0, panels) else (4 if index % 2 else 2)
        value = density(theta)
        total += coefficient * theta * value
        weight += coefficient * value
    return total / weight


class BeliefTests(unittest.TestCase):
    def test_truncated_mean_matches_simpson_and_not_the_kernel_center(self) -> None:
        belief = Belief(1.0, 1.0 / (0.2 ** 2), 0.8, 1.4)
        mean, variance = belief.moments()
        self.assertAlmostEqual(mean, _simpson_mean(belief), places=5)
        self.assertGreater(abs(mean - belief.mu), 1.0e-3)
        self.assertGreater(variance, 0.0)
        self.assertGreater(belief.quantile(0.95), mean)

    def test_update_matches_the_conjugate_step(self) -> None:
        belief = Belief(1.1, 25.0, 0.8, 1.4)
        updated = belief.update(5000.0, 6250.0, 200.0)
        precision = 25.0 + (5000.0 ** 2) / (200.0 ** 2)
        center = (25.0 * 1.1 + 5000.0 * 6250.0 / (200.0 ** 2)) / precision
        self.assertAlmostEqual(updated.lam, precision)
        self.assertAlmostEqual(updated.mu, center)
        self.assertNotAlmostEqual(updated.mean(), updated.mu, places=6)

    def test_quadrature_budget(self) -> None:
        nodes = quadrature(prior_belief(load_spec()), 7)
        self.assertEqual(len(nodes), 7)
        self.assertAlmostEqual(sum(weight for _theta, weight in nodes), 1.0)

    def test_service_and_calibration_share_the_observation(self) -> None:
        self.assertEqual(telemetry_y(1200.0, 200.0, 15.0), telemetry_y(1200.0, 200.0, 15.0))
        y_service = telemetry_y(5000.0, 150.0, -20.0)
        y_cal = telemetry_y(5000.0, 150.0, -20.0)
        belief = prior_belief(load_spec())
        self.assertEqual(belief.update(4850.0, y_service, 180.0), belief.update(4850.0, y_cal, 180.0))


class AccountingTests(unittest.TestCase):
    def test_meter_identity_and_nonoverlapping_intervals(self) -> None:
        spec = load_spec()
        world = make_world(spec, [1.0e7, 1.0e7], "meter")
        beliefs = [prior_belief(spec), prior_belief(spec)]
        theta = [1.2, 1.2]
        advance(world, [("calibrate", None), ("wait", None)], theta, beliefs, spec, 1.05, None, planning=True, interval_index=0, noise_j=lambda *_: 0.0)
        uav = world.uavs[0]
        self.assertGreater(uav.z_j, 1000.0)
        self.assertAlmostEqual(uav.true_drop_j, theta[0] * uav.z_j + uav.aux_j, places=6)
        self.assertAlmostEqual(telemetry_y(uav.true_drop_j, uav.aux_j, 0.0), theta[0] * uav.z_j, places=6)
        advance(world, [("wait", None), ("wait", None)], theta, beliefs, spec, 1.05, None, planning=True, interval_index=1, noise_j=lambda *_: 0.0)
        self.assertEqual(world.interval_windows, [(0, 0.0, 30.0), (1, 30.0, 60.0)])

    def test_packet_is_credited_once(self) -> None:
        spec = load_spec()
        _scenarios, _budget, factor = build_scenarios(spec)
        from wt.active_model_identification.scenarios import _point_belief, _run, _service_script

        beliefs = [_point_belief(spec, 1.0), _point_belief(spec, 1.0)]
        world = _run(spec, factor, [1.0e7, 1.0e7], beliefs, [1.0, 1.0], _service_script(spec, 0, 0), 20, "once")
        bits = sum(packet.bits for packet in world.packets if packet.buoy == 0 and packet.holder == "shore")
        self.assertEqual(bits, 20000.0)
        self.assertEqual(sum(1 for packet in world.packets if packet.buoy == 0 and packet.holder == "shore"), 2)


class PlannerTests(unittest.TestCase):
    def test_copy_does_not_leak_and_uses_the_sampled_theta(self) -> None:
        spec = load_spec()
        world = make_world(spec, [1.0e7, 1.0e7], "copy")
        world.truth_theta = (9.0, 9.0)  # type: ignore[attr-defined]
        energy = world.uavs[0].energy_j
        belief = prior_belief(spec)
        snap = world.snapshot()
        copied = [belief.copy(), belief.copy()]
        advance(
            snap,
            [("calibrate", None), ("wait", None)],
            [1.25, 1.0],
            copied,
            spec,
            1.05,
            None,
            planning=True,
            interval_index=0,
            noise_j=lambda *_: 0.0,
        )
        self.assertEqual(world.uavs[0].energy_j, energy)
        self.assertEqual(belief.mu, prior_belief(spec).mu)
        self.assertGreater(abs(copied[0].mu - belief.mu), 0.01)
        self.assertLess(abs(copied[0].mean() - 1.25), abs(copied[0].mean() - 9.0))
        self.assertFalse(hasattr(snap, "truth_theta"))
        snap.uavs[0].energy_j -= 10.0
        self.assertEqual(world.uavs[0].energy_j, energy)

    def test_budget_and_scheduler_choices(self) -> None:
        spec = load_spec()
        scenarios, budget, factor = build_scenarios(spec)
        self.assertGreater(budget["integration_factor"], 1.0)
        self.assertLess(budget["integration_factor"], 1.5)
        self.assertEqual(len(scenarios), 8)
        sensitive = next(item for item in scenarios if item["id"] == "b1_sensitive")
        insensitive = next(item for item in scenarios if item["id"] == "b1_insensitive")
        for scenario, expect_calibrate, expect_far in (
            (sensitive, True, False),
            (insensitive, False, True),
        ):
            world = make_world(spec, scenario["battery_j"], scenario["id"])
            beliefs = [prior_belief(spec), prior_belief(spec)]
            passive = select_actions("passive", world, beliefs, spec, factor, 0, None)
            fixed = select_actions("fixed_calibration", world, beliefs, spec, factor, 0, None)
            lookahead = select_actions("finite_lookahead", world, [item.copy() for item in beliefs], spec, factor, 0, None)
            self.assertTrue(all(action[0] != "calibrate" for action in passive))
            self.assertTrue(all(action[0] == "calibrate" for action in fixed))
            self.assertEqual(any(action[0] == "calibrate" for action in lookahead), expect_calibrate)
            far_ids = {1, 3}
            requested_far = any(action[0] == "service" and action[1] in far_ids for action in lookahead)
            self.assertEqual(requested_far, expect_far)
            self.assertEqual(world.uavs[0].energy_j, scenario["battery_j"][0])


if __name__ == "__main__":
    unittest.main()
