"""Necessary checks for the M1.1 accounting correction and matched planner."""

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
    gaussian_quadrature,
    prior_belief,
    quadrature,
)
from wt.active_model_identification.policies import candidates, select_lookahead, terminal_value  # noqa: E402
from wt.active_model_identification.scenarios import ablate_beliefs  # noqa: E402
from wt.active_model_identification.spec import load_m11_spec, load_spec  # noqa: E402
from wt.active_model_identification.world import (  # noqa: E402
    Packet,
    _move,
    advance,
    make_world,
    mission_terminal,
    on_time_bits,
)


def _spec_fast() -> dict:
    spec = copy.deepcopy(load_m11_spec())
    spec["integral"]["noise_nodes"] = 1
    spec["integral"]["tie_abs"] = 0.0001
    spec["belief"]["quadrature_nodes"] = 7
    return spec


class IntegralTests(unittest.TestCase):
    def test_narrow_quantile_nodes_follow_the_truncated_mean(self) -> None:
        belief = Belief(1.25, 1.0 / (0.02 ** 2), 0.8, 1.4)
        width_mean = sum(weight * theta for theta, weight in equal_width_quadrature(belief, 7))
        quantile_mean = sum(weight * theta for theta, weight in quadrature(belief, 7))
        self.assertGreater(width_mean, 1.26)
        self.assertLess(abs(quantile_mean - belief.mean()), 0.002)
        edge = Belief(0.82, 1.0 / (0.05 ** 2), 0.8, 1.4)
        edge_mean = sum(weight * theta for theta, weight in quadrature(edge, 15))
        self.assertLess(abs(edge_mean - edge.mean()), 0.002)

    def test_noise_nodes_match_the_gaussian_second_moment(self) -> None:
        nodes = gaussian_quadrature(180.0, 3)
        self.assertAlmostEqual(sum(weight for _epsilon, weight in nodes), 1.0)
        second = sum(weight * epsilon * epsilon for epsilon, weight in nodes)
        self.assertAlmostEqual(second, 180.0 ** 2, places=6)
        self.assertGreater(sum(weight * max(epsilon, 0.0) for epsilon, weight in nodes), 0.0)
        self.assertEqual(gaussian_quadrature(180.0, 1), [(0.0, 1.0)])


class AccountingTests(unittest.TestCase):
    def test_remote_backhaul_and_wait_stay_airborne(self) -> None:
        spec = load_spec()
        world = make_world(spec, [1.0e7, 1.0e7], "air")
        uav = world.uavs[0]
        uav.x = 800.0
        uav.vx = 18.0
        uav.airborne = True
        uav.energy_j = 1.0e7
        _move(uav, ("backhaul", None), spec, 1.1, 0)
        self.assertTrue(uav.airborne)
        self.assertGreater(uav.vx, 15.0)
        self.assertGreater(uav.z_j, 100.0)
        self.assertAlmostEqual(uav.true_drop_j, 1.1 * uav.z_j + uav.aux_j, places=6)
        beliefs = [prior_belief(spec), prior_belief(spec)]
        advance(world, [("backhaul", None), ("wait", None)], [1.1, 1.0], beliefs, spec, 1.05, None, planning=True, interval_index=0, noise_j=lambda *_: 0.0)
        self.assertTrue(world.uavs[0].airborne)
        self.assertGreater(world.uavs[0].z_j, 4000.0)
        advance(world, [("wait", None), ("wait", None)], [1.1, 1.0], beliefs, spec, 1.05, None, planning=True, interval_index=1, noise_j=lambda *_: 0.0)
        self.assertTrue(world.uavs[0].airborne)
        self.assertGreater(world.uavs[0].z_j, 4000.0)

    def test_pad_landing_then_wait_has_no_propulsion(self) -> None:
        spec = load_spec()
        world = make_world(spec, [1.0e7, 1.0e7], "land")
        uav = world.uavs[0]
        uav.airborne = True
        uav.energy_j = 1.0e7
        for phase in range(5):
            _move(uav, ("return", None), spec, 1.0, phase)
        self.assertFalse(uav.airborne)
        self.assertGreater(uav.z_j, 100.0)
        self.assertAlmostEqual(uav.true_drop_j, uav.z_j + uav.aux_j, places=5)
        beliefs = [prior_belief(spec), prior_belief(spec)]
        advance(world, [("wait", None), ("wait", None)], [1.0, 1.0], beliefs, spec, 1.05, None, planning=True, interval_index=0, noise_j=lambda *_: 0.0)
        self.assertFalse(world.uavs[0].airborne)
        self.assertEqual(world.uavs[0].z_j, 0.0)
        self.assertGreater(world.uavs[0].aux_j, 30.0)
        self.assertLess(world.uavs[0].aux_j, 31.0)
        flags = mission_terminal(world, spec)
        self.assertTrue(flags[0]["returned"])
        self.assertFalse(flags[0]["energy_exhausted"])

    def test_delivery_uses_channel_completion_and_terminal_flags_split_failure_modes(self) -> None:
        spec = load_spec()
        world = make_world(spec, [1.0e7, 1.0e7], "deliver")
        world.packets = [
            Packet("late", 0, 8.0, 8.0, 0.0, 0.01, holder="uav:0"),
            Packet("ok", 0, 8.0, 8.0, 0.0, 1800.0, holder="uav:0"),
        ]
        world.uavs[0].cache_ids = ["late", "ok"]
        world.uavs[0].airborne = True
        world.uavs[0].x = 800.0
        world.uavs[0].energy_j = 1.0e7
        beliefs = [prior_belief(spec), prior_belief(spec)]
        advance(world, [("backhaul", None), ("wait", None)], [1.0, 1.0], beliefs, spec, 1.05, None, planning=True, interval_index=0, noise_j=lambda *_: 0.0)
        late = next(packet for packet in world.packets if packet.id == "late")
        ok = next(packet for packet in world.packets if packet.id == "ok")
        self.assertIsNotNone(late.delivered_s)
        self.assertGreater(late.delivered_s, 0.01)
        self.assertLess(late.delivered_s, 30.0)
        self.assertGreater(ok.delivered_s, late.delivered_s)
        self.assertEqual(on_time_bits(world), 8.0)
        self.assertEqual(len(world.deliveries), 2)
        away = make_world(spec, [1.0e7, 1.0e7], "away")
        away.uavs[0].x = 900.0
        away.uavs[0].airborne = True
        away.uavs[0].energy_j = 1.0e7
        flags = mission_terminal(away, spec)
        self.assertTrue(flags[0]["not_returned"])
        self.assertTrue(flags[0]["not_landed"])
        self.assertFalse(flags[0]["energy_exhausted"])
        self.assertFalse(flags[0]["returned"])

    def test_removed_calibration_observation_is_not_reused(self) -> None:
        spec = load_spec()
        world = make_world(spec, [1.0e7, 1.0e7], "once-obs")
        beliefs = [prior_belief(spec), prior_belief(spec)]
        world.assimilated_ids.append("u0-i0")
        before = beliefs[0].copy()
        advance(world, [("calibrate", None), ("wait", None)], [1.25, 1.0], beliefs, spec, 1.05, None, planning=True, interval_index=0, noise_j=lambda *_: 0.0)
        self.assertEqual(beliefs[0].mu, before.mu)
        self.assertEqual(beliefs[0].lam, before.lam)
        updated = before.update(5000.0, 6000.0, 180.0)
        kept, removed = ablate_beliefs([before, before], [updated, updated.update(100.0, 80.0, 180.0)], [("calibrate", None), ("service", 0)])
        self.assertEqual(removed, [0])
        self.assertEqual(kept[0].mu, before.mu)
        self.assertNotEqual(kept[1].mu, before.mu)


class PlannerTests(unittest.TestCase):
    def test_calibration_switch_is_the_only_candidate_change(self) -> None:
        spec = _spec_fast()
        world = make_world(spec, [1.0e7, 1.0e7], "switch")
        self.assertTrue(any(action[0] == "calibrate" for action in candidates(spec, world, 0, include_calibrate=True)))
        self.assertTrue(all(action[0] != "calibrate" for action in candidates(spec, world, 0, include_calibrate=False)))
        beliefs = [prior_belief(spec), prior_belief(spec)]
        off, off_tables = select_lookahead("lookahead_calibration_off", world, beliefs, spec, 1.05, None)
        on, on_tables = select_lookahead("lookahead_calibration_on", world, [item.copy() for item in beliefs], spec, 1.05, None)
        self.assertTrue(all(action[0] != "calibrate" for action in off))
        self.assertTrue(all(item[1][0] != "calibrate" for table in off_tables for item in table))
        self.assertTrue(any(item[1][0] == "calibrate" for table in on_tables for item in table))
        self.assertEqual(world.uavs[0].energy_j, 1.0e7)
        _ = on

    def test_cached_bits_need_time_and_energy(self) -> None:
        spec = load_spec()
        world = make_world(spec, [1.0e7, 1.0e7], "cache")
        packet = world.packets[0]
        packet.holder = "uav:0"
        world.uavs[0].cache_ids = [packet.id]
        beliefs = [prior_belief(spec), prior_belief(spec)]
        credited = terminal_value(world, beliefs, spec, 1.05, 0)
        self.assertGreater(credited, 0.0)
        world.t = float(spec["time"]["mission_s"]) - 0.1
        self.assertEqual(terminal_value(world, beliefs, spec, 1.05, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
