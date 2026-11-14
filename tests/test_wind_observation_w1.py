"""Necessary checks for the W1 wind pilot.

zero wind: a saved-request replay that drifts from the M1.6 actions, bits,
energy, or return flags means the new wind arguments changed the calm scheduler.

relative wind: joules must equal P(|v_ground - wind|) * dt. A second wind factor,
or power taken from ground speed, fails this identity. Headwind on a westbound
leg must cost more than calm, and tailwind less.

deadline: a late copy of the same message must be absent from the decision
prediction, and a script replay must leave the source ledger unchanged.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.belief import prior_belief  # noqa: E402
from wt.active_model_identification.physics import propulsion_power_w, step_kinematics  # noqa: E402
from wt.active_model_identification.run_w1 import load_case, replay_case  # noqa: E402
from wt.active_model_identification.wind import (  # noqa: E402
    fly_script,
    message_cost_j,
    observe,
    predict,
)
from wt.active_model_identification.world import advance, make_world  # noqa: E402


class WindObservationW1Test(unittest.TestCase):
    def test_zero_wind_matches_m16_replay(self):
        found = replay_case()
        self.assertTrue(found["ok"], found["mismatches"])
        self.assertEqual(int(round(found["bits"])), 140000)
        self.assertEqual(found["returned"], 2)

    def test_relative_wind_energy_is_applied_once(self):
        calm = step_kinematics(0.0, 0.0, -18.0, 0.0, -18.0, 0.0, True, 1.0)
        explicit = step_kinematics(0.0, 0.0, -18.0, 0.0, -18.0, 0.0, True, 1.0, (0.0, 0.0))
        self.assertEqual(calm[-1], explicit[-1])
        head = step_kinematics(0.0, 0.0, -18.0, 0.0, -18.0, 0.0, True, 1.0, (6.0, 0.0))
        tail = step_kinematics(0.0, 0.0, -18.0, 0.0, -18.0, 0.0, True, 1.0, (-6.0, 0.0))
        self.assertEqual(head[-1], propulsion_power_w(24.0))
        self.assertEqual(tail[-1], propulsion_power_w(12.0))
        self.assertNotEqual(head[-1], propulsion_power_w(18.0) + propulsion_power_w(6.0))
        self.assertGreater(head[-1], calm[-1])
        self.assertLess(tail[-1], calm[-1])
        case = load_case()
        self.assertEqual(self._leg(case, None), self._leg(case, (0.0, 0.0)))
        self.assertGreater(self._leg(case, (6.0, 0.0)), self._leg(case, (0.0, 0.0)))
        self.assertLess(self._leg(case, (-6.0, 0.0)), self._leg(case, (0.0, 0.0)))

    def test_late_message_does_not_enter_the_decision_or_the_source_ledger(self):
        case = load_case()
        cfg = case["cfg"]
        field = cfg["field"]
        condition = cfg["conditions"][0]
        rng = random.Random("w1-deadline-check")
        message = observe(condition, 100.0, 200.0, 150.0, field, float(cfg["observation_sigma_m_s"]), rng)
        timely = predict(message, 0.0, 0.0, 180.0, field, float(cfg["prior_std_m_s"]))
        late = predict(None, 0.0, 0.0, 180.0, field, float(cfg["prior_std_m_s"]))
        self.assertTrue(timely["used_message"])
        self.assertFalse(late["used_message"])
        self.assertNotEqual(timely["wind_xy"], late["wind_xy"])
        cost = message_cost_j(int(cfg["payload_bytes"]))
        self.assertEqual(cost, message_cost_j(int(cfg["payload_bytes"])))
        self.assertGreater(cost["sender_j"], 0.0)
        self.assertGreater(cost["receiver_j"], 0.0)
        spec = case["spec"]
        world = make_world(spec, [1.0e8, 1.0e8], "w1-ledger")
        uav = world.uavs[1]
        uav.airborne = True
        uav.x = uav.home_x + 400.0
        uav.y = uav.home_y
        before = (uav.consumed_j, uav.energy_j, world.t)
        beliefs = [prior_belief(spec), prior_belief(spec)]
        fly_script(
            world,
            beliefs,
            spec,
            case["factor"],
            1,
            ("return", None),
            [1.0, 1.0],
            lambda x, y, t: (0.0, 0.0),
            lambda _uav, _interval: 0.0,
        )
        self.assertEqual((world.uavs[1].consumed_j, world.uavs[1].energy_j, world.t), before)
        self.assertTrue(math.isfinite(before[0]))

    def test_direct_return_keeps_the_cache(self):
        case = load_case()
        spec = case["spec"]
        world = make_world(spec, [1.0e8, 1.0e8], "w1-cache")
        packet = world.packets[0]
        packet.holder = "uav:1"
        world.uavs[1].cache_ids.append(packet.id)
        uav = world.uavs[1]
        uav.airborne = True
        uav.x = uav.home_x + 400.0
        uav.y = uav.home_y
        beliefs = [prior_belief(spec), prior_belief(spec)]
        common = dict(
            spec=spec,
            factor=case["factor"],
            uav_index=1,
            thetas=[1.0, 1.0],
            wind_at=lambda x, y, t: (0.0, 0.0),
            noise=lambda _uav, _interval: 0.0,
        )
        returned = fly_script(world, beliefs, action=("return", None), **common)
        delivered = fly_script(world, beliefs, action=("backhaul", None), **common)
        self.assertEqual(returned["on_time_bits"], 0.0)
        self.assertGreater(delivered["on_time_bits"], 0.0)
        self.assertEqual(world.packets[0].holder, "uav:1")

    def _leg(self, case: dict, wind):
        spec = case["spec"]
        world = make_world(spec, [1.0e8, 1.0e8], "w1-leg")
        uav = world.uavs[1]
        uav.airborne = True
        uav.x = uav.home_x + 900.0
        uav.y = uav.home_y
        uav.vx = -18.0
        uav.vy = 0.0
        beliefs = [prior_belief(spec), prior_belief(spec)]
        kwargs = {}
        if wind is not None:
            kwargs["wind_at"] = lambda _x, _y, _t, wind=wind: wind
        advance(
            world,
            [("wait", None), ("return", None)],
            [1.0, 1.0],
            beliefs,
            spec,
            case["factor"],
            None,
            planning=False,
            interval_index=0,
            noise_j=lambda _uav, _interval: 0.0,
            **kwargs,
        )
        return world.uavs[1].consumed_j


if __name__ == "__main__":
    unittest.main()
