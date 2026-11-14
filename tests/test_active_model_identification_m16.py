"""Necessary checks for the M1.6 initial-guard selection mode.

The frozen 180 s snapshots are scored by the trial runner. These tests cover
the selection rule on two constructed states and the unchanged repair rule.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.active_model_identification.belief import prior_belief  # noqa: E402
from wt.active_model_identification.completion import clear_completion_cache  # noqa: E402
from wt.active_model_identification.policies import (  # noqa: E402
    GUARD_SELECTION,
    _repair,
    initial_effective_action,
    score_actions,
    select_lookahead,
)
from wt.active_model_identification.segment import clear_rollout_cache  # noqa: E402
from wt.active_model_identification.spec import load_m13_spec  # noqa: E402
from wt.active_model_identification.world import make_world  # noqa: E402


def _spec(mode: str) -> dict:
    spec = copy.deepcopy(load_m13_spec())
    spec["integral"]["scoring"] = "segment"
    spec["integral"]["segment_inner_nodes"] = 3
    spec["belief"]["quadrature_nodes"] = 3
    spec["selection"] = {"mode": mode}
    return spec


def _only_buoy(world, buoy_id: int) -> None:
    for packet in world.packets:
        if packet.buoy == buoy_id:
            continue
        packet.holder = "shore"
        packet.remaining_bits = 0.0
        packet.delivered_s = 0.0


class InitialGuardSelectionTests(unittest.TestCase):
    def test_admissible_conflict_keeps_the_original_repair(self) -> None:
        clear_completion_cache()
        clear_rollout_cache()
        legacy = _spec("legacy")
        guarded = _spec(GUARD_SELECTION)
        world = make_world(legacy, [1.0e8, 1.0e8], "m16-conflict")
        _only_buoy(world, 0)
        beliefs = [prior_belief(legacy), prior_belief(legacy)]
        legacy_joint, legacy_tables = select_lookahead(
            "lookahead_calibration_off", world, beliefs, legacy, 1.0, {}
        )
        guard_joint, guard_tables = select_lookahead(
            "lookahead_calibration_off", world, [item.copy() for item in beliefs], guarded, 1.0, {}
        )
        self.assertEqual(guard_joint, legacy_joint)
        self.assertTrue(all(action == ("service", 0) or action[0] != "service" for _score, action in guard_tables[0]))
        for index, (left, right) in enumerate(zip(legacy_tables, guard_tables)):
            self.assertEqual([action for _score, action in left], [action for _score, action in right])
            for (left_score, _left_action), (right_score, _right_action) in zip(left, right):
                self.assertAlmostEqual(left_score, right_score, places=12)
            self.assertEqual(
                initial_effective_action(world, index, guard_joint[index], beliefs[index], guarded, 1.0),
                guard_joint[index],
            )
        tied = [
            [(1.0, ("service", 0)), (0.2, ("return", None))],
            [(1.0, ("service", 0)), (0.2, ("backhaul", None))],
        ]
        chosen = [(1.0, ("service", 0)), (1.0, ("service", 0))]
        self.assertEqual(_repair(chosen, tied, 1.0e-4), [("service", 0), ("backhaul", None)])
        weaker_first = _repair([(0.5, ("service", 0)), (1.0, ("service", 0))], tied, 1.0e-4)
        self.assertEqual(weaker_first, [("return", None), ("service", 0)])

    def test_inadmissible_service_does_not_occupy_the_target(self) -> None:
        clear_completion_cache()
        clear_rollout_cache()
        legacy = _spec("legacy")
        guarded = _spec(GUARD_SELECTION)
        world = make_world(legacy, [1.0e8, 8000.0], "m16-guard")
        _only_buoy(world, 0)
        poor = world.uavs[1]
        poor.x = 1850.0
        poor.y = 1600.0
        poor.vx = 0.0
        poor.vy = 0.0
        poor.airborne = True
        poor.land_timer_s = 0.0
        beliefs = [prior_belief(legacy), prior_belief(legacy)]
        self.assertEqual(
            initial_effective_action(world, 0, ("service", 0), beliefs[0], guarded, 1.0),
            ("service", 0),
        )
        self.assertEqual(
            initial_effective_action(world, 1, ("service", 0), beliefs[1], guarded, 1.0),
            ("return", None),
        )
        legacy_table = score_actions(world, beliefs, legacy, 1.0, 1, include_calibrate=False, stats={})
        clear_rollout_cache()
        stats: dict = {}
        guard_table = score_actions(world, beliefs, guarded, 1.0, 1, include_calibrate=False, stats=stats)
        guard_labels = [action for _score, action in guard_table]
        self.assertIn(("service", 0), [action for _score, action in legacy_table])
        self.assertNotIn(("service", 0), guard_labels)
        self.assertEqual(guard_labels.count(("return", None)), 1)
        self.assertEqual(stats["rollouts"], len(guard_table) * 3)
        self.assertGreater(stats["selection_aliases"], 0)
        joint, tables = select_lookahead("lookahead_calibration_off", world, beliefs, guarded, 1.0, {})
        self.assertNotEqual(joint[1], ("service", 0))
        for index, action in enumerate(joint):
            self.assertIn(action, [item[1] for item in tables[index]])
            self.assertEqual(
                initial_effective_action(world, index, action, beliefs[index], guarded, 1.0),
                action,
            )


if __name__ == "__main__":
    unittest.main()
