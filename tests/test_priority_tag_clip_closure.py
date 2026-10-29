"""Closure checks. They read the finished comparison and do not train."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_priority_tag_clip_closure.review import (
    audit,
    counter_flags,
    pair_distance,
    projector_accepts_separation,
)


class ClosureRules(unittest.TestCase):
    def test_counter_and_projector_use_different_bands(self) -> None:
        distance = 1.0 - 1.0e-8
        self.assertTrue(counter_flags(distance))
        self.assertTrue(projector_accepts_separation(distance))
        self.assertFalse(counter_flags(1.0))
        self.assertFalse(projector_accepts_separation(1.0 - 1.0e-5))

    def test_pair_distance_uses_the_saved_endpoints(self) -> None:
        self.assertAlmostEqual(pair_distance([0.0, 0.0], [0.4, 0.0]), 0.4)

    def test_boundary_label_does_not_drop_the_saved_count(self) -> None:
        body = audit()
        self.assertEqual(body["safety"]["safety_violations"], 40)
        self.assertEqual(body["safety"]["constraint_failures"], 1)
        self.assertEqual(body["safety"]["boundary_count"], 40)
        self.assertTrue(body["safety"]["boundary_does_not_reduce_count"])
        self.assertTrue(body["safety"]["constraint_review"]["supported"])
        self.assertTrue(body["safety"]["constraint_review"]["not_a_prediction_error"])
        self.assertEqual(body["recomputed_e15_gate"], "CLIP_GROUP_BENCHMARK_INCONCLUSIVE")
        self.assertEqual(body["score"]["passing_seeds"], 0)
        self.assertEqual(body["budget"]["ppo_updates"], 1800)
        self.assertEqual(body["budget"]["adam_steps"], 28800)
        self.assertTrue(body["consistent"], body["problems"])

    def test_new_sources_omit_forbidden_tokens(self) -> None:
        token = "deci" + "de("
        held = ("322" + "048", "322" + "064")
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_clip_closure"
        for path in list(root.glob("*.py")) + [Path(__file__)]:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(token, text)
            for item in held:
                self.assertNotIn(item, text)
