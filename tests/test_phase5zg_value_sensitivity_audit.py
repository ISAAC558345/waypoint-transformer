"""Locked sensitivity rules. Thresholds are not retuned from confirmation scores."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.phase5zg_value_sensitivity_audit.metrics import (
    choose_gate,
    group_dot,
    sign_agreement,
    trapezoid,
    unique_dominant,
)
from wt.lowlevel_v2.phase5zg_value_sensitivity_audit.protocol import GROUPS, assert_consumed


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5zg_value_sensitivity_audit"


class Phase5ZGTests(unittest.TestCase):
    def test_groups_cover_each_response_once(self) -> None:
        covered = [index for indexes in GROUPS.values() for index in indexes]
        self.assertEqual(sorted(covered), list(range(14)))
        self.assertEqual(len(covered), len(set(covered)))

    def test_gate_priority(self) -> None:
        self.assertEqual(choose_gate(aligned=False, first_order_ok=True, interaction_ok=True, ordering_failure=True), "SENS-D")
        self.assertEqual(choose_gate(aligned=True, first_order_ok=True, interaction_ok=True, ordering_failure=True), "SENS-A")
        self.assertEqual(choose_gate(aligned=True, first_order_ok=False, interaction_ok=True, ordering_failure=True), "SENS-B")
        self.assertEqual(choose_gate(aligned=True, first_order_ok=False, interaction_ok=False, ordering_failure=True), "SENS-C")
        self.assertEqual(choose_gate(aligned=True, first_order_ok=False, interaction_ok=False, ordering_failure=False), "SENS-E")

    def test_trapezoid_and_group_dot(self) -> None:
        constant = np.ones((16, 4))
        self.assertTrue(np.allclose(trapezoid(constant), np.ones(4)))
        line = np.linspace(0.0, 1.0, 16).reshape(16, 1)
        self.assertAlmostEqual(float(np.asarray(trapezoid(line)).reshape(-1)[0]), 0.5)
        gradient = np.arange(24, dtype=np.float64).reshape(1, 2, 12)
        error = np.ones((1, 2, 12))
        pieces = [group_dot(gradient, error, indexes) for indexes in ((0, 1), (2, 3), (4, 5), (6,), (7, 8, 10, 11), (9,))]
        self.assertAlmostEqual(float(sum(piece[0] for piece in pieces)), float(np.sum(gradient * error)))

    def test_dominant_must_be_negative_and_unique(self) -> None:
        self.assertEqual(unique_dominant({"sinr": -5.0, "rate": -1.0, "progress": 0.2}), "sinr")
        self.assertIsNone(unique_dominant({"sinr": 2.0, "rate": 1.0}))
        self.assertIsNone(unique_dominant({"sinr": -1.0, "rate": -0.99}))

    def test_sign_agreement(self) -> None:
        self.assertEqual(sign_agreement([1.0, -2.0, 0.0], [0.5, -0.2, 0.0]), 1.0)
        self.assertEqual(sign_agreement([1.0], [-1.0]), 0.0)

    def test_confirm_scope_and_package_text(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        banned = ("torch.save", "Adam", "decide(")
        for path in PACKAGE.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in banned:
                self.assertNotIn(token, text, msg=f"{path.name} contains {token}")


if __name__ == "__main__":
    unittest.main()
