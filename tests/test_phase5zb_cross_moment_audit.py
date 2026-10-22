"""Cross-moment diagnostics stay untrained and swap-sensitive."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.phase5zb_cross_moment_audit.metrics import choose_gate
from wt.lowlevel_v2.phase5zb_cross_moment_audit.moments import cross_moment, permute_buoys, permute_uavs
from wt.lowlevel_v2.phase5zb_cross_moment_audit.protocol import assert_consumed


class Phase5ZBCrossMomentAuditTest(unittest.TestCase):
    def test_swap_changes_cross_moment_but_not_buoy_sum(self) -> None:
        h_u = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        h_b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        h_edge = np.zeros((2, 2, 2), dtype=np.float64)
        assignments = np.array([[0, 1], [1, 0]], dtype=np.int64)
        first = cross_moment(h_u, h_b, assignments[:1])
        second = cross_moment(h_u, h_b, assignments[1:])
        self.assertFalse(np.allclose(first["moment"], second["moment"]))
        self.assertTrue(np.allclose(h_b[assignments[0]].sum(axis=0), h_b[assignments[1]].sum(axis=0)))
        del h_edge

    def test_global_relabel_is_invariant(self) -> None:
        rng = np.random.default_rng(0)
        h_u = rng.normal(size=(6, 8))
        h_b = rng.normal(size=(12, 8))
        h_edge = rng.normal(size=(6, 12, 8))
        assignments = np.array([[2, 5, -1, 4, -1, -1], [5, 2, -1, 4, -1, -1]], dtype=np.int64)
        base = cross_moment(h_u, h_b, assignments)["representation"]
        h_u2, h_edge2, assign_u = permute_uavs(h_u, h_edge, assignments, np.array([1, 0, 2, 3, 4, 5]))
        self.assertLessEqual(float(np.max(np.abs(base - cross_moment(h_u2, h_b, assign_u)["representation"]))), 1.0e-5)
        h_b2, _edge, assign_b = permute_buoys(h_b, h_edge2, assignments, np.arange(12)[::-1])
        self.assertLessEqual(float(np.max(np.abs(base - cross_moment(h_u, h_b2, assign_b)["representation"]))), 1.0e-5)

    def test_gate_priority(self) -> None:
        self.assertEqual(choose_gate(permutation_ok=False, cm_a=True, cm_all_passes=True, helped=True, no_reduction=False), "CM-D")
        self.assertEqual(choose_gate(permutation_ok=True, cm_a=True, cm_all_passes=False, helped=True, no_reduction=False), "CM-A")
        self.assertEqual(choose_gate(permutation_ok=True, cm_a=False, cm_all_passes=True, helped=True, no_reduction=False), "CM-B")
        self.assertEqual(choose_gate(permutation_ok=True, cm_a=False, cm_all_passes=False, helped=True, no_reduction=False), "CM-B")
        self.assertEqual(choose_gate(permutation_ok=True, cm_a=False, cm_all_passes=False, helped=False, no_reduction=True), "CM-C")

    def test_holdout_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230100)

    def test_package_does_not_train(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "lowlevel_v2" / "phase5zb_cross_moment_audit"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("torch.optim", text)
        self.assertNotIn(".backward(", text)
        self.assertNotIn("decide(", text)


if __name__ == "__main__":
    unittest.main()
