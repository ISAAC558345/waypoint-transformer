"""Original residual scale order. Alpha multiplies tanh(pre) before the unit clip."""
from __future__ import annotations

import math
import unittest

import numpy as np

from wt.benchmark_v2.phase6in_residual_scale.action import heading_delta_deg, residual_command, scale_limits
from wt.lowlevel_v2.contracts import clip_unit_step
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL


class Phase6INResidualScaleTests(unittest.TestCase):
    def test_alpha_is_applied_before_the_clip(self) -> None:
        self.assertEqual(float(ALPHA_RESIDUAL), 0.25)
        base = np.array([[1.0, 0.0]])
        pre = np.array([[-20.0, 20.0]])
        command, delta = residual_command(base, pre, 0.50)
        self.assertTrue(np.allclose(delta, [[-0.50, 0.50]], atol=1.0e-6))
        expected = clip_unit_step(base + delta)
        self.assertTrue(np.allclose(command, expected))
        self.assertAlmostEqual(float(np.linalg.norm(command)), math.hypot(0.5, 0.5), places=6)
        self.assertAlmostEqual(float(heading_delta_deg(command, base)[0]), 45.0, places=3)
        limits = scale_limits(0.25)
        self.assertAlmostEqual(limits["max_heading_deg"], math.degrees(math.atan2(0.25, 0.75)), places=6)
        self.assertAlmostEqual(limits["min_speed_fraction"], 0.75, places=6)
        self.assertFalse(limits["can_stop"])
        self.assertFalse(limits["can_reverse"])

    def test_idle_base_keeps_the_cartesian_residual(self) -> None:
        command, delta = residual_command(np.zeros((1, 2)), np.array([[20.0, 0.0]]), 0.35)
        self.assertTrue(np.allclose(delta, [[0.35, 0.0]], atol=1.0e-6))
        self.assertTrue(np.allclose(command, [[0.35, 0.0]], atol=1.0e-6))

    def test_larger_scale_uses_the_same_addition_order(self) -> None:
        base = np.array([[1.0, 0.0], [0.0, 1.0]])
        pre = np.array([[0.2, -0.4], [1.5, 0.3]])
        small, small_delta = residual_command(base, pre, 0.25)
        large, large_delta = residual_command(base, pre, 0.75)
        self.assertTrue(np.allclose(large_delta, 3.0 * small_delta))
        self.assertTrue(np.allclose(small, clip_unit_step(base + 0.25 * np.tanh(pre))))
        self.assertTrue(np.allclose(large, clip_unit_step(base + 0.75 * np.tanh(pre))))
        self.assertLessEqual(float(np.max(np.linalg.norm(large, axis=-1))), 1.0 + 1.0e-8)


if __name__ == "__main__":
    unittest.main()
