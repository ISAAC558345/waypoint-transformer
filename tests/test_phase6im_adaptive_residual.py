"""Adaptive residual kinematics. The round-1 limits stay outside the official config."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option
from wt.benchmark_v2.phase6im_adaptive_residual.action import adaptive_command, heading_delta_deg
from wt.benchmark_v2.phase6im_adaptive_residual.protocol import MAX_TURN_DEG, OFFICIAL_CONFIG
from wt.benchmark_v2.phase6im_adaptive_residual.worker import apply_command, make_projector
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import minimum_pairwise_distance


def _full_speed_bias() -> float:
    return float(np.arctanh(0.999))


class Phase6IMAdaptiveResidualTests(unittest.TestCase):
    def test_turn_and_speed_stay_inside_the_round1_limits(self) -> None:
        self.assertFalse(OFFICIAL_CONFIG)
        base = np.array([[1.0, 0.0]])
        positive = adaptive_command(
            base,
            np.array([[80.0, 0.0]]),
            steer_center=0.0,
            steer_temperature=1.0,
            speed_bias=_full_speed_bias(),
            max_turn_deg=MAX_TURN_DEG,
        )
        angle = float(heading_delta_deg(positive, base)[0])
        self.assertAlmostEqual(angle, 45.0, places=3)
        self.assertGreater(float(np.linalg.norm(positive)), 0.99)
        self.assertLessEqual(float(np.linalg.norm(positive)), 1.0 + 1.0e-8)
        negative = adaptive_command(
            base,
            np.array([[-80.0, 0.0]]),
            steer_center=0.0,
            steer_temperature=1.0,
            speed_bias=_full_speed_bias(),
        )
        self.assertAlmostEqual(float(heading_delta_deg(negative, base)[0]), -45.0, places=3)
        stopped = adaptive_command(
            base,
            np.array([[0.0, -80.0]]),
            steer_center=0.0,
            steer_temperature=1.0,
            speed_bias=0.0,
        )
        self.assertLess(float(np.linalg.norm(stopped)), 1.0e-6)

    def test_zero_base_and_nonfinite_inputs_hover(self) -> None:
        hover = adaptive_command(
            np.zeros((2, 2)),
            np.array([[10.0, 10.0], [-10.0, 4.0]]),
            steer_center=0.0,
            steer_temperature=1.0,
            speed_bias=2.0,
        )
        self.assertTrue(np.allclose(hover, 0.0))
        invalid = adaptive_command(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, np.nan], [1.0, 0.0]]),
            steer_center=0.0,
            steer_temperature=1.0,
            speed_bias=0.0,
        )
        self.assertTrue(np.allclose(invalid, 0.0))
        with self.assertRaises(ValueError):
            adaptive_command(np.zeros((2, 3)), np.zeros((2, 3)), steer_center=0.0, steer_temperature=1.0, speed_bias=0.0)

    def test_environment_axis_accounting_and_projector(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=321000)
        env.uav_positions[:] = np.array([[20.0 + (i % 3) * 40.0, 20.0 + (i // 3) * 30.0] for i in range(6)])
        buoy = env.buoy_positions[0].copy()
        env.uav_positions[0] = buoy + np.array([-30.0, 0.0])
        association = np.full(6, -1, dtype=np.int64)
        association[0] = 0
        option = make_option(env, association)
        base = fly_toward(env, association)
        command = adaptive_command(base, np.array([[0.0, 80.0]] * 6), steer_center=0.0, steer_temperature=1.0, speed_bias=0.0)
        executed, report = apply_command(env, option, command)
        self.assertFalse(report["failure"])
        self.assertLessEqual(float(np.max(np.linalg.norm(executed, axis=-1))), 1.0 + 1.0e-8)
        before = env.uav_positions.copy()
        info = env.step(executed, association)
        delta = env.uav_positions - before
        self.assertAlmostEqual(float(np.linalg.norm(delta[0])), float(env.config.step_length_m), places=4)
        self.assertGreater(float(delta[0, 0]), 0.0)
        self.assertLessEqual(float(np.max(np.abs(env.conservation_error_bits()))), float(env.tol_bits))
        self.assertAlmostEqual(float(env.payload_bits), float(np.sum(info["served_bits"])), places=5)
        self.assertTrue(np.array_equal(env.assignment, association))

        env.reset(seed=321000)
        env.uav_positions[:] = np.array([[15.0 + i * 25.0, 10.0] for i in range(6)])
        env.uav_positions[0] = np.array([40.0, 40.0])
        env.uav_positions[1] = np.array([40.2, 40.0])
        association = np.full(6, -1, dtype=np.int64)
        option = make_option(env, association)
        executed, report = apply_command(env, option, np.zeros((6, 2)))
        self.assertFalse(report["failure"])
        self.assertGreater(report["intervention_norm"], 1.0e-3)
        self.assertEqual(float(make_projector().collision.d_safe), 1.0)
        env.step(executed, association)
        self.assertGreaterEqual(minimum_pairwise_distance(env.uav_positions) + 1.0e-6, 1.0)


if __name__ == "__main__":
    unittest.main()
