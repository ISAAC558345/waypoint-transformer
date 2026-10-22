"""Deterministic checks for residual limits and direct velocity commands."""
from __future__ import annotations

import math
import unittest

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option
from wt.benchmark_v2.phase6il_direct_action.action import direct_command, residual_command, residual_limits
from wt.benchmark_v2.phase6il_direct_action.worker import execute_command
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import minimum_pairwise_distance


class Phase6ILActionTests(unittest.TestCase):
    def test_residual_cannot_reverse_or_stop(self) -> None:
        limits = residual_limits()
        self.assertAlmostEqual(limits["max_heading_deg"], math.degrees(math.atan2(0.25, 0.75)), places=6)
        self.assertAlmostEqual(limits["min_speed_fraction"], 0.75, places=6)
        self.assertFalse(limits["can_stop"])
        self.assertFalse(limits["can_reverse"])
        env = L3BBitQueueEnvV2()
        env.reset(seed=321000)
        association = np.zeros(env.config.num_uavs, dtype=np.int64)
        env.uav_positions[:] = env.buoy_positions[0] + np.array([-40.0, 0.0])
        pre = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
        pre[:, 0] = -20.0
        pre[:, 1] = 20.0
        command, _delta = residual_command(env, association, pre)
        base = fly_toward(env, association)
        for uav in range(env.config.num_uavs):
            cosine = float(np.dot(command[uav], base[uav]) / np.linalg.norm(command[uav]))
            self.assertGreater(cosine, math.cos(math.radians(limits["max_heading_deg"] + 1.0e-6)))
            self.assertGreater(float(np.linalg.norm(command[uav])), 0.74)

    def test_direct_command_covers_direction_and_speed(self) -> None:
        hover = direct_command(np.zeros((6, 2)))
        self.assertTrue(np.allclose(hover, 0.0))
        east = direct_command(np.array([[20.0, 0.0], [0.0, 20.0], [-20.0, 0.0], [0.0, -20.0], [20.0, 20.0], [-20.0, -20.0]]))
        self.assertTrue(np.all(np.linalg.norm(east, axis=-1) <= 1.0 + 1.0e-8))
        self.assertGreater(east[0, 0], 0.99)
        self.assertGreater(east[1, 1], 0.99)
        self.assertLess(east[2, 0], -0.99)
        self.assertLess(east[3, 1], -0.99)
        diagonal = east[4]
        self.assertAlmostEqual(float(np.linalg.norm(diagonal)), 1.0, places=6)
        self.assertAlmostEqual(diagonal[0], diagonal[1], places=6)

    def test_illegal_preactivation_is_rejected_or_held(self) -> None:
        with self.assertRaises(ValueError):
            direct_command(np.zeros((6, 3)))
        held = direct_command(np.array([[0.0, np.nan], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))
        self.assertTrue(np.allclose(held, 0.0))

    def test_world_frame_step_matches_the_environment_axis(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=321000)
        env.uav_positions[:] = np.array([[20.0 + (i % 3) * 40.0, 20.0 + (i // 3) * 30.0] for i in range(env.config.num_uavs)])
        association = np.full(env.config.num_uavs, -1, dtype=np.int64)
        option = make_option(env, association)
        command = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
        command[0] = (1.0, 0.0)
        command[1] = (0.0, 1.0)
        executed, safety = execute_command(env, option, command)
        self.assertFalse(safety["failure"])
        before = env.uav_positions.copy()
        info = env.step(executed, association)
        delta = env.uav_positions - before
        step = float(env.config.step_length_m)
        self.assertAlmostEqual(float(delta[0, 0]), step, places=5)
        self.assertAlmostEqual(float(delta[0, 1]), 0.0, places=5)
        self.assertAlmostEqual(float(delta[1, 1]), step, places=5)
        self.assertTrue(np.all(np.linalg.norm(delta, axis=-1) <= step + 1.0e-6))
        self.assertLessEqual(float(np.max(np.abs(env.conservation_error_bits()))), float(env.tol_bits))
        self.assertAlmostEqual(float(env.payload_bits), float(np.sum(info["served_bits"])), places=6)
        self.assertTrue(np.array_equal(env.assignment, association))

    def test_projector_separates_and_keeps_the_service_envelope(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=321000)
        env.uav_positions[:] = np.array([[20.0 + i * 25.0, 15.0] for i in range(env.config.num_uavs)])
        env.uav_positions[0] = np.array([40.0, 40.0])
        env.uav_positions[1] = np.array([40.2, 40.0])
        association = np.full(env.config.num_uavs, -1, dtype=np.int64)
        option = make_option(env, association)
        executed, safety = execute_command(env, option, np.zeros((env.config.num_uavs, 2)))
        self.assertFalse(safety["failure"])
        self.assertGreater(safety["intervention_norm"], 1.0e-3)
        env.step(executed, association)
        self.assertGreaterEqual(minimum_pairwise_distance(env.uav_positions) + 1.0e-6, 1.0)

        env.reset(seed=321000)
        env.uav_positions[:] = np.array([[20.0 + i * 20.0, 70.0] for i in range(env.config.num_uavs)])
        buoy = env.buoy_positions[0].copy()
        env.uav_positions[0] = buoy + np.array([10.0, 0.0])
        association = np.full(env.config.num_uavs, -1, dtype=np.int64)
        association[0] = 0
        option = make_option(env, association)
        command = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
        command[0] = (1.0, 0.0)
        executed, safety = execute_command(env, option, command)
        self.assertFalse(safety["failure"])
        self.assertGreater(safety["intervention_norm"], 1.0e-3)
        nxt = env.uav_positions[0] + executed[0] * env.config.step_length_m
        self.assertLessEqual(float(np.linalg.norm(nxt - buoy)), 15.5 + 1.0e-6)


if __name__ == "__main__":
    unittest.main()
