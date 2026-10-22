"""Phase-10.3 gate tests. No training."""
from __future__ import annotations

import unittest

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.htw_phase100.rule_executor import RuleExecutorAdapter
from wt.htw_phase9.const import HOLD
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward

from wt.htw_phase102.replay import classify_rho
from wt.htw_phase103.gate import InRangeZeroMotionGate
from wt.htw_phase103.protocol import DESIGN_VAL_SEEDS, HIGH_CKPT, LOW_FINAL, sha256_bytes


class Phase103Tests(unittest.TestCase):
    def test_01_idle_does_not_read_last_buoy(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        assignment = np.full(6, -1, dtype=np.int64)
        option = TaskOption(assignment, 0, HOLD, np.ones(6, dtype=bool), np.ones(12, dtype=bool), assignment.copy())
        raw = np.ones((6, 2), dtype=np.float64)
        gated, flags = InRangeZeroMotionGate().apply(env, option, raw)
        np.testing.assert_array_equal(flags, np.zeros(6, dtype=bool))
        np.testing.assert_array_equal(gated, raw)

    def test_02_empty_queue_and_invalid_do_not_gate(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        env.queue[:] = 0.0
        env.uav_positions[:] = env.buoy_positions[0]
        assignment = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
        option = TaskOption(assignment, 0, HOLD, np.ones(6, dtype=bool), np.ones(12, dtype=bool), assignment.copy())
        raw = np.ones((6, 2))
        _g, flags = InRangeZeroMotionGate().apply(env, option, raw)
        self.assertFalse(bool(np.any(flags)))
        mismatch = assignment.copy()
        opt2 = TaskOption(mismatch, 0, HOLD, np.ones(6, dtype=bool), np.ones(12, dtype=bool), mismatch.copy())
        opt2.association = mismatch.copy()
        opt2.association[0] = 1
        with self.assertRaises(RuntimeError):
            InRangeZeroMotionGate().apply(env, opt2, raw)
        invalid = np.array([99, -1, -1, -1, -1, -1], dtype=np.int64)
        opt3 = TaskOption(invalid, 0, HOLD, np.ones(6, dtype=bool), np.ones(12, dtype=bool), invalid.copy())
        _g, flags3 = InRangeZeroMotionGate().apply(env, opt3, raw)
        self.assertFalse(flags3[0])

    def test_03_out_of_range_keeps_action(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        env.uav_positions[:] = (0.0, 0.0)
        env.queue[0] = 2.0
        assignment = np.array([0, -1, -1, -1, -1, -1], dtype=np.int64)
        option = TaskOption(assignment, 0, HOLD, np.ones(6, dtype=bool), np.ones(12, dtype=bool), assignment.copy())
        raw = np.array([[0.3, -0.4]] + [[0.0, 0.0]] * 5, dtype=np.float64)
        gated, flags = InRangeZeroMotionGate().apply(env, option, raw)
        self.assertFalse(flags[0])
        np.testing.assert_allclose(gated[0], raw[0])

    def test_04_in_range_nonempty_zeros_movement(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        env.uav_positions[0] = env.buoy_positions[0] + np.array([8.0, 0.0])
        env.queue[0] = 2.0
        assignment = np.array([0, -1, -1, -1, -1, -1], dtype=np.int64)
        option = TaskOption(assignment, 0, HOLD, np.ones(6, dtype=bool), np.ones(12, dtype=bool), assignment.copy())
        raw = fly_toward(env, assignment)
        self.assertGreater(float(np.linalg.norm(raw[0])), 0.0)
        gated, flags = InRangeZeroMotionGate().apply(env, option, raw)
        self.assertTrue(flags[0])
        np.testing.assert_array_equal(gated[0], (0.0, 0.0))
        self.assertTrue(env._in_range(0, 0))

    def test_05_gate_uses_pre_action_state(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        env.uav_positions[0] = env.buoy_positions[0]
        env.queue[0] = 1.0
        assignment = np.array([0, -1, -1, -1, -1, -1], dtype=np.int64)
        option = TaskOption(assignment, 0, HOLD, np.ones(6, dtype=bool), np.ones(12, dtype=bool), assignment.copy())
        flags = InRangeZeroMotionGate().flags(env, option)
        self.assertTrue(flags[0])
        env.step(np.ones((6, 2)), assignment)
        self.assertEqual(int(env.t), 1)

    def test_06_new_target_after_switch(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        env.uav_positions[0] = env.buoy_positions[0]
        env.queue[:] = 1.0
        a0 = np.array([0, -1, -1, -1, -1, -1], dtype=np.int64)
        a1 = np.array([6, -1, -1, -1, -1, -1], dtype=np.int64)
        opt0 = TaskOption(a0, 0, HOLD, np.ones(6, dtype=bool), np.ones(12, dtype=bool), a0.copy())
        opt1 = TaskOption(a1, 0, HOLD, np.ones(6, dtype=bool), np.ones(12, dtype=bool), a1.copy())
        self.assertTrue(InRangeZeroMotionGate().flags(env, opt0)[0])
        self.assertFalse(InRangeZeroMotionGate().flags(env, opt1)[0])

    def test_07_gate_off_matches_rule_executor(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        assignment = np.array([0, 1, 2, -1, 4, 5], dtype=np.int64)
        option = TaskOption(assignment, 0, HOLD, np.ones(6, dtype=bool), np.ones(12, dtype=bool), assignment.copy())
        raw = RuleExecutorAdapter().act(env, option)
        gated, flags = InRangeZeroMotionGate().apply(env, option, raw)
        ungated = raw.copy()
        ungated[flags] = 0.0
        np.testing.assert_array_equal(gated, ungated)

    def test_08_does_not_change_assignment_or_hold(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=8)
        assignment = np.array([0, -1, -1, -1, -1, -1], dtype=np.int64)
        option = TaskOption(assignment.copy(), int(info["time_step"]), 15, np.ones(6, dtype=bool), np.ones(12, dtype=bool), assignment.copy())
        InRangeZeroMotionGate().apply(env, option, np.zeros((6, 2)))
        np.testing.assert_array_equal(option.association, assignment)
        self.assertEqual(int(option.remaining_hold), 15)

    def test_09_checkpoints_exist_and_not_m5(self) -> None:
        self.assertTrue(HIGH_CKPT.is_file())
        self.assertTrue(LOW_FINAL.is_file())
        self.assertEqual(len(sha256_bytes(LOW_FINAL.read_bytes())), 64)
        self.assertEqual(DESIGN_VAL_SEEDS, tuple(range(8, 16)))

    def test_10_normal_completion_not_target_loss(self) -> None:
        self.assertIn("NORMAL_COMPLETION", classify_rho(0.99, 0.0, True, False, False, False, False, False))
        self.assertNotIn("TARGET_CHANGE", classify_rho(0.99, 0.0, True, False, False, False, False, False))


if __name__ == "__main__":
    unittest.main()
