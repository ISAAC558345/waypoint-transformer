"""Ledger conservation, restore, and log-invariance checks. No training."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import (
    AdmissionAgeLedger,
    AgeBucket,
    LedgerError,
    snap_vector,
)
from wt.benchmark_v2.phase6i_urgency_stage_a.protocol import (
    AGE_ORIGIN_ARRIVAL,
    AGE_ORIGIN_RESET,
    smoke_scenario_ids,
)
from wt.benchmark_v2.phase6i_urgency_stage_a.runner import (
    branch_env,
    communication_at_service,
    fly_action,
    run_episode,
    same_outcome,
)
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.lowlevel_v2.critic_state import expected_critic_dim
from wt.lowlevel_v2.observation import TASK_DIM
from wt.lowlevel_v2.residual_policy import ResidualActor


def _zeros(count: int) -> np.ndarray:
    return np.zeros(count, dtype=np.float64)


class LedgerRuleTest(unittest.TestCase):
    def test_reset_stock_is_age_zero_with_reset_origin(self):
        env = L3BBitQueueEnvV2()
        env.reset(seed=int(smoke_scenario_ids()[0]))
        ledger = AdmissionAgeLedger(int(env.config.num_buoys))
        ledger.reset_from_queue(env.q_bits)
        gap = ledger.assert_matches(env.q_bits, float(env.tol_bits))
        self.assertLessEqual(float(np.max(np.abs(gap))), 1.0)
        view = ledger.view(0, env.q_bits, float(env.tol_bits))
        for buoy in range(int(env.config.num_buoys)):
            if float(env.q_bits[buoy]) > 0.0:
                self.assertEqual(ledger.buckets[buoy][0].age_origin, AGE_ORIGIN_RESET)
                self.assertEqual(ledger.buckets[buoy][0].admission_step, 0)
                self.assertEqual(float(view["oldest_wait_age"][buoy]), 0.0)
                self.assertEqual(int(view["urgency_valid"][buoy]), 1)
                self.assertEqual(float(view["urgency"][buoy]), 0.0)
            else:
                self.assertEqual(ledger.buckets[buoy], [])
                self.assertEqual(int(view["urgency_valid"][buoy]), 0)
        self.assertEqual(float(env.q_bits[0]), 4.0e6)
        self.assertEqual(float(env.q_bits[3]), 1.0e6)
        self.assertEqual(float(env.q_bits[6]), 0.0)

    def test_service_drains_oldest_bucket_before_the_new_arrival(self):
        ledger = AdmissionAgeLedger(1)
        ledger.buckets[0] = [AgeBucket(0, AGE_ORIGIN_RESET, 100.0), AgeBucket(2, AGE_ORIGIN_ARRIVAL, 50.0)]
        ledger.apply_transition(
            served_bits=np.asarray([120.0]),
            generated_arrival_bits=np.asarray([40.0]),
            admitted_arrival_bits=np.asarray([40.0]),
            dropped_bits=_zeros(1),
            snap_bits=_zeros(1),
            admission_step=4,
            tol_bits=1.0,
        )
        self.assertEqual(len(ledger.buckets[0]), 2)
        self.assertEqual(ledger.buckets[0][0].admission_step, 2)
        self.assertAlmostEqual(ledger.buckets[0][0].bits_remaining, 30.0)
        self.assertEqual(ledger.buckets[0][1].admission_step, 4)
        self.assertEqual(ledger.buckets[0][1].age_origin, AGE_ORIGIN_ARRIVAL)
        self.assertAlmostEqual(ledger.buckets[0][1].bits_remaining, 40.0)

    def test_drop_reduces_only_the_new_bucket(self):
        ledger = AdmissionAgeLedger(1)
        ledger.buckets[0] = [AgeBucket(0, AGE_ORIGIN_RESET, 100.0)]
        ledger.apply_transition(
            served_bits=np.asarray([10.0]),
            generated_arrival_bits=np.asarray([40.0]),
            admitted_arrival_bits=np.asarray([25.0]),
            dropped_bits=np.asarray([15.0]),
            snap_bits=_zeros(1),
            admission_step=6,
            tol_bits=1.0,
        )
        self.assertAlmostEqual(ledger.buckets[0][0].bits_remaining, 90.0)
        self.assertEqual(ledger.buckets[0][0].age_origin, AGE_ORIGIN_RESET)
        self.assertAlmostEqual(ledger.buckets[0][1].bits_remaining, 25.0)
        self.assertEqual(ledger.buckets[0][1].age_origin, AGE_ORIGIN_ARRIVAL)

    def test_snap_removes_the_same_residual_the_environment_clears(self):
        ledger = AdmissionAgeLedger(1)
        ledger.buckets[0] = [AgeBucket(0, AGE_ORIGIN_RESET, 0.4)]
        ledger.apply_transition(
            served_bits=_zeros(1),
            generated_arrival_bits=_zeros(1),
            admitted_arrival_bits=_zeros(1),
            dropped_bits=_zeros(1),
            snap_bits=np.asarray([0.4]),
            admission_step=1,
            tol_bits=1.0,
        )
        self.assertEqual(ledger.buckets[0], [])
        ledger.assert_matches(_zeros(1), 1.0)

    def test_empty_queue_and_age_zero_queue_use_different_validity(self):
        ledger = AdmissionAgeLedger(2)
        ledger.buckets[1] = [AgeBucket(2, AGE_ORIGIN_RESET, 5.0)]
        view = ledger.view(2, np.asarray([0.0, 5.0]), 1.0)
        self.assertEqual(int(view["urgency_valid"][0]), 0)
        self.assertEqual(float(view["urgency"][0]), 0.0)
        self.assertEqual(int(view["urgency_valid"][1]), 1)
        self.assertEqual(float(view["oldest_wait_age"][1]), 0.0)
        self.assertEqual(float(view["urgency"][1]), 0.0)

    def test_unserved_age_increases_when_the_view_is_read(self):
        ledger = AdmissionAgeLedger(1)
        ledger.buckets[0] = [AgeBucket(0, AGE_ORIGIN_RESET, 80.0)]
        before = ledger.buckets[0][0].bits_remaining
        self.assertEqual(float(ledger.view(0, np.asarray([80.0]), 1.0)["oldest_wait_age"][0]), 0.0)
        self.assertEqual(float(ledger.view(3, np.asarray([80.0]), 1.0)["oldest_wait_age"][0]), 3.0)
        self.assertEqual(ledger.buckets[0][0].bits_remaining, before)

    def test_capture_restore_lets_branches_diverge(self):
        ledger = AdmissionAgeLedger(1)
        ledger.buckets[0] = [AgeBucket(0, AGE_ORIGIN_RESET, 4.0e6)]
        snap = ledger.capture()
        other = AdmissionAgeLedger(1)
        other.restore(snap)
        other.apply_transition(np.asarray([1.0e6]), _zeros(1), _zeros(1), _zeros(1), _zeros(1), 1, 1.0)
        self.assertAlmostEqual(ledger.bits_sum()[0], 4.0e6)
        self.assertAlmostEqual(other.bits_sum()[0], 3.0e6)

    def test_snap_vector_uses_the_cumulative_counter_already_updated_by_the_step(self):
        env = type("Env", (), {})()
        env.config = type("Config", (), {"num_buoys": 2})()
        env.cumulative_snap_bits = np.asarray([0.25, 0.0])
        self.assertAlmostEqual(float(snap_vector(env, _zeros(2), {})[0]), 0.25)
        bare = type("Env", (), {})()
        bare.config = env.config
        with self.assertRaises(LedgerError):
            snap_vector(bare, _zeros(2), {})

    def test_real_arrival_is_not_served_in_its_admission_step(self):
        scenario_id = int(smoke_scenario_ids()[0])
        env = L3BBitQueueEnvV2()
        install_light_info(env)
        env.reset(seed=scenario_id)
        ledger = AdmissionAgeLedger(int(env.config.num_buoys))
        ledger.reset_from_queue(env.q_bits)
        assignment = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
        option_hold = 15
        while int(env.t) < 4:
            from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option

            option = make_option(env, assignment)
            option.remaining_hold = option_hold
            action = fly_action(env, option, np.zeros((6, 2)))
            before = np.asarray(env.cumulative_snap_bits, dtype=np.float64).copy()
            t_before = int(env.t)
            info = env.step(action.executed, assignment)
            ledger.apply_transition(
                info["served_bits"],
                info["generated_arrival_bits"],
                info["admitted_arrival_bits"],
                info["dropped_bits_step"],
                snap_vector(env, before, info),
                t_before,
                float(env.tol_bits),
            )
            ledger.assert_matches(env.q_bits, float(env.tol_bits))
            option_hold -= 1
        before = np.asarray(env.cumulative_snap_bits, dtype=np.float64).copy()
        from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option

        option = make_option(env, assignment)
        action = fly_action(env, option, np.zeros((6, 2)))
        info = env.step(action.executed, assignment)
        self.assertEqual(float(info["served_bits"][6]), 0.0)
        self.assertEqual(float(info["admitted_arrival_bits"][6]), 1.0e6)
        ledger.apply_transition(
            info["served_bits"],
            info["generated_arrival_bits"],
            info["admitted_arrival_bits"],
            info["dropped_bits_step"],
            snap_vector(env, before, info),
            4,
            float(env.tol_bits),
        )
        ledger.assert_matches(env.q_bits, float(env.tol_bits))
        self.assertEqual(len(ledger.buckets[6]), 1)
        self.assertEqual(ledger.buckets[6][0].age_origin, AGE_ORIGIN_ARRIVAL)
        self.assertEqual(ledger.buckets[6][0].admission_step, 4)
        self.assertAlmostEqual(ledger.buckets[6][0].bits_remaining, 1.0e6)

    def test_restored_env_branch_keeps_its_own_ledger(self):
        scenario_id = int(smoke_scenario_ids()[0])
        env = L3BBitQueueEnvV2()
        install_light_info(env)
        env.reset(seed=scenario_id)
        ledger = AdmissionAgeLedger(int(env.config.num_buoys))
        ledger.reset_from_queue(env.q_bits)
        clone = branch_env(env, scenario_id)
        other = AdmissionAgeLedger(int(env.config.num_buoys))
        other.restore(ledger.capture())
        other.assert_matches(clone.q_bits, float(clone.tol_bits))
        other.apply_transition(np.asarray([1.0e6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), _zeros(12), _zeros(12), _zeros(12), _zeros(12), 1, 1.0)
        self.assertAlmostEqual(ledger.bits_sum()[0], 4.0e6)
        self.assertAlmostEqual(other.bits_sum()[0], 3.0e6)
        other.assert_matches(np.asarray(clone.q_bits) - np.asarray([1.0e6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1.0)


class LogInvariantTest(unittest.TestCase):
    def test_recording_rows_does_not_change_the_episode_and_repeat_matches(self):
        scenario_id = int(smoke_scenario_ids()[0])
        assignment = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)

        def decision(_env, _position_prev, _predecessor):
            return assignment, {"candidate_pool_hash": "unit", "candidate_index": 0, "scheme_c_score": 0.0}

        def movement(env, option, last):
            return fly_action(env, option, last)

        logged = run_episode(scenario_id, movement=movement, decision=decision, record=True, max_steps=6)
        silent = run_episode(scenario_id, movement=movement, decision=decision, record=False, max_steps=6)
        repeat = run_episode(scenario_id, movement=movement, decision=decision, record=True, max_steps=6)
        self.assertTrue(same_outcome(logged, silent))
        self.assertTrue(same_outcome(logged, repeat))
        self.assertLessEqual(logged.max_ledger_mismatch_bits, 1.0)
        self.assertGreater(len(logged.rows), 0)
        self.assertEqual(silent.rows, [])
        buoy0 = [row for row in logged.rows if row["record_kind"] == "buoy" and row["index"] == 0 and row["time_step"] == 0][0]
        self.assertEqual(float(buoy0["oldest_wait_age"]), 0.0)
        self.assertEqual(int(buoy0["urgency_valid"]), 1)

    def test_service_sinr_matches_the_pre_service_queue_and_the_queue_is_restored(self):
        scenario_id = int(smoke_scenario_ids()[0])
        env = L3BBitQueueEnvV2()
        install_light_info(env)
        env.reset(seed=scenario_id)
        from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option

        assignment = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
        option = make_option(env, assignment)
        action = fly_action(env, option, np.zeros((6, 2)))
        queue_before = np.asarray(env.q_bits, dtype=np.float64).copy()
        info = env.step(action.executed, assignment)
        queue_after = np.asarray(env.queue, dtype=np.float64).copy()
        comm = communication_at_service(env, queue_before)
        self.assertLess(float(np.max(np.abs(np.asarray(info["sinr"]) - comm["sinr"]))), 1.0e-9)
        self.assertLess(float(np.max(np.abs(np.asarray(info["rate_bps"]) - comm["rate"]))), 1.0e-6)
        self.assertTrue(np.array_equal(env.queue, queue_after))

    def test_reserved_scenario_is_refused(self):
        scenario_id = int(tuple(RESERVED_TEST)[0])

        def decision(_env, _position_prev, _predecessor):
            return np.asarray([-1, -1, -1, -1, -1, -1]), {"candidate_pool_hash": "", "candidate_index": "", "scheme_c_score": ""}

        with self.assertRaises(RuntimeError):
            run_episode(scenario_id, movement=fly_action, decision=decision, record=False, max_steps=1)

    def test_sources_do_not_read_future_arrivals_or_change_formal_modules(self):
        root = Path(__file__).resolve().parents[1]
        package = root / "src" / "wt" / "benchmark_v2" / "phase6i_urgency_stage_a"
        text = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
        self.assertNotIn("remaining_arrivals", text)
        self.assertNotIn("decide(", text)
        for relative in (
            "src/wt/benchmark_v2/bit_queue_v2.py",
            "src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py",
            "src/wt/lowlevel_v2/safety_projector.py",
            "src/wt/benchmark_v2/phase6ib_highlevel_dataset/hgnn_revision.py",
        ):
            formal = (root / relative).read_text(encoding="utf-8")
            self.assertNotIn("phase6i_urgency_stage_a", formal)
        actor = ResidualActor()
        self.assertEqual(int(actor.body[0].in_features), 34)
        self.assertEqual(int(TASK_DIM), 8)
        self.assertEqual(expected_critic_dim(6, 12), 188)


if __name__ == "__main__":
    unittest.main()
