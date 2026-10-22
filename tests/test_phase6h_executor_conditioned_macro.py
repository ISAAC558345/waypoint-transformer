"""Phase-6H gate and isolation checks. Full pools are not searched here."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import actor_is_stateless, load_primary
from wt.benchmark_v2.phase6h_executor_conditioned_macro.labels import batch_j15_mappo
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import PRIMARY_SEED, assert_allowed, choose_gate
from wt.benchmark_v2.phase6f_system_bringup.episode import _contamination, _fingerprint


class Phase6HTests(unittest.TestCase):
    def test_reserved_ids_and_primary_seed_rule(self) -> None:
        for scenario in (320100, 320131, 321000, 321096, 310118):
            with self.assertRaises(RuntimeError):
                assert_allowed(scenario, split="label")
        assert_allowed(320000, split="label")
        assert_allowed(320016, split="closed")
        assert_allowed(320048, split="robust")
        self.assertEqual(PRIMARY_SEED, 0)

    def test_shadow_mappo_label_does_not_move_the_live_env(self) -> None:
        env = L3BBitQueueEnvV2()
        install_light_info(env)
        env.reset(seed=320000)
        actor = load_primary()
        self.assertTrue(actor_is_stateless(actor))
        option = make_option(env, np.asarray([-1, -1, -1, -1, -1, -1], dtype=np.int64))
        before = _fingerprint(env, option)
        _values, _elapsed, _first, error = batch_j15_mappo(env, actor, [option.association], 1)
        self.assertEqual(_contamination(before, _fingerprint(env, option)), 0)
        self.assertLessEqual(error, float(env.tol_bits))
        self.assertEqual(int(env.t), 0)
        self.assertFalse(actor.training)

    def test_gate_requires_both_primary_comparisons(self) -> None:
        only_rate = choose_gate(failures=0, seed_cv=0.0, c2_c0_pass=True, c2_c1_pass=False, label_shift_material=True, alignment_improved=True)
        self.assertEqual(only_rate["gate"], "ALIGN-C")
        no_gain = choose_gate(failures=0, seed_cv=0.0, c2_c0_pass=False, c2_c1_pass=False, label_shift_material=True, alignment_improved=True)
        self.assertEqual(no_gain["gate"], "ALIGN-B")
        both = choose_gate(failures=0, seed_cv=0.0, c2_c0_pass=True, c2_c1_pass=True, label_shift_material=False, alignment_improved=False)
        self.assertEqual(both["gate"], "ALIGN-A")
        unstable = choose_gate(failures=0, seed_cv=0.8, c2_c0_pass=True, c2_c1_pass=True, label_shift_material=True, alignment_improved=True)
        self.assertEqual(unstable["gate"], "ALIGN-D")

    def test_package_does_not_train_or_call_decide(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6h_executor_conditioned_macro"
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("decide(", text)
            self.assertNotIn("loss.backward", text)
            self.assertNotIn("planner_ready = True", text)


if __name__ == "__main__":
    unittest.main()
