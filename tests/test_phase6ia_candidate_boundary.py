"""Phase-6I-A gate rules. Full pools are not searched here."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6e.generators import build_pools
from wt.benchmark_v2.phase6ia_candidate_boundary.protocol import PRIMARY_SEED, assert_allowed, choose_gate, evaluate_flags


def _stats(**overrides) -> dict:
    base = {
        "design_mean_ngain_6_5": 0.0,
        "design_p95_ngain_6_5": 0.0,
        "confirm_mean_ngain_6_5": 0.0,
        "confirm_p95_ngain_6_5": 0.0,
        "design_ngain_6_5_ci_low": -0.1,
        "confirm_ngain_6_5_ci_low": -0.1,
        "design_mean_ngain_7_6": 0.0,
        "design_p95_ngain_7_6": 0.0,
        "confirm_mean_ngain_7_6": 0.0,
        "confirm_p95_ngain_7_6": 0.0,
        "design_ngain_7_6_ci_low": -0.1,
        "confirm_ngain_7_6_ci_low": -0.1,
        "design_coverage_k5": 1.0,
        "confirm_coverage_k5": 1.0,
        "design_rank7_fraction": 0.0,
        "confirm_rank7_fraction": 0.0,
        "closed_mean_bits": -1.0,
        "closed_ci_low_bits": -2.0,
    }
    base.update(overrides)
    return base


class Phase6IATests(unittest.TestCase):
    def test_splits_are_fixed_before_evaluation(self) -> None:
        assert_allowed(320000, split="design")
        assert_allowed(320032, split="confirm")
        assert_allowed(320060, split="confirm")
        assert_allowed(320039, split="sanity")
        with self.assertRaises(RuntimeError):
            assert_allowed(320033, split="confirm")
        for scenario in (320100, 320131, 321000, 310118, 320031):
            with self.assertRaises(RuntimeError):
                assert_allowed(scenario, split="sanity" if scenario < 320100 else "design")
        with self.assertRaises(RuntimeError):
            assert_allowed(320016, split="confirm")
        self.assertEqual(PRIMARY_SEED, 0)

    def test_nested_g1_pools_keep_the_same_legality(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=320000)
        pools, _packed, illegal = build_pools(env, (("G1_RATE", 5), ("G1_RATE", 6), ("G1_RATE", 7)))
        self.assertEqual(int(illegal), 0)
        keys = [{tuple(int(item) for item in assignment) for assignment in pools[("G1_RATE", k)]} for k in (5, 6, 7)]
        self.assertTrue(keys[0] < keys[1] < keys[2])
        self.assertEqual(int(env.t), 0)

    def test_gate_priority(self) -> None:
        sufficient = evaluate_flags(_stats())
        self.assertEqual(choose_gate(sufficient, failures=0)["gate"], "BOUND-A")
        self.assertEqual(choose_gate(sufficient, failures=1)["gate"], "BOUND-E")
        k6 = evaluate_flags(
            _stats(
                design_mean_ngain_6_5=0.04,
                confirm_mean_ngain_6_5=0.04,
                design_ngain_6_5_ci_low=0.01,
                confirm_ngain_6_5_ci_low=0.01,
                design_p95_ngain_6_5=0.08,
                confirm_p95_ngain_6_5=0.08,
                design_coverage_k5=0.40,
                confirm_coverage_k5=0.40,
            )
        )
        self.assertEqual(choose_gate(k6, failures=0)["gate"], "BOUND-B")
        k7 = evaluate_flags(
            _stats(
                design_mean_ngain_7_6=0.05,
                confirm_mean_ngain_7_6=0.05,
                design_ngain_7_6_ci_low=0.01,
                confirm_ngain_7_6_ci_low=0.01,
                design_p95_ngain_7_6=0.08,
                confirm_p95_ngain_7_6=0.08,
                design_coverage_k5=0.40,
                confirm_coverage_k5=0.40,
            )
        )
        self.assertEqual(choose_gate(k7, failures=0)["gate"], "BOUND-C")
        heavy = evaluate_flags(
            _stats(
                design_mean_ngain_7_6=0.05,
                confirm_mean_ngain_7_6=0.05,
                design_ngain_7_6_ci_low=0.01,
                confirm_ngain_7_6_ci_low=0.01,
                design_p95_ngain_7_6=0.08,
                confirm_p95_ngain_7_6=0.08,
                design_coverage_k5=0.40,
                confirm_coverage_k5=0.40,
                confirm_rank7_fraction=0.70,
            )
        )
        self.assertEqual(choose_gate(heavy, failures=0)["gate"], "BOUND-D")

    def test_package_does_not_train_or_call_decide(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6ia_candidate_boundary"
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("decide(", text)
            self.assertNotIn("loss.backward", text)
            self.assertNotIn("planner_ready = True", text)


if __name__ == "__main__":
    unittest.main()
