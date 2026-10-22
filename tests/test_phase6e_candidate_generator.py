"""Phase-6E checks. They do not open the reserved confirmation pool."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6e.generators import build_pools, proposal_scores
from wt.benchmark_v2.phase6e.protocol import (
    assert_allowed,
    choose_gate,
    k4_sufficient,
    select_rule,
)
from wt.benchmark_v2.phase6e.value import batch_j15


class Phase6ECandidateGeneratorTest(unittest.TestCase):
    def test_reserved_and_wrong_split_are_refused(self) -> None:
        for scenario in (320100, 320131, 310118):
            with self.assertRaises(RuntimeError):
                assert_allowed(scenario, split="design")
        with self.assertRaises(RuntimeError):
            assert_allowed(320048, split="design")
        with self.assertRaises(RuntimeError):
            assert_allowed(320000, split="confirm")
        assert_allowed(320000, split="design")
        assert_allowed(320063, split="confirm")

    def test_scores_ignore_future_arrivals(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=320000)
        from wt.benchmark_v2.phase6e.generators import feature_matrices

        before = proposal_scores(feature_matrices(env))
        env.arrivals = lambda: np.ones(env.config.num_buoys, dtype=np.float64) * 10.0
        after = proposal_scores(feature_matrices(env))
        for name in before:
            self.assertTrue(np.allclose(before[name], after[name]))

    def test_current_shadow_drops_future_arrivals(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=320000)
        assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
        assignment[0] = 6
        values, _elapsed, error = batch_j15(env, [assignment], 15, blind_interference=False)
        self.assertLessEqual(float(values[( -1, -1, -1, -1, -1, 6) if False else tuple(int(v) for v in assignment)]), 1.0)
        self.assertLessEqual(error, env.tol_bits)
        self.assertEqual(int(env.t), 0)

    def test_idle_does_not_consume_a_buoy_slot(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=320000)
        pools, packed, illegal = build_pools(env, (("G1_RATE", 3), ("G1_RATE", 5)))
        self.assertEqual(illegal, 0)
        top3 = packed["tops"][("G1_RATE", 3)]
        top5 = packed["tops"][("G1_RATE", 5)]
        self.assertEqual(int(top3.shape[1]), 3)
        self.assertTrue(np.array_equal(top3, top5[:, :3]))
        self.assertTrue(np.all(top5 >= 0))
        identities = {tuple(int(item) for item in row) for row in pools[("G1_RATE", 3)]}
        self.assertIn(tuple([-1] * env.config.num_uavs), identities)

    def test_gate_priority(self) -> None:
        failed = choose_gate(conservation_ok=False, illegal_assignments=0, passing_rules=[{"k": 3}], confirmation_pass=True)
        self.assertEqual(failed["gate"], "CAND-E")
        self.assertFalse(failed["planner_ready"])
        self.assertFalse(failed["train_hgnn"])
        none = choose_gate(conservation_ok=True, illegal_assignments=0, passing_rules=[], confirmation_pass=False)
        self.assertEqual(none["gate"], "CAND-C")
        generalize = choose_gate(conservation_ok=True, illegal_assignments=0, passing_rules=[{"k": 4}], confirmation_pass=False)
        self.assertEqual(generalize["gate"], "CAND-B")
        top5 = choose_gate(conservation_ok=True, illegal_assignments=0, passing_rules=[{"k": 5}], confirmation_pass=True)
        self.assertEqual(top5["gate"], "CAND-D")
        compact = choose_gate(conservation_ok=True, illegal_assignments=0, passing_rules=[{"k": 4}, {"k": 5}], confirmation_pass=True)
        self.assertEqual(compact["gate"], "CAND-A")

    def test_tie_prefers_the_simpler_rule(self) -> None:
        def row(name, k, simplicity):
            return {
                "generator": name,
                "k": k,
                "simplicity": simplicity,
                "full_assignment_coverage": 0.95,
                "mean_nregret": 0.01,
                "state_pass_fraction": 1.0,
                "p95_nregret": 0.02,
                "mean_pool_size": 100.0,
                "mean_runtime_s": 1.0,
                "illegal_assignments": 0,
            }
        chosen = select_rule([row("G6_REACHABLE_QUEUE_RATE", 4, 5), row("G1_RATE", 4, 0)])
        self.assertEqual(chosen["selected_generator"], "G1_RATE")
        self.assertEqual(chosen["selected_k"], 4)
        self.assertTrue(chosen["candidate_generator_frozen"])
        self.assertFalse(
            k4_sufficient(
                {"mean_nregret": 0.05, "full_assignment_coverage": 0.90, "mean_pool_size": 100.0},
                {"mean_nregret": 0.0, "full_assignment_coverage": 1.0, "mean_pool_size": 100.0},
            )
        )

    def test_package_does_not_call_decide(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6e"
        for path in root.glob("*.py"):
            self.assertNotIn("decide(", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
