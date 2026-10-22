"""Phase-6I-B rules. Full J15 pools are not searched here."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6e.generators import build_pools
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _dense_groups
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import PRIMARY_SEED, assert_allowed, choose_gate, integrity_selected
from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import step_observation
from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_arrays


def _flags(**overrides) -> dict:
    base = {"lineage_failure": False, "leakage": False, "reproducibility_failure": False, "degenerate": False, "data_a": True}
    base.update(overrides)
    return base


class Phase6IBTests(unittest.TestCase):
    def test_new_splits_do_not_open_reserved_or_historical_ids(self) -> None:
        assert_allowed(322000, split="train")
        assert_allowed(322047, split="train")
        assert_allowed(322048, split="validation")
        assert_allowed(322063, split="validation")
        assert_allowed(322003, split="pilot")
        for scenario in (322064, 322079, 320100, 320131, 320000, 320063, 321000, 310118):
            with self.assertRaises(RuntimeError):
                assert_allowed(scenario, split="train")
        self.assertEqual(PRIMARY_SEED, 0)

    def test_integrity_sample_is_fixed(self) -> None:
        self.assertEqual(integrity_selected(322000, 0, 3), integrity_selected(322000, 0, 3))
        selected = sum(integrity_selected(322000, 0, item) for item in range(10000))
        self.assertGreater(selected, 400)
        self.assertLess(selected, 600)

    def test_dense_rank_keeps_one_bit_ties(self) -> None:
        keys = [(0, 1), (1, 0), (2, 3)]
        values = {(0, 1): 10.0, (1, 0): 10.4, (2, 3): 8.0}
        ranks, groups, unique = _dense_groups(keys, values)
        self.assertEqual(unique, 2)
        self.assertEqual(int(ranks[0]), int(ranks[1]))
        self.assertEqual(int(groups[0]), int(groups[1]))
        self.assertGreater(int(ranks[2]), int(ranks[0]))

    def test_gate_priority(self) -> None:
        self.assertEqual(choose_gate(_flags())["gate"], "DATA-A")
        self.assertFalse(choose_gate(_flags())["planner_ready"])
        self.assertEqual(choose_gate(_flags(lineage_failure=True))["gate"], "DATA-E")
        self.assertEqual(choose_gate(_flags(leakage=True))["gate"], "DATA-C")
        self.assertEqual(choose_gate(_flags(reproducibility_failure=True))["gate"], "DATA-B")
        self.assertEqual(choose_gate(_flags(degenerate=True, data_a=False))["gate"], "DATA-D")

    def test_worker_selection_rejects_contention(self) -> None:
        from wt.benchmark_v2.phase6ib_highlevel_dataset.run_phase6ib import select_workers

        def row(workers: int, seconds: float, projected: float, failed: bool = False) -> dict:
            return {"workers": workers, "mean_scenario_seconds": seconds, "projected_seconds": projected, "oom": False, "failed": failed}

        self.assertEqual(select_workers([row(8, 100, 700), row(12, 100, 500), row(16, 100, 400)]), 16)
        self.assertEqual(select_workers([row(8, 100, 700), row(12, 100, 500), row(16, 140, 455)]), 12)
        self.assertEqual(select_workers([row(8, 100, 700), row(12, 140, 700), row(16, 140, 455)]), 8)

    def test_state_snapshot_has_no_label_or_future(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=322000)
        arrays = state_arrays(env)
        self.assertIn("q_bits", arrays)
        self.assertIn("uav_buoy_hypothetical_rate", arrays)
        self.assertNotIn("j15_bits", arrays)
        pools, _packed, illegal = build_pools(env, (("G1_RATE", 6),))
        self.assertEqual(int(illegal), 0)
        self.assertGreater(len(pools[("G1_RATE", 6)]), 0)
        self.assertEqual(int(env.t), 0)
        self.assertTrue(np.all(np.isfinite(arrays["q_bits"])))

    def test_step_displacement_is_one_environment_transition(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=322000)
        previous = env.uav_positions.copy()
        initial = step_observation(env, previous, None)
        self.assertTrue(np.all(initial["uav_step_displacement_m"] == 0.0))
        self.assertTrue(np.array_equal(initial["service_progress"], env.progress))
        move = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
        move[:, 0] = 1.0
        env.step(move, np.full(env.config.num_uavs, -1, dtype=np.int64))
        moved = step_observation(env, previous, 0)
        delta = env.uav_positions - previous
        self.assertTrue(np.allclose(moved["uav_step_displacement_m"], delta))
        self.assertLessEqual(float(np.max(np.linalg.norm(delta, axis=1))), float(env.config.step_length_m) + 1.0e-8)
        self.assertGreater(float(np.max(np.linalg.norm(delta, axis=1))), 1.0)

    def test_package_does_not_train_or_call_decide(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6ib_highlevel_dataset"
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("decide(", text)
            self.assertNotIn("loss.backward", text)
            self.assertNotIn("planner_ready = True", text)


if __name__ == "__main__":
    unittest.main()
