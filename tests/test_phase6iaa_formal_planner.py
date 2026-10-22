"""Planner safety checks. They do not run an episode or open reserved scenarios."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6f_system_bringup.episode import _contamination, _fingerprint
from wt.benchmark_v2.phase6f_system_bringup.protocol import CANDIDATE_K
from wt.benchmark_v2.phase6iaa_formal_planner.planner import accept_scored_row, graph_from_env
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import (
    HISTORICAL_FORMAL_K,
    assert_scenario,
    assign_gate,
    read_policy,
    status_flags,
    write_policy,
)
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import GRAPH_KEYS
from wt.benchmark_v2.phase6iw_inference_adapter.interface import stable_choice
from wt.htw_phase9.const import EDGE_UB_DIM


class Phase6IAAFormalPlannerTests(unittest.TestCase):
    def test_gate_keeps_deployment_unverified(self) -> None:
        ready = assign_gate(interface_ok=True, smoke_ok=True, episode_ok=True, scheme_c_called=True, failures=0)
        blocked = assign_gate(interface_ok=False, smoke_ok=True, episode_ok=True, scheme_c_called=True, failures=1)
        waiting = assign_gate(interface_ok=True, smoke_ok=False, episode_ok=False, scheme_c_called=False, failures=0)
        self.assertEqual(ready, "PLANNER_INTEGRATION_READY")
        self.assertEqual(blocked, "PLANNER_INTEGRATION_BLOCKED")
        self.assertEqual(waiting, "PLANNER_INTEGRATION_INCONCLUSIVE")
        self.assertTrue(status_flags(ready)["planner_ready"])
        self.assertFalse(status_flags(ready)["deployment_verified"])
        self.assertFalse(status_flags(blocked)["planner_ready"])

    def test_bad_scores_do_not_return_an_index(self) -> None:
        self.assertIsNone(accept_scored_row(np.asarray([]), np.asarray([])))
        self.assertIsNone(accept_scored_row(np.asarray([1.0, np.nan]), np.asarray([True, True])))
        self.assertIsNone(accept_scored_row(np.asarray([1.0, 2.0]), np.asarray([True, False])))
        scores = np.asarray([1.0, 1.0, 0.5])
        self.assertEqual(accept_scored_row(scores, np.ones(3, dtype=bool)), 0)
        index, chosen = stable_choice(scores, np.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 0], [0, 1, 2, 3, 4, 5]]))
        self.assertEqual(index, 0)
        self.assertEqual(chosen, [1, 2, 3, 4, 5, 6])

    def test_policy_file_and_reserved_pool(self) -> None:
        self.assertEqual(int(CANDIDATE_K), 5)
        self.assertEqual(HISTORICAL_FORMAL_K, 5)
        with self.assertRaises(RuntimeError):
            assert_scenario(322064)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "active_policy.json"
            write_policy("g1_rate", path)
            self.assertEqual(read_policy(path), "g1_rate")
            write_policy("scheme_c", path)
            self.assertEqual(read_policy(path), "scheme_c")

    def test_live_graph_uses_the_raw_edge_contract(self) -> None:
        env = L3BBitQueueEnvV2()
        env.reset(seed=322000)
        before = _fingerprint(env, None)
        graph = graph_from_env(env, env.uav_positions.copy(), None)
        self.assertEqual(tuple(graph), GRAPH_KEYS)
        self.assertEqual(int(graph["edge"].shape[-1]), int(EDGE_UB_DIM))
        self.assertEqual(int(graph["uav"].shape[-1]), 8)
        self.assertEqual(_contamination(before, _fingerprint(env, None)), 0)

    def test_package_does_not_call_decide(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6iaa_formal_planner"
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("decide(", text)
            self.assertNotIn('("G1_RATE", 5)', text)


if __name__ == "__main__":
    unittest.main()
