"""Adapter binding checks. They do not open reserved scenarios or the planner."""
from __future__ import annotations

import unittest

from wt.benchmark_v2.phase6f_system_bringup.protocol import CANDIDATE_K
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _key
from wt.benchmark_v2.phase6iy_candidate_adapter.adapter import CandidateBinding, bind_keys
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import FORMAL_ENTRY_K, assign_gate, config_payload


class Phase6IYCandidateAdapterTests(unittest.TestCase):
    def test_gate_and_unchanged_formal_k(self) -> None:
        self.assertEqual(assign_gate(comparable=False, identical=False), "CANDIDATE_POOL_ALIGNMENT_INCONCLUSIVE")
        self.assertEqual(assign_gate(comparable=True, identical=False), "CANDIDATE_POOL_ALIGNMENT_BLOCKED")
        self.assertEqual(assign_gate(comparable=True, identical=True), "CANDIDATE_POOL_ALIGNMENT_SUPPORTED")
        self.assertEqual(FORMAL_ENTRY_K, 5)
        self.assertEqual(int(CANDIDATE_K), 5)
        self.assertEqual(config_payload()["k"], 6)

    def test_binding_round_trip_uses_assignment_content(self) -> None:
        keys = sorted({_key(item) for item in ((1, -1), (-1, -1), (0, 1), (1, -1))})
        bindings, mapping = bind_keys(keys, n_uavs=2, n_buoys=2)
        self.assertEqual([binding.score_index for binding in bindings], [0, 1, 2])
        self.assertEqual(bindings[0], CandidateBinding(0, (-1, -1), True, True))
        self.assertEqual(mapping[(0, 1)], 1)
        self.assertEqual(mapping[bindings[1].assignment], bindings[1].score_index)

    def test_duplicate_sorted_key_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            bind_keys([(-1, -1), (-1, -1)], n_uavs=2, n_buoys=2)
