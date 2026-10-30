"""Boundary checks for the matched-mask sample. They do not step the eight training scenarios."""
from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.attribution import attribute_served
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.session import _window_match
from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.codec import decode, encode
from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.coverage import audit_saved_prefix
from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.criteria import opportunity_evidence
from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.detail import bucket_service_rows, detail_matches_aggregate
from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.identity import contract_problems, history_problems
from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.protocol import (
    OPPORTUNITY_ABSENT,
    OPPORTUNITY_LOCAL,
    OPPORTUNITY_REPLICATED,
    OPPORTUNITY_THIN,
    candidate_seed,
    registered_flags,
)
from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.selection import public_record, select_matched


def _mask() -> tuple[int, int, int, int, int, int]:
    return (0, 1, -1, -1, -1, -1)


class AssignmentCoverageTests(unittest.TestCase):
    def test_same_mask_cap_does_not_pad_or_follow_generation_prefix(self) -> None:
        baseline = _mask()
        alternatives = [(0, 2, -1, -1, -1, -1), (0, 3, -1, -1, -1, -1), (0, 4, -1, -1, -1, -1), (1, 2, -1, -1, -1, -1)]
        other = (-1, 0, 1, -1, -1, -1)
        idle = (-1, -1, -1, -1, -1, -1)
        pool = [idle, other, alternatives[0], alternatives[1], baseline, alternatives[2], alternatives[3]]
        chosen = select_matched(pool, 4, 1, 15)
        self.assertEqual(chosen["matched_alternative_count"], 4)
        self.assertEqual(chosen["layer_sizes"], [])
        self.assertEqual(len(chosen["selected"]), 5)
        self.assertFalse(chosen["selected"][0]["is_baseline"])
        self.assertTrue(any(row["is_baseline"] for row in chosen["selected"]))
        self.assertEqual([row["pool_index"] for row in chosen["selected"]], [2, 3, 4, 5, 6])
        for row in chosen["selected"]:
            self.assertEqual(row["active_mask"], [1, 1, 0, 0, 0, 0])
        self.assertNotIn(list(other), [row["assignment"] for row in chosen["selected"]])
        self.assertNotIn(list(idle), [row["assignment"] for row in chosen["selected"]])
        again = select_matched(pool, 4, 1, 15)
        self.assertEqual(again["selected"], chosen["selected"])
        short = select_matched([idle, other, alternatives[0], baseline], 3, 1, 15)
        self.assertEqual(short["matched_alternative_count"], 1)
        self.assertEqual(len(short["selected"]), 2)
        self.assertEqual(select_matched([baseline], 0, 1, 0)["selected"][0]["is_baseline"], True)
        self.assertEqual(len(select_matched([baseline], 0, 1, 0)["selected"]), 1)

    def test_four_strata_are_stable_and_do_not_touch_the_global_stream(self) -> None:
        baseline = _mask()
        alternatives = [(0, index, -1, -1, -1, -1) for index in range(2, 8)]
        pool = [(-1, -1, -1, -1, -1, -1), *alternatives, baseline]
        before = np.random.get_state()
        chosen = select_matched(pool, len(pool) - 1, 1, 0)
        self.assertTrue(np.array_equal(before[1], np.random.get_state()[1]))
        self.assertEqual(chosen["matched_alternative_count"], 6)
        self.assertEqual(chosen["layer_sizes"], [2, 2, 1, 1])
        self.assertEqual(len(chosen["selected"]), 5)
        self.assertEqual(select_matched(pool, len(pool) - 1, 1, 0)["stratum_offsets"], chosen["stratum_offsets"])
        self.assertNotEqual(select_matched(pool, len(pool) - 1, 1, 15)["candidate_seed"], chosen["candidate_seed"])
        layers = [list(map(int, layer)) for layer in np.array_split(np.arange(6), 4)]
        picked = []
        for offset, layer in zip(chosen["stratum_offsets"], layers):
            self.assertGreaterEqual(offset, 0)
            self.assertLess(offset, len(layer))
            picked.append(int(layer[offset]))
        self.assertEqual(chosen["stratum_pool_indices"], [item + 1 for item in picked])
        self.assertEqual(candidate_seed(1, 15), 9_300_000 + 115)

    def test_saved_prefix_recount_matches_the_registered_bias(self) -> None:
        audit = audit_saved_prefix()
        self.assertEqual(audit["problems"], [])
        self.assertEqual(audit["baseline_count"], 24)
        self.assertEqual(audit["alternative_count"], 96)

    def test_opportunity_order_does_not_treat_a_missing_candidate_as_absence(self) -> None:
        covered = [{"interpretable_alternatives": 1} for _ in range(24)]
        thin = [{"interpretable_alternatives": 1} for _ in range(23)] + [{"interpretable_alternatives": 0}]
        replicated = [
            {"scenario_id": 1, "interpretable": True, "is_baseline": False, "s0": True, "s2": True},
            {"scenario_id": 2, "interpretable": True, "is_baseline": False, "s0": True, "s2": True},
        ]
        self.assertEqual(opportunity_evidence(replicated, thin), OPPORTUNITY_REPLICATED)
        local = [{"scenario_id": 1, "interpretable": True, "is_baseline": False, "s0": True, "s2": False}]
        self.assertEqual(opportunity_evidence(local, thin), OPPORTUNITY_LOCAL)
        absent = [{"scenario_id": 1, "interpretable": True, "is_baseline": False, "s0": False, "s2": False}]
        self.assertEqual(opportunity_evidence(absent, covered), OPPORTUNITY_ABSENT)
        self.assertEqual(opportunity_evidence(absent, thin), OPPORTUNITY_THIN)
        ignored = [{"scenario_id": 1, "interpretable": False, "is_baseline": False, "s0": True, "s2": True}]
        self.assertEqual(opportunity_evidence(ignored, covered), OPPORTUNITY_ABSENT)

    def test_codec_roundtrip_keeps_rng_and_window_fields(self) -> None:
        generator = np.random.default_rng(123)
        state = generator.bit_generator.state
        first = float(generator.random())
        restored = np.random.default_rng(0)
        restored.bit_generator.state = decode(encode(state))
        self.assertEqual(first, float(restored.random()))
        python_state = random.getstate()
        random.setstate(decode(encode(python_state)))
        numpy_state = np.random.get_state()
        np.random.set_state(decode(encode(numpy_state)))
        torch_state = torch.get_rng_state().clone()
        restored_torch = decode(encode(torch_state))
        self.assertTrue(torch.equal(torch_state, restored_torch))
        payload = 52114872.479325905
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        row = {
            "action": np.zeros((6, 2)),
            "positions": np.ones((6, 2)),
            "payload": payload,
            "q_bits": np.arange(12, dtype=np.float64),
            "ledger": {"buckets": [[] for _ in range(12)]},
            "generated": np.zeros(12),
            "tags": (0, 1, 0),
            "assignment_rng": state,
            "env_rng": state,
        }
        self.assertTrue(_window_match([row], [decode(encode(row))])["ok"])

    def test_bucket_detail_matches_the_aggregate_and_does_not_mutate(self) -> None:
        buckets = [[
            {"admission_step": 0, "priority_tag": 0, "bits_remaining": 100.0, "insert_index": 1},
            {"admission_step": 4, "priority_tag": 1, "bits_remaining": 50.0, "insert_index": 2},
        ]]
        rows = bucket_service_rows(buckets, [120.0], 6, {1})
        self.assertEqual(sum(row["bits_taken"] for row in rows if row["service_class"] == "known_ordinary"), 100.0)
        self.assertEqual(sum(row["bits_taken"] for row in rows if row["service_class"] == "future_tagged"), 20.0)
        self.assertTrue(detail_matches_aggregate(rows, [120.0], buckets, 6, {1}, lambda _bucket, _time: 0.5))
        self.assertEqual(attribute_served(buckets, [120.0], 6, {1}, lambda _bucket, _time: 0.5)["known_tagged"], 0.0)
        self.assertEqual(buckets[0][0]["bits_remaining"], 100.0)

    def test_identity_and_release_flags_stay_closed(self) -> None:
        self.assertEqual(contract_problems(), [])
        self.assertEqual(history_problems(), [])
        flags = registered_flags()
        self.assertFalse(flags["release_eligible"])
        self.assertFalse(flags["merged"])
        self.assertFalse(flags["deployment_verified"])
        self.assertEqual(flags["training_updates"], 0)
        self.assertEqual(flags["parameter_updates"], 0)

    def test_new_sources_omit_forbidden_controls(self) -> None:
        token = "deci" + "de("
        held = ("322" + "048", "322" + "064")
        step = "optimizer." + "step"
        backward = "back" + "ward("
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_assignment_coverage"
        replay = (root / "replay.py").read_text(encoding="utf-8")
        selection = (root / "selection.py").read_text(encoding="utf-8")
        self.assertEqual(replay.count("session.draw()"), 1)
        self.assertNotIn("select_registered", replay)
        self.assertNotIn("select_registered", selection)
        self.assertNotIn("served_tagged", selection)
        self.assertNotIn("follow_step", selection)
        record = public_record([_mask(), (0, 2, -1, -1, -1, -1)], 0, 1, 0)
        self.assertEqual(record["candidate_seed"], candidate_seed(1, 0))
        for path in list(root.glob("*.py")) + [Path(__file__)]:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(token, text)
            self.assertNotIn(step, text)
            self.assertNotIn(backward, text)
            for item in held:
                self.assertNotIn(item, text)


if __name__ == "__main__":
    unittest.main()
