"""Offline guards for the replay reproducibility rules."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag_replay_repro.compare import (
    bits_within_tolerance,
    cause_letter,
    choose_gate,
    first_difference,
)
from wt.benchmark_v2.phase6i_priority_tag_replay_repro.protocol import ACCOUNTING_TOLERANCE_BITS


def _row(mu: float, payload: float) -> dict:
    base = {
        "mu": np.asarray([mu, 0.0]),
        "desired": np.zeros(2),
        "projected": np.zeros(2),
        "executed": np.zeros(2),
        "position": np.zeros(2),
        "q_bits": np.zeros(1),
        "served": 0.0,
        "dropped": 0.0,
        "snap": 0.0,
        "payload": payload,
        "path_increment": 0.0,
        "obs_sha256": "same",
    }
    return base


class ReplayReproTest(unittest.TestCase):
    def test_first_difference_is_the_earliest_nonzero_field(self) -> None:
        left = [_row(0.0, 0.0), _row(0.0, 1.0)]
        right = [_row(0.0, 0.0), _row(0.25, 2.0)]
        found = first_difference(left, right)
        self.assertEqual(found["step_index"], 1)
        self.assertEqual(found["field"], "mu")

    def test_tolerance_stays_at_one_bit(self) -> None:
        self.assertEqual(ACCOUNTING_TOLERANCE_BITS, 1.0)
        self.assertTrue(bits_within_tolerance({key: 1.0 for key in (
            "payload_bits",
            "served_ordinary_bits",
            "served_tagged_bits",
            "dropped_ordinary_bits",
            "dropped_tagged_bits",
            "urgency_served_bits",
        )}))
        self.assertFalse(bits_within_tolerance({key: 1.1 for key in (
            "payload_bits",
            "served_ordinary_bits",
            "served_tagged_bits",
            "dropped_ordinary_bits",
            "dropped_tagged_bits",
            "urgency_served_bits",
        )}))

    def test_gate_requires_a_verified_step_and_a_matching_inherited_run(self) -> None:
        self.assertEqual(
            choose_gate(
                hashes_ok=True,
                split_ok=True,
                instrumented_same=True,
                inherited_matches=True,
                first_step_found=True,
                fresh_matches=False,
            ),
            "REPLAY_DIVERGENCE_LOCALIZED",
        )
        self.assertEqual(
            choose_gate(
                hashes_ok=True,
                split_ok=True,
                instrumented_same=True,
                inherited_matches=False,
                first_step_found=True,
                fresh_matches=False,
            ),
            "REPLAY_DIVERGENCE_INCONCLUSIVE",
        )
        self.assertEqual(
            choose_gate(
                hashes_ok=False,
                split_ok=True,
                instrumented_same=True,
                inherited_matches=True,
                first_step_found=True,
                fresh_matches=False,
            ),
            "REPLAY_DIVERGENCE_BLOCKED",
        )
        self.assertEqual(
            cause_letter(
                hashes_ok=True,
                split_ok=True,
                reset_equal=True,
                inherited_matches=True,
                first_field="mu",
            ),
            "A",
        )

    def test_package_source_has_no_held_out_id_or_decision_call(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_replay_repro"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("322048", text)
        self.assertNotIn("322064", text)


if __name__ == "__main__":
    unittest.main()
