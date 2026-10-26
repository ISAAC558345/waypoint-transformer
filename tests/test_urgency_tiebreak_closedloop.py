"""Stage D3 rule checks. They do not read held-out splits or run the closed loop."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.protocol import (
    GATE_BLOCKED,
    GATE_COMPLETE,
    GATE_INCONCLUSIVE,
    choose_gate,
    label_horizon,
)
from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.rules import choose_executed, payload_first, urgency_tiebreak


FORBIDDEN = ("deci" + "de(", "322048", "322064")


def _row(candidate_id: int, payload: float, urgent: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "payload_bits": payload,
        "urgent_service_bits": urgent,
        "assignment": str(candidate_id),
        "assignment_vector": [candidate_id],
    }


class TiebreakRuleTest(unittest.TestCase):
    def test_urgency_cannot_leave_the_one_bit_group(self) -> None:
        rows = [_row(1, 100.0, 1.0), _row(2, 99.2, 50.0), _row(3, 90.0, 999.0)]
        self.assertEqual(payload_first(rows)["candidate_id"], 1)
        self.assertEqual(urgency_tiebreak(rows)["candidate_id"], 2)
        _payload, tie, executed = choose_executed(rows, "payload_first_urgency_tiebreak")
        self.assertEqual(executed["candidate_id"], tie["candidate_id"])
        self.assertGreaterEqual(float(executed["payload_bits"]), 99.0)

    def test_equal_urgent_uses_the_smaller_id(self) -> None:
        rows = [_row(4, 100.0, 5.0), _row(5, 100.0, 5.2)]
        self.assertEqual(urgency_tiebreak(rows)["candidate_id"], 4)
        self.assertEqual(payload_first(rows)["candidate_id"], 4)

    def test_horizons_stay_separate(self) -> None:
        self.assertEqual(label_horizon(0), 15)
        self.assertEqual(label_horizon(15), 15)
        self.assertEqual(label_horizon(30), 15)
        self.assertEqual(label_horizon(45), 3)

    def test_gate_blocks_a_weak_or_failed_pilot(self) -> None:
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, tiebreak_fraction=0.75, urgent_direction="consistent_nonnegative"), GATE_COMPLETE)
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, tiebreak_fraction=0.2, urgent_direction="consistent_nonnegative"), GATE_INCONCLUSIVE)
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, tiebreak_fraction=0.8, urgent_direction="mixed"), GATE_INCONCLUSIVE)
        self.assertEqual(choose_gate(mechanical_ok=False, budget_ok=True, tiebreak_fraction=1.0, urgent_direction="consistent_nonnegative"), GATE_BLOCKED)

    def test_sources_do_not_name_held_out_splits(self) -> None:
        root = Path("src/wt/benchmark_v2/phase6i_urgency_tiebreak_closedloop")
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        for token in FORBIDDEN:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
