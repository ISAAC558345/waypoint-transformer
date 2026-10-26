"""E1 ledger guards. Formal transitions are compared, not replaced."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag.ledger import PriorityTagLedger
from wt.benchmark_v2.phase6i_priority_tag.protocol import (
    GATE_BLOCKED,
    GATE_SCALE_READY,
    LAMBDA_CANDIDATES,
    LAMBDA_REFERENCE,
    LAMBDA_SELECTED,
    choose_gate,
)
from wt.benchmark_v2.phase6i_priority_tag.smoke import run_smoke
from wt.benchmark_v2.phase6i_priority_tag.tags import draw_tag


def _zeros(count: int = 2) -> np.ndarray:
    return np.zeros(count, dtype=np.float64)


class PriorityTagPrototypeTest(unittest.TestCase):
    def test_reset_stock_is_ordinary_and_conserved(self) -> None:
        ledger = PriorityTagLedger(2)
        queue = np.asarray([4.0e6, 1.0e6], dtype=np.float64)
        ledger.reset_from_queue(queue)
        gap = ledger.assert_matches(queue, 1.0)
        self.assertLessEqual(float(np.max(np.abs(gap))), 1.0)
        ordinary, tagged = ledger.bits_by_tag()
        self.assertEqual(float(np.sum(tagged)), 0.0)
        self.assertEqual(float(np.sum(ordinary)), float(np.sum(queue)))
        for buckets in ledger.buckets:
            for bucket in buckets:
                self.assertEqual(int(bucket.priority_tag), 0)
                self.assertEqual(bucket.age_origin, "reset_convention")
                self.assertEqual(int(bucket.admission_step), 0)

    def test_synthetic_tag_is_reproducible_and_binary(self) -> None:
        first = [draw_tag(321000, step, buoy) for step in (4, 6, 8) for buoy in range(6, 12)]
        second = [draw_tag(321000, step, buoy) for step in (4, 6, 8) for buoy in range(6, 12)]
        self.assertEqual(first, second)
        self.assertTrue(all(tag in (0, 1) for tag in first))
        self.assertIsNone(LAMBDA_SELECTED)
        self.assertNotIn(LAMBDA_REFERENCE, LAMBDA_CANDIDATES)

    def test_fifo_serves_older_ordinary_before_younger_tagged(self) -> None:
        ledger = PriorityTagLedger(1)
        ledger.reset_from_queue(np.asarray([100.0], dtype=np.float64))
        admitted = ledger.apply_transition(
            _zeros(1),
            np.asarray([1.0e6]),
            np.asarray([1.0e6]),
            _zeros(1),
            _zeros(1),
            4,
            [1],
        )
        self.assertEqual(float(admitted["served_tagged_bits"][0]), 0.0)
        self.assertEqual(float(admitted["admitted_tagged_bits"][0]), 1.0e6)
        served = ledger.apply_transition(
            np.asarray([150.0]),
            _zeros(1),
            _zeros(1),
            _zeros(1),
            _zeros(1),
            5,
            [None],
        )
        self.assertEqual(float(served["served_ordinary_bits"][0]), 100.0)
        self.assertEqual(float(served["served_tagged_bits"][0]), 50.0)
        self.assertAlmostEqual(
            float(served["urgency_served_bits"][0]),
            (5.0 / 48.0) * 100.0 + (0.25 + 1.0 / 48.0) * 50.0,
        )
        ordinary, tagged = ledger.bits_by_tag()
        self.assertEqual(float(tagged[0]), 1.0e6 - 50.0)
        self.assertEqual(float(ordinary[0]), 0.0)

    def test_drop_and_snap_do_not_enter_the_service_reward(self) -> None:
        ledger = PriorityTagLedger(1)
        ledger.reset_from_queue(np.asarray([4.0e6], dtype=np.float64))
        dropped = ledger.apply_transition(
            _zeros(1),
            np.asarray([1.0e6]),
            _zeros(1),
            np.asarray([1.0e6]),
            _zeros(1),
            4,
            [1],
        )
        self.assertEqual(float(dropped["dropped_tagged_bits"][0]), 1.0e6)
        self.assertEqual(float(dropped["served_tagged_bits"][0]), 0.0)
        self.assertEqual(float(dropped["urgency_served_bits"][0]), 0.0)
        self.assertEqual(float(ledger.bits_sum()[0]), 4.0e6)
        snapped = ledger.apply_transition(
            _zeros(1),
            _zeros(1),
            _zeros(1),
            _zeros(1),
            np.asarray([25.0]),
            5,
            [None],
        )
        self.assertEqual(float(snapped["served_ordinary_bits"][0]), 0.0)
        self.assertEqual(float(snapped["urgency_served_bits"][0]), 0.0)
        self.assertEqual(float(snapped["snap_ordinary_bits"][0]), 25.0)
        self.assertEqual(float(ledger.bits_sum()[0]), 4.0e6 - 25.0)

    def test_partial_drop_keeps_one_tag_on_the_new_arrival(self) -> None:
        ledger = PriorityTagLedger(1)
        ledger.reset_from_queue(np.zeros(1))
        split = ledger.apply_transition(
            np.zeros(1),
            np.asarray([1.0e6]),
            np.asarray([670000.0]),
            np.asarray([330000.0]),
            np.zeros(1),
            12,
            [1],
        )
        self.assertEqual(float(split["served_tagged_bits"][0]), 0.0)
        self.assertEqual(float(split["dropped_tagged_bits"][0]), 330000.0)
        self.assertEqual(float(split["admitted_tagged_bits"][0]), 670000.0)
        self.assertEqual(float(split["urgency_served_bits"][0]), 0.0)
        ordinary, tagged = ledger.bits_by_tag()
        self.assertEqual(float(ordinary[0]), 0.0)
        self.assertEqual(float(tagged[0]), 670000.0)

    def test_capture_restore_keeps_every_bucket_tag(self) -> None:
        ledger = PriorityTagLedger(1)
        ledger.reset_from_queue(np.asarray([10.0], dtype=np.float64))
        ledger.apply_transition(np.zeros(1), np.asarray([1.0e6]), np.asarray([1.0e6]), np.zeros(1), np.zeros(1), 4, [1])
        snap = ledger.capture()
        other = PriorityTagLedger(1)
        other.restore(snap)
        self.assertEqual(other.capture(), snap)
        gap = other.assert_matches(np.asarray([10.0 + 1.0e6]), 1.0)
        self.assertLessEqual(float(np.max(np.abs(gap))), 1.0)

    def test_view_uses_only_buckets_already_present(self) -> None:
        ledger = PriorityTagLedger(1)
        ledger.reset_from_queue(np.asarray([1.0e6], dtype=np.float64))
        before = ledger.capture()
        view = ledger.view(12)
        self.assertEqual(ledger.capture(), before)
        self.assertEqual(int(view["urgency_valid"][0]), 1)
        self.assertEqual(int(view["head_tag"][0]), 0)
        self.assertAlmostEqual(float(view["head_u"][0]), 12.0 / 48.0)
        empty = PriorityTagLedger(1)
        empty.reset_from_queue(np.zeros(1))
        blank = empty.view(12)
        self.assertEqual(int(blank["urgency_valid"][0]), 0)
        self.assertEqual(float(blank["head_u"][0]), 0.0)

    def test_short_smoke_matches_the_formal_transition(self) -> None:
        smoke = run_smoke()
        self.assertEqual(smoke["snap_source"], "cumulative_snap_bits_delta")
        self.assertNotIn("float_snap_bits", smoke["info_keys"])
        self.assertIn("dropped_bits_step", smoke["info_keys"])
        self.assertLessEqual(smoke["max_ledger_gap_bits"], 1.0)
        self.assertEqual(smoke["transition_gap"], 0.0)
        self.assertEqual(smoke["repeat_gap"], 0.0)
        self.assertTrue(smoke["tag_repeat_match"])
        for row in smoke["rows"]:
            self.assertLessEqual(abs(row["served_ordinary_bits"] + row["served_tagged_bits"] - row["served_bits"]), 1.0)
            self.assertLessEqual(abs(row["generated_bits"] - row["admitted_bits"] - row["dropped_bits"]), 1.0)
            self.assertLessEqual(
                abs(row["admitted_ordinary_bits"] + row["admitted_tagged_bits"] - row["admitted_bits"]),
                1.0,
            )
            self.assertLessEqual(
                abs(row["dropped_ordinary_bits"] + row["dropped_tagged_bits"] - row["dropped_bits"]),
                1.0,
            )
            drawn = [tag for tag in row["tags"] if tag is not None]
            self.assertEqual(len(drawn), int(round(row["generated_bits"] / 1.0e6)))
            self.assertTrue(all(tag in (0, 1) for tag in drawn))
        self.assertGreater(smoke["served_ordinary_bits"], 0.0)
        self.assertGreater(smoke["served_tagged_bits"], 0.0)
        self.assertEqual(smoke["tag_draws"], 36)
        self.assertLessEqual(
            abs(smoke["remaining_ordinary_bits"] + smoke["remaining_tagged_bits"] - smoke["final_q_bits"]),
            1.0,
        )
        self.assertLessEqual(abs(smoke["admitted_ordinary_bits"] + smoke["admitted_tagged_bits"] - smoke["admitted_bits"]), 1.0)
        self.assertLessEqual(abs(smoke["dropped_ordinary_bits"] + smoke["dropped_tagged_bits"] - smoke["dropped_bits"]), 1.0)

    def test_gate_stops_when_tracking_fails(self) -> None:
        self.assertEqual(choose_gate(ledger_ok=True, tracking_ok=True, scale_reported=True), GATE_SCALE_READY)
        self.assertEqual(choose_gate(ledger_ok=False, tracking_ok=True, scale_reported=True), GATE_BLOCKED)
        self.assertEqual(choose_gate(ledger_ok=True, tracking_ok=False, scale_reported=True), GATE_BLOCKED)

    def test_package_source_has_no_held_out_id_or_decision_call(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("322048", text)
        self.assertNotIn("322064", text)


if __name__ == "__main__":
    unittest.main()
