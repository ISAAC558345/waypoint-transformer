"""Guards for the urgency attribution diagnostic."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag.ledger import PriorityTagLedger, TagBucket
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.attribute import allocate_service, service_origin_split
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.decide import next_design


class PriorityTagRewardAttributionTest(unittest.TestCase):
    def test_age_and_tag_increment_add_to_the_ledger_urgency(self) -> None:
        ledger = PriorityTagLedger(1)
        ledger.buckets[0] = [
            TagBucket(0, "environment_arrival", 0, 10.0, 0),
            TagBucket(36, "environment_arrival", 1, 10.0, 1),
        ]
        before = [float(bucket.bits_remaining) for bucket in ledger.buckets[0]]
        allocation = allocate_service(ledger.buckets, np.asarray([15.0]), 48)
        after = [float(bucket.bits_remaining) for bucket in ledger.buckets[0]]
        self.assertEqual(before, after)
        self.assertAlmostEqual(allocation["age_service_bits"], 11.25)
        self.assertAlmostEqual(allocation["tag_increment_service_bits"], 1.25)
        self.assertAlmostEqual(allocation["combined_urgency_served_bits"], 12.5)
        self.assertAlmostEqual(allocation["served_tagged_bits"], 5.0)
        self.assertEqual(allocation["tagged_served_fully_saturated_bits"], 0.0)
        split = ledger.apply_transition(
            np.asarray([15.0]),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            48,
            [None],
        )
        self.assertAlmostEqual(float(split["urgency_served_bits"][0]), allocation["combined_urgency_served_bits"])

    def test_old_tagged_bucket_loses_the_full_boost(self) -> None:
        buckets = [[TagBucket(0, "environment_arrival", 1, 8.0, 0)]]
        allocation = allocate_service(buckets, np.asarray([8.0]), 48)
        self.assertAlmostEqual(allocation["tag_increment_service_bits"], 0.0)
        self.assertAlmostEqual(allocation["tagged_served_fully_saturated_bits"], 8.0)
        self.assertAlmostEqual(allocation["age_service_bits"], 8.0)

    def test_partial_clip_is_not_counted_as_full_saturation(self) -> None:
        buckets = [[TagBucket(0, "environment_arrival", 1, 4.0, 0)]]
        allocation = allocate_service(buckets, np.asarray([4.0]), 40)
        self.assertGreater(allocation["tag_increment_service_bits"], 0.0)
        self.assertLess(allocation["tag_increment_service_bits"], 0.25 * 4.0)
        self.assertEqual(allocation["tagged_served_fully_saturated_bits"], 0.0)
        self.assertAlmostEqual(allocation["tagged_served_below_full_boost_bits"], 4.0)

    def test_fifo_serves_the_older_ordinary_bucket_before_a_younger_tag(self) -> None:
        buckets = [[
            TagBucket(5, "environment_arrival", 1, 3.0, 0),
            TagBucket(1, "environment_arrival", 0, 3.0, 1),
        ]]
        allocation = allocate_service(buckets, np.asarray([3.0]), 10)
        self.assertAlmostEqual(allocation["served_ordinary_bits"], 3.0)
        self.assertAlmostEqual(allocation["served_tagged_bits"], 0.0)

    def test_origin_split_adds_back_to_the_tagged_delta(self) -> None:
        split = service_origin_split(
            np.asarray([5.0, 0.0, 4.0]),
            np.asarray([5.0, 0.0, 1.0]),
            np.asarray([7.0, 2.0, 0.0]),
            np.asarray([1.0, 2.0, 0.0]),
        )
        self.assertAlmostEqual(split["tagged_delta_split_gap"], 0.0)

    def test_design_letter_stays_on_the_locked_rule(self) -> None:
        age_rows = [
            {"age_delta_mean": 10.0, "tag_increment_delta_mean": 1.0, "tagged_delta_mean": -2.0}
            for _ in range(3)
        ]
        self.assertEqual(next_design(True, True, True, age_rows), "B")
        tag_rows = [
            {"age_delta_mean": 1.0, "tag_increment_delta_mean": 5.0, "tagged_delta_mean": -2.0}
            for _ in range(3)
        ]
        self.assertEqual(next_design(True, True, True, tag_rows), "A")
        self.assertEqual(next_design(False, True, True, age_rows), "C")
        self.assertEqual(next_design(True, True, False, age_rows), "C")

    def test_package_source_has_no_held_out_id_or_decision_call(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_reward_attribution"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("322048", text)
        self.assertNotIn("322064", text)


if __name__ == "__main__":
    unittest.main()
