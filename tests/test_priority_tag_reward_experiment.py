"""Guards for the isolated priority-tag reward experiment."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag.ledger import PriorityTagLedger, TagBucket
from wt.benchmark_v2.phase6i_priority_tag_reward.criteria import leave_one_out, paired_interval
from wt.benchmark_v2.phase6i_priority_tag_reward.features import head_bucket, head_feature
from wt.benchmark_v2.phase6i_priority_tag_reward.reward import capped_bonus, experiment_reward


class PriorityTagRewardExperimentTest(unittest.TestCase):
    def test_bonus_is_bounded_and_zero_without_service(self) -> None:
        self.assertEqual(capped_bonus(10.0, 0.0, 0.0005, 0.001), 0.0)
        self.assertAlmostEqual(capped_bonus(1.0e6, 1.0e6, 0.0005, 0.001), 0.0005)
        self.assertAlmostEqual(capped_bonus(1.0e7, 1.0e6, 0.0005, 0.001), 0.001)
        executed = np.zeros((6, 2))
        reward, base, bonus = experiment_reward(0.0, 5.0, executed, executed, executed, 1.0e6, lambda_u=0.0, rho=0.001)
        self.assertEqual(bonus, 0.0)
        self.assertEqual(reward, base)

    def test_validity_distinguishes_empty_and_zero_urgency(self) -> None:
        ledger = PriorityTagLedger(1)
        self.assertTrue(np.all(head_feature(ledger, 0, 0) == 0.0))
        ledger.buckets[0].append(TagBucket(0, "reset_convention", 0, 10.0, 0))
        feature = head_feature(ledger, 0, 0)
        self.assertEqual(float(feature[0]), 0.0)
        self.assertEqual(float(feature[1]), 0.0)
        self.assertEqual(float(feature[2]), 1.0)

    def test_fifo_ignores_the_synthetic_tag(self) -> None:
        ledger = PriorityTagLedger(1)
        ledger.buckets[0] = [
            TagBucket(1, "environment_arrival", 1, 5.0, 2),
            TagBucket(0, "environment_arrival", 0, 5.0, 3),
        ]
        self.assertEqual(int(head_bucket(ledger, 0).admission_step), 0)
        self.assertEqual(int(head_bucket(ledger, 0).priority_tag), 0)

    def test_leave_one_out_rule_is_fixed(self) -> None:
        self.assertTrue(leave_one_out(np.asarray([1.0, 1.0, 1.0]), floor=0.0, strict=True))
        self.assertFalse(leave_one_out(np.asarray([10.0, -1.0, -1.0]), floor=0.0, strict=True))
        self.assertTrue(leave_one_out(np.asarray([0.0, 0.0, 0.0]), floor=0.0, strict=False))
        interval = paired_interval(np.asarray([1.0, 1.0, 1.0, 1.0]))
        self.assertGreater(interval["ci_low"], 0.0)

    def test_package_source_has_no_held_out_id_or_decision_call(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_reward"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("322048", text)
        self.assertNotIn("322064", text)


if __name__ == "__main__":
    unittest.main()
