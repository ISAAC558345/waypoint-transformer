"""E3 guards for the offline reward-form comparison."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_priority_tag_reward_form_e3.compare import capped_bonus, payload_step_reorder, reward_reorder
from wt.benchmark_v2.phase6i_priority_tag_reward_form_e3.protocol import (
    GATE_BLOCKED,
    GATE_INCONCLUSIVE,
    GATE_READY,
    choose_gate,
)


class RewardFormE3Test(unittest.TestCase):
    def test_zero_payload_has_no_bounded_bonus(self) -> None:
        self.assertEqual(capped_bonus(0.25, 0.0, 0.002, 0.005), 0.0)
        self.assertAlmostEqual(capped_bonus(1.0, 2.0, 0.002, 0.001), 0.002)

    def test_one_bit_payload_gap_is_not_material(self) -> None:
        flags = reward_reorder([1.0, 1.0], [1.01, 1.0], [1.0, 1.0 + 5.0e-7])
        self.assertTrue(flags["order_changed"])
        self.assertFalse(flags["material_order_changed"])
        wider = reward_reorder([1.02, 1.0], [1.02, 1.05], [1.02, 1.0])
        self.assertTrue(wider["material_order_changed"])
        separated = reward_reorder([1.0, 0.9], [1.0, 1.1], [2.0, 1.0])
        self.assertTrue(separated["material_order_changed"])
        payload = payload_step_reorder([2.0, 1.0], [0.0, 0.2])
        self.assertFalse(payload["material_order_changed"])

    def test_gate_does_not_relax_the_standard(self) -> None:
        self.assertEqual(choose_gate(source_ok=False, analysis_ok=True, any_candidate_meets_standard=True), GATE_BLOCKED)
        self.assertEqual(choose_gate(source_ok=True, analysis_ok=True, any_candidate_meets_standard=False), GATE_INCONCLUSIVE)
        self.assertEqual(choose_gate(source_ok=True, analysis_ok=True, any_candidate_meets_standard=True), GATE_READY)

    def test_package_source_has_no_held_out_id_or_decision_call(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_reward_form_e3"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("322048", text)
        self.assertNotIn("322064", text)


if __name__ == "__main__":
    unittest.main()
