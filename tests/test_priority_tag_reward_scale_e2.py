"""E2 guards that do not roll a full episode."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import assert_allowed
from wt.benchmark_v2.phase6i_priority_tag_reward_scale_e2.protocol import (
    GATE_BLOCKED,
    GATE_INCONCLUSIVE,
    GATE_REPLICATED,
    choose_gate,
)
from wt.benchmark_v2.phase6i_priority_tag_reward_scale_e2.rollout import train_scenarios
from wt.benchmark_v2.phase6i_priority_tag_reward_scale_e2.scale import material_reorder


class RewardScaleE2Test(unittest.TestCase):
    def test_train_prefix_is_the_registered_eight(self) -> None:
        chosen = train_scenarios()
        self.assertEqual(chosen, tuple(range(321000, 321008)))
        for scenario in chosen:
            assert_allowed(int(scenario), split="train")

    def test_gate_stops_without_a_conservative_bound(self) -> None:
        self.assertEqual(
            choose_gate(
                integrity_ok=True,
                episodes=8,
                budget_exceeded=False,
                positive_candidate_supported=True,
                ratio_span_ok=True,
                e1_scale_close=True,
            ),
            GATE_REPLICATED,
        )
        self.assertEqual(
            choose_gate(
                integrity_ok=True,
                episodes=8,
                budget_exceeded=False,
                positive_candidate_supported=False,
                ratio_span_ok=True,
                e1_scale_close=True,
            ),
            GATE_INCONCLUSIVE,
        )
        self.assertEqual(
            choose_gate(
                integrity_ok=False,
                episodes=8,
                budget_exceeded=False,
                positive_candidate_supported=True,
                ratio_span_ok=True,
                e1_scale_close=True,
            ),
            GATE_BLOCKED,
        )
        self.assertEqual(
            choose_gate(
                integrity_ok=True,
                episodes=1,
                budget_exceeded=True,
                positive_candidate_supported=False,
                ratio_span_ok=False,
                e1_scale_close=False,
            ),
            GATE_BLOCKED,
        )

    def test_one_bit_payload_gap_is_not_a_material_reorder(self) -> None:
        flags = material_reorder([1.0, 1.0 + 5.0e-7], [0.01, 0.0])
        self.assertTrue(flags["order_changed"])
        self.assertFalse(flags["material_order_changed"])
        wider = material_reorder([1.02, 1.0], [0.0, 0.05])
        self.assertTrue(wider["material_order_changed"])

    def test_package_source_has_no_held_out_id_or_decision_call(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_reward_scale_e2"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("322048", text)
        self.assertNotIn("322064", text)


if __name__ == "__main__":
    unittest.main()
