"""Stage D5 guards. Real arrivals do not choose a candidate."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_urgency_arrival_aware_labels.analyze import _top, pairwise_inversion, rank_state
from wt.benchmark_v2.phase6i_urgency_arrival_aware_labels.protocol import (
    GATE_BLOCKED,
    GATE_COMPLETE,
    GATE_INCONCLUSIVE,
    HORIZON,
    LABEL_SEED,
    choose_gate,
    scenario_effect,
    scenarios,
    study_effect,
)


class ArrivalAwareLabelTest(unittest.TestCase):
    def test_gate_and_effect_thresholds(self) -> None:
        self.assertEqual(scenario_effect(urgent_reversal=0.2, payload_reversal=0.0, payload_top1_within_one_bit=True, urgent_top1_within_one_bit=False), "material")
        self.assertEqual(scenario_effect(urgent_reversal=0.0, payload_reversal=0.0, payload_top1_within_one_bit=True, urgent_top1_within_one_bit=True), "small")
        self.assertEqual(study_effect(["material", "material", "material", "material"]), "material_consistent")
        self.assertEqual(study_effect(["material", "small", "material", "material"]), "unstable")
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, effect="material_consistent"), GATE_COMPLETE)
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, effect="unstable"), GATE_INCONCLUSIVE)
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, effect="small"), GATE_INCONCLUSIVE)
        self.assertEqual(choose_gate(mechanical_ok=False, budget_ok=True, effect="material_consistent"), GATE_BLOCKED)
        self.assertEqual(HORIZON, 15)
        self.assertEqual(LABEL_SEED, 604)
        self.assertEqual(scenarios(), (322000, 322001, 322002, 322003))

    def test_one_bit_top_ignores_a_lower_payload(self) -> None:
        rows = [
            {"candidate_id": 2, "zero_payload_bits": 100.0, "real_payload_bits": 1.0e9},
            {"candidate_id": 1, "zero_payload_bits": 100.2, "real_payload_bits": 0.0},
            {"candidate_id": 3, "zero_payload_bits": 90.0, "real_payload_bits": 1.0e9},
        ]
        chosen = _top(rows, "zero_payload_bits")
        self.assertEqual(int(chosen["candidate_id"]), 1)
        pairs = pairwise_inversion(
            [100.0, 90.0, 80.0],
            [70.0, 90.0, 100.0],
        )
        self.assertGreater(pairs["inversion_rate"], 0.0)

    def test_rank_state_reports_regret_without_selecting_a_policy(self) -> None:
        rows = []
        for index in range(4):
            rows.append(
                {
                    "candidate_id": index + 1,
                    "zero_payload_bits": 1000.0 - (10.0 * index),
                    "real_payload_bits": 500.0 + (10.0 * index),
                    "zero_urgent_service_bits": 100.0 + (10.0 * index),
                    "real_urgent_service_bits": 100.0 - (10.0 * index),
                    "payload_delta_bits": float(index),
                    "urgent_delta_bits": float(-index),
                    "dropped_delta_bits": 0.0,
                    "final_queue_delta_bits": 0.0,
                    "zero_payload_gap_vs_j15_bits": 0.0,
                }
            )
        summary = rank_state(rows)
        self.assertEqual(summary["zero_payload_top1"], 1)
        self.assertEqual(summary["real_payload_top1"], 4)
        self.assertGreater(summary["payload_top1_regret_bits"], 1.0)
        self.assertEqual(summary["zero_urgent_top1"], 4)
        self.assertEqual(summary["real_urgent_top1"], 1)

    def test_package_source_has_no_held_out_id_or_decision_call(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_urgency_arrival_aware_labels"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("322048", text)
        self.assertNotIn("322064", text)


if __name__ == "__main__":
    unittest.main()
