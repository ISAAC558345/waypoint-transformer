"""Stage D7 rule guards. Existing labels are not rescored."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_urgency_label_rule_design.analyze import delta_for, sensitivity_rows
from wt.benchmark_v2.phase6i_urgency_label_rule_design.protocol import (
    DELTA_PAYLOAD_MAX,
    FRACTIONS,
    GATE_BLOCKED,
    GATE_INCONCLUSIVE,
    GATE_READY,
    LAMBDA_H,
    choose_gate,
)
from wt.benchmark_v2.phase6i_urgency_label_rule_design.rules import payload_loss, rule_c, rule_p, urgent_first
from wt.benchmark_v2.phase6i_urgency_label_rule_design.sources import StateBatch


def _row(candidate_id: int, payload: float, urgent: float) -> dict:
    return {"candidate_id": candidate_id, "assignment": f"a{candidate_id}", "payload_bits": payload, "urgent_bits": urgent}


class LabelRuleDesignTest(unittest.TestCase):
    def test_rule_p_stays_inside_the_one_bit_payload_group(self) -> None:
        rows = [_row(3, 100.0, 5.0), _row(2, 100.4, 9.0), _row(1, 90.0, 40.0)]
        self.assertEqual(payload_loss(100.4, 100.0), 0.0)
        self.assertEqual(payload_loss(99.0, 100.0), 0.0)
        self.assertGreater(payload_loss(98.0, 100.0), 1.0)
        chosen = rule_p(rows)
        self.assertEqual(int(chosen["candidate_id"]), 2)
        self.assertEqual(int(rule_c(rows, 0.0)["candidate_id"]), 2)
        self.assertEqual(int(rule_c(rows, 11.0)["candidate_id"]), 1)
        self.assertEqual(int(urgent_first(rows)["candidate_id"]), 1)
        self.assertIsNone(DELTA_PAYLOAD_MAX)
        self.assertIsNone(LAMBDA_H)
        self.assertEqual(FRACTIONS[0], 0.0)
        self.assertEqual(FRACTIONS[-1], 1.0)
        self.assertEqual(delta_for(20.0, 0.25), 5.0)

    def test_gate_keeps_the_loss_budget_unset(self) -> None:
        self.assertEqual(
            choose_gate(sources_verified=True, split_ok=True, sensitivity_complete=True, fields_support_rules=True, delta_unset=True),
            GATE_READY,
        )
        self.assertEqual(
            choose_gate(sources_verified=True, split_ok=True, sensitivity_complete=False, fields_support_rules=True, delta_unset=True),
            GATE_INCONCLUSIVE,
        )
        self.assertEqual(
            choose_gate(sources_verified=False, split_ok=True, sensitivity_complete=True, fields_support_rules=True, delta_unset=True),
            GATE_BLOCKED,
        )
        self.assertEqual(
            choose_gate(sources_verified=True, split_ok=True, sensitivity_complete=True, fields_support_rules=True, delta_unset=False),
            GATE_BLOCKED,
        )

    def test_zero_fraction_matches_rule_p_and_selects_nothing(self) -> None:
        batch = StateBatch(
            source="D6",
            scenario_id=1,
            time_step=30,
            horizon=15,
            horizon_source="labels_csv",
            label_family="real_arrival",
            arrival_semantics="real",
            state_path="frozen",
            rows=[_row(4, 10.0, 1.0), _row(5, 30.0, 2.0), _row(6, 20.0, 8.0)],
            pool_count=10,
        )
        rows = sensitivity_rows(batch)
        self.assertEqual(len(rows), len(FRACTIONS))
        self.assertEqual(int(rows[0]["candidate_id"]), int(rule_p(batch.rows)["candidate_id"]))
        self.assertTrue(all(int(row["budget_selected"]) == 0 for row in rows))
        self.assertEqual(int(rows[-1]["candidate_id"]), 6)

    def test_package_source_has_no_held_out_id_or_decision_call(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_urgency_label_rule_design"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("322048", text)
        self.assertNotIn("322064", text)


if __name__ == "__main__":
    unittest.main()
