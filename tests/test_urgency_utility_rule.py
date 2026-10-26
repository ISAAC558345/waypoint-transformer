"""Stage D2 checks. They use synthetic candidates and do not read held-out splits."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_urgency_utility_rule.analyze import (
    Candidate,
    dominates,
    leave_one_out,
    material_regret_bits,
    payload_first,
    state_analysis,
    urgent_first,
)
from wt.benchmark_v2.phase6i_urgency_utility_rule.protocol import (
    GATE_BLOCKED,
    GATE_INCONCLUSIVE,
    GATE_READY,
    choose_gate,
)


FORBIDDEN = ("deci" + "de(", "322048", "322064", "validation", "reserved")


def _candidate(candidate_id: int, payload: float, urgent: float) -> Candidate:
    return Candidate(candidate_id, payload, urgent, str(candidate_id))


class UtilityRuleTest(unittest.TestCase):
    def test_tie_is_not_a_payload_loss(self) -> None:
        rows = [_candidate(1, 100.0, 1.0), _candidate(2, 100.2, 5.0), _candidate(3, 90.0, 9.0)]
        self.assertTrue(dominates(rows[1], rows[0]))
        self.assertFalse(dominates(rows[2], rows[1]))
        self.assertEqual(payload_first(rows).candidate_id, 2)
        self.assertEqual(urgent_first(rows).candidate_id, 3)
        self.assertEqual(material_regret_bits(100.2, 100.0), 0.0)
        self.assertEqual(material_regret_bits(90.0, 100.0), -10.0)
        analysis = state_analysis(rows)
        self.assertEqual(analysis["urgent_opt_inside_payload_tie"], 0)
        self.assertAlmostEqual(analysis["material_top1_payload_regret_bits"], -10.2)
        self.assertEqual(analysis["segments"][0]["candidate_id"], 2)
        self.assertEqual(analysis["segments"][0]["material_payload_loss"], 0)
        self.assertGreater(analysis["segments"][1]["lambda_low"], 0.0)

    def test_leave_one_out_keeps_the_primary_count(self) -> None:
        records = [
            {"scenario_id": 322014, "material_top1_payload_regret_bits": 0.0, "urgent_opt_inside_payload_tie": 1},
            {"scenario_id": 322015, "material_top1_payload_regret_bits": -1601020.0, "urgent_opt_inside_payload_tie": 0},
        ]
        result = leave_one_out(records, 322015)
        self.assertEqual(result["full_count"], 2)
        self.assertEqual(result["without_count"], 1)
        self.assertFalse(result["removed_from_primary_gate"])
        self.assertEqual(result["full_material_loss_states"], 1)
        self.assertEqual(result["without_material_loss_states"], 0)

    def test_gate_stays_ready_when_lambda_is_unset(self) -> None:
        self.assertEqual(choose_gate(inputs_ok=True, frontier_ok=True, rules_ok=True), GATE_READY)
        self.assertEqual(choose_gate(inputs_ok=False, frontier_ok=True, rules_ok=True), GATE_BLOCKED)
        self.assertEqual(choose_gate(inputs_ok=True, frontier_ok=False, rules_ok=True), GATE_INCONCLUSIVE)

    def test_sources_do_not_name_held_out_splits(self) -> None:
        root = Path("src/wt/benchmark_v2/phase6i_urgency_utility_rule")
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        for token in FORBIDDEN:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
