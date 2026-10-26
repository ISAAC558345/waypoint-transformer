"""Stage D4 guards. The diagnostic does not read a held-out split or spend payload."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.protocol import label_horizon
from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.rules import choose_executed
from wt.benchmark_v2.phase6i_urgency_tiebreak_single_intervention.d3_reuse import load_t0
from wt.benchmark_v2.phase6i_urgency_tiebreak_single_intervention.protocol import (
    GATE_BLOCKED,
    GATE_COMPLETE,
    GATE_INCONCLUSIVE,
    arm_name,
    choose_gate,
    mechanism_name,
    payload_sign,
    scenarios,
)


class UrgencyTiebreakSingleInterventionTest(unittest.TestCase):
    def test_gate_and_mechanism(self) -> None:
        self.assertEqual(payload_sign([-2.0, -3.0, -4.0, -5.0]), "consistent_nonpositive")
        self.assertEqual(payload_sign([0.1, -0.2, 0.0, 0.4]), "flat")
        self.assertEqual(payload_sign([-2.0, 3.0, -4.0, 5.0]), "mixed")
        self.assertEqual(
            mechanism_name(sign_0="consistent_nonpositive", sign_30="flat", arrival_paired=True),
            "initial_assignment",
        )
        self.assertEqual(
            mechanism_name(sign_0="mixed", sign_30="consistent_nonpositive", arrival_paired=True),
            "unresolved",
        )
        self.assertEqual(
            mechanism_name(sign_0="consistent_nonpositive", sign_30="consistent_nonpositive", arrival_paired=False),
            "unpaired_arrivals",
        )
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, mechanism="initial_assignment"), GATE_COMPLETE)
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, mechanism="unresolved"), GATE_INCONCLUSIVE)
        self.assertEqual(choose_gate(mechanical_ok=False, budget_ok=True, mechanism="both_clocks"), GATE_BLOCKED)
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=False, mechanism="mid_episode"), GATE_BLOCKED)

    def test_real_rollout_cannot_select_the_candidate(self) -> None:
        rows = [
            {"candidate_id": 1, "payload_bits": 100.0, "urgent_service_bits": 1.0, "real_payload_bits": 1.0e9},
            {"candidate_id": 2, "payload_bits": 100.0, "urgent_service_bits": 50.0, "real_payload_bits": 0.0},
            {"candidate_id": 3, "payload_bits": 90.0, "urgent_service_bits": 999.0, "real_payload_bits": 1.0e9},
        ]
        _payload, _tie, executed = choose_executed(rows, "payload_first_urgency_tiebreak")
        self.assertEqual(int(executed["candidate_id"]), 2)
        self.assertEqual(label_horizon(0), 15)
        self.assertEqual(label_horizon(30), 15)
        self.assertEqual(label_horizon(45), 3)

    def test_reused_opening_choice_stays_inside_one_bit(self) -> None:
        row = load_t0(322000)
        self.assertEqual(row["tiebreak_assignment"], "3|4|1|5|2|0")
        self.assertEqual(row["payload_first_assignment"], "0|1|2|3|4|5")
        self.assertLessEqual(abs(float(row["label_payload_gap_bits"])), 1.0)
        self.assertEqual(arm_name(None), "scheme_c")
        self.assertEqual(arm_name(30), "intervention_30")
        with self.assertRaises(RuntimeError):
            arm_name(15)
        self.assertEqual(scenarios(), (322000, 322001, 322002, 322003))

    def test_package_source_has_no_held_out_id_or_decision_call(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_urgency_tiebreak_single_intervention"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("322048", text)
        self.assertNotIn("322064", text)


if __name__ == "__main__":
    unittest.main()
