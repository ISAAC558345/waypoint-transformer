"""E4 guards for the protocol review."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_priority_tag_reward_protocol_e4.protocol import (
    CANDIDATE_STATUS,
    E3_GATE,
    GATE_BLOCKED,
    GATE_READY,
    GATE_WAITING,
    OWNER_APPROVED,
    choose_gate,
)
from wt.benchmark_v2.phase6i_priority_tag_reward_protocol_e4.report import render_report


class RewardProtocolE4Test(unittest.TestCase):
    def test_unapproved_review_waits(self) -> None:
        self.assertFalse(OWNER_APPROVED)
        self.assertEqual(choose_gate(sources_ok=False, owner_approved=False), GATE_BLOCKED)
        self.assertEqual(choose_gate(sources_ok=True, owner_approved=False), GATE_WAITING)
        self.assertEqual(choose_gate(sources_ok=True, owner_approved=True), GATE_READY)

    def test_report_keeps_the_e3_gate_and_candidate_state(self) -> None:
        text = render_report(
            {
                "instruction_commit": "fe2967115cbb86e97e1b2d92c2c6bc47399681d0",
                "e2_commit": "5210dd7611befceed37c897a998cc1c5bd2a76bd",
                "e3_commit": "6492e9960af094dc28e71748ea4a5b6c8d988f86",
                "e2_gate": "URGENCY_REWARD_SCALE_INCONCLUSIVE",
                "e3_gate": E3_GATE,
                "e3_gate_preserved": E3_GATE,
                "gate": GATE_WAITING,
                "problems": [],
                "formal_formula": "r_base = served_bits/1e6 - 0.05*c_project - 0.01*c_residual",
                "checkpoint_sha256": "abc",
                "e2_files": [],
                "e3_files": [],
                "e1_schema_file": {},
                "formal_files": [],
                "cited_forms": [],
                "conditional_candidate": {
                    "status": CANDIDATE_STATUS,
                    "evidence": {
                        "bonus_over_penalty_p95": 11.613369238438576,
                        "bonus_over_payload_mean": 0.00012977092654096073,
                        "signal_retained_fraction": 1.0,
                        "payload_material_scenarios": [],
                        "base_material_scenarios": [321000],
                    },
                },
            }
        )
        self.assertIn(E3_GATE, text)
        self.assertIn(CANDIDATE_STATUS, text)
        self.assertIn(GATE_WAITING, text)
        self.assertNotIn("E3 Gate 改为 READY", text)

    def test_package_source_has_no_held_out_id_or_decision_call(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_reward_protocol_e4"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("322048", text)
        self.assertNotIn("322064", text)


if __name__ == "__main__":
    unittest.main()
