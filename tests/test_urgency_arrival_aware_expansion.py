"""Stage D6 guards. Real arrivals do not choose a candidate."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_urgency_arrival_aware_expansion.analyze import _top, pairwise_inversion, pareto_frontier, rank_state
from wt.benchmark_v2.phase6i_urgency_arrival_aware_expansion.protocol import (
    GATE_BLOCKED,
    GATE_COMPLETE,
    GATE_INCONCLUSIVE,
    MAX_CANDIDATES,
    MAX_STUDY_ROLLOUTS,
    choose_gate,
    horizon_for,
    project_seconds,
    repeat_state,
    scenario_effect,
    scenarios,
    assert_same_horizon,
    time_effect,
)


def _row(index: int, zero_payload: float, real_payload: float, zero_urgent: float, real_urgent: float) -> dict:
    return {
        "candidate_id": index,
        "assignment": "0|1|2|3|4|5",
        "idle_count": 0,
        "horizon": 15,
        "zero_payload_bits": zero_payload,
        "real_payload_bits": real_payload,
        "zero_urgent_service_bits": zero_urgent,
        "real_urgent_service_bits": real_urgent,
        "payload_delta_bits": real_payload - zero_payload,
        "urgent_delta_bits": real_urgent - zero_urgent,
        "dropped_delta_bits": 0.0,
        "final_queue_delta_bits": 0.0,
        "real_dropped_bits": 0.0,
        "real_admitted_bits": 100.0,
        "real_generated_bits": 100.0,
        "zero_payload_gap_vs_j15_bits": 0.0,
        "arrival_account_gap_bits": 0.0,
    }


class ArrivalAwareExpansionTest(unittest.TestCase):
    def test_registry_scenarios_and_horizons_stay_separate(self) -> None:
        chosen = scenarios()
        self.assertEqual(chosen, tuple(range(322016, 322024)))
        self.assertEqual(horizon_for(0), 15)
        self.assertEqual(horizon_for(15), 15)
        self.assertEqual(horizon_for(30), 15)
        self.assertEqual(horizon_for(45), 3)
        self.assertEqual(assert_same_horizon((0, 15, 30)), 15)
        self.assertEqual(assert_same_horizon((45,)), 3)
        with self.assertRaises(RuntimeError):
            assert_same_horizon((30, 45))
        self.assertEqual(repeat_state(), (322016, 30))
        self.assertEqual(MAX_CANDIDATES, 128)
        self.assertEqual(MAX_STUDY_ROLLOUTS, 8192)
        self.assertLess(project_seconds(2.0, 20.0), 20.0 * 60.0)
        self.assertGreater(project_seconds(2.0, 40.0), 20.0 * 60.0)

    def test_gate_keeps_a_quiet_window_out_of_the_arrival_average(self) -> None:
        self.assertEqual(scenario_effect(urgent_reversal=0.2, payload_reversal=0.0, payload_top1_within_one_bit=True, urgent_top1_within_one_bit=False), "material")
        self.assertEqual(scenario_effect(urgent_reversal=0.0, payload_reversal=0.0, payload_top1_within_one_bit=True, urgent_top1_within_one_bit=True), "small")
        self.assertEqual(time_effect(["material", "material"]), "material_consistent")
        self.assertEqual(time_effect(["arms_match", "arms_match"]), "arms_match")
        self.assertEqual(time_effect(["material", "small"]), "unstable")
        effects = {0: "material_consistent", 15: "material_consistent", 30: "material_consistent", 45: "arms_match"}
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, reproducible=True, time_effects=effects), GATE_COMPLETE)
        mixed = dict(effects)
        mixed[30] = "unstable"
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, reproducible=True, time_effects=mixed), GATE_INCONCLUSIVE)
        weak = {0: "small", 15: "small", 30: "small", 45: "arms_match"}
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, reproducible=True, time_effects=weak), GATE_INCONCLUSIVE)
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, reproducible=False, time_effects=effects), GATE_BLOCKED)
        diverged = dict(effects)
        diverged[45] = "arms_diverged"
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, reproducible=True, time_effects=diverged), GATE_BLOCKED)

    def test_one_bit_top_and_pareto(self) -> None:
        rows = [
            _row(2, 100.0, 1.0e9, 10.0, 10.0),
            _row(1, 100.2, 0.0, 10.0, 10.0),
            _row(3, 90.0, 1.0e9, 10.0, 10.0),
        ]
        self.assertEqual(int(_top(rows, "zero_payload_bits")["candidate_id"]), 1)
        pairs = pairwise_inversion([30.0, 20.0, 10.0], [10.0, 20.0, 30.0])
        self.assertEqual(pairs["inverted_pairs"], 3)
        self.assertAlmostEqual(pairs["kendall_tau"], -1.0)
        tied = pairwise_inversion([5.0, 5.5], [1.0, 9.0])
        self.assertEqual(tied["compared_pairs"], 0)
        self.assertIsNone(tied["kendall_tau"])
        frontier = pareto_frontier(
            [
                _row(1, 0.0, 100.0, 0.0, 10.0),
                _row(2, 0.0, 50.0, 0.0, 5.0),
                _row(3, 0.0, 40.0, 0.0, 30.0),
            ]
        )
        self.assertEqual(frontier["candidate_ids"], [1, 3])
        summary = rank_state([_row(index + 1, 1000.0 - (10.0 * index), 500.0 + (10.0 * index), 100.0 + (10.0 * index), 100.0 - (10.0 * index)) for index in range(4)])
        self.assertEqual(summary["zero_payload_top1"], 1)
        self.assertEqual(summary["real_payload_top1"], 4)
        self.assertGreater(summary["payload_top1_regret_bits"], 1.0)
        self.assertEqual(summary["horizon"], 15)

    def test_package_source_has_no_held_out_id_or_decision_call(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_urgency_arrival_aware_expansion"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("322048", text)
        self.assertNotIn("322064", text)


if __name__ == "__main__":
    unittest.main()
