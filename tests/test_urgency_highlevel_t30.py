"""Stage D1 checks. Scenario selection stays on the train registry."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.sampling import select_candidates
from wt.benchmark_v2.phase6i_urgency_highlevel_t30_pilot.protocol import (
    EXPECTED_SCENARIOS,
    GATE_BLOCKED,
    GATE_INCONCLUSIVE,
    GATE_SUPPORTED,
    PILOT_SEED_T30,
    assert_t30_scenario,
    choose_gate,
    t30_scenarios,
)


FORBIDDEN = ("deci" + "de(", "generate_scenario(", "_store(", "FormalPlanner", "phase6i_urgency_lowlevel")


class T30PilotTest(unittest.TestCase):
    def test_registry_continuation(self) -> None:
        chosen = t30_scenarios()
        self.assertEqual(chosen, EXPECTED_SCENARIOS)
        self.assertEqual(chosen, tuple(range(322004, 322016)))
        self.assertNotIn(322000, chosen)
        self.assertNotIn(322003, chosen)
        with self.assertRaises(RuntimeError):
            assert_t30_scenario(322000)
        with self.assertRaises(RuntimeError):
            assert_t30_scenario(322048)
        with self.assertRaises(RuntimeError):
            assert_t30_scenario(322064)

    def test_gate_requires_average_and_state_coverage(self) -> None:
        common = {"mechanical_ok": True, "budget_ok": True, "coverage_complete": True}
        self.assertEqual(
            choose_gate(mean_spearman=0.90, mean_reversal=0.08, states_above=8, **common),
            GATE_SUPPORTED,
        )
        self.assertEqual(
            choose_gate(mean_spearman=0.90, mean_reversal=0.08, states_above=7, **common),
            GATE_INCONCLUSIVE,
        )
        self.assertEqual(
            choose_gate(mean_spearman=0.97, mean_reversal=0.02, states_above=2, **common),
            GATE_INCONCLUSIVE,
        )
        self.assertEqual(
            choose_gate(mechanical_ok=False, budget_ok=True, coverage_complete=True, mean_spearman=0.9, mean_reversal=0.2, states_above=12),
            GATE_BLOCKED,
        )

    def test_seed_602_is_distinct_and_deterministic(self) -> None:
        ids = np.arange(200, dtype=np.int64)
        labels = 10000.0 - ids.astype(np.float64)
        ranks = ids + 1
        first = select_candidates(ids, labels, ranks, scenario_id=322004, time_step=30, seed=PILOT_SEED_T30)
        second = select_candidates(ids, labels, ranks, scenario_id=322004, time_step=30, seed=PILOT_SEED_T30)
        previous = select_candidates(ids, labels, ranks, scenario_id=322004, time_step=30, seed=601)
        self.assertEqual([row["candidate_id"] for row in first["rows"]], [row["candidate_id"] for row in second["rows"]])
        self.assertNotEqual([row["candidate_id"] for row in first["rows"]], [row["candidate_id"] for row in previous["rows"]])
        self.assertEqual(first["count"], 128)
        self.assertEqual(first["seed"], 602)

    def test_sources_stay_away_from_held_out_splits(self) -> None:
        root = Path("src/wt/benchmark_v2/phase6i_urgency_highlevel_t30_pilot")
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        for token in FORBIDDEN:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
