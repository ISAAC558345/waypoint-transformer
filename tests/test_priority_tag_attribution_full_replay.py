"""Guards for the full attribution replay. These tests do not run the 224 episodes."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_priority_tag_attribution_full_replay.compare import choose_gate, episode_status
from wt.benchmark_v2.phase6i_priority_tag_attribution_full_replay.context import assignment_seed
from wt.benchmark_v2.phase6i_priority_tag_attribution_full_replay.protocol import (
    ACCOUNTING_TOLERANCE_BITS,
    DECOMPOSITION_TOLERANCE_BITS,
    FULL_EPISODE_COUNT,
    GATE_BLOCKED,
    GATE_COMPLETE,
    GATE_INCONCLUSIVE,
    GLOBAL_SEED_ROLE,
    HISTORICAL_INHERITED_GLOBAL_SEED,
    PATH_TOLERANCE_M,
    PREFLIGHT_SCENARIO,
    TIME_LIMIT_SECONDS,
)


def _row(**overrides) -> dict:
    base = {
        "payload_bits": 10.0,
        "urgency_served_bits": 4.0,
        "served_tagged_bits": 3.0,
        "served_ordinary_bits": 7.0,
        "dropped_tagged_bits": 1.0,
        "dropped_ordinary_bits": 2.0,
        "dropped_bits": 3.0,
        "final_queue_bits": 6.0,
        "final_tagged_queue_bits": 2.0,
        "final_ordinary_queue_bits": 4.0,
        "path_length_m": 100.0,
        "age_service_bits": 3.0,
        "tag_increment_service_bits": 1.0,
        "combined_urgency_served_bits": 4.0,
        "recorded_urgency_served_bits": 4.0,
        "max_ledger_gap_bits": 0.0,
        "steps": 48,
    }
    base.update(overrides)
    return base


def _groups(age: float, tag: float, combined: float) -> list[dict]:
    return [
        {"age_delta_mean": age, "tag_increment_delta_mean": tag, "combined_delta_mean": combined}
        for _ in range(3)
    ]


class FullReplayAttributionTest(unittest.TestCase):
    def test_seed_roles_stay_distinct(self) -> None:
        self.assertEqual(HISTORICAL_INHERITED_GLOBAL_SEED, 2)
        self.assertEqual(GLOBAL_SEED_ROLE, "historical_inherited_process_seed")
        self.assertNotEqual(GLOBAL_SEED_ROLE, "eval_protocol_seed")
        self.assertEqual(PREFLIGHT_SCENARIO, 321101)
        self.assertEqual(assignment_seed(321101), 7_321_101)
        self.assertNotEqual(assignment_seed(321101), HISTORICAL_INHERITED_GLOBAL_SEED)

    def test_tolerances_stay_at_the_validated_values(self) -> None:
        self.assertEqual(ACCOUNTING_TOLERANCE_BITS, 1.0)
        self.assertEqual(PATH_TOLERANCE_M, 1.0e-9)
        self.assertEqual(DECOMPOSITION_TOLERANCE_BITS, 1.0e-6)
        self.assertEqual(TIME_LIMIT_SECONDS, 600.0)
        self.assertEqual(FULL_EPISODE_COUNT, 224)

    def test_episode_status_rejects_a_path_outside_1e9(self) -> None:
        saved = _row()
        self.assertTrue(episode_status(_row(), saved)["matches_saved"])
        missed = episode_status(_row(path_length_m=100.0 + 1.1e-9), saved)
        self.assertFalse(missed["matches_saved"])
        self.assertGreater(missed["path_gap_m"], PATH_TOLERANCE_M)

    def test_gate_follows_reproduction_and_l2_agreement(self) -> None:
        common = {"integrity_ok": True, "budget_ok": True, "reproduced": True, "decomposition_ok": True}
        self.assertEqual(choose_gate(**common, l2_groups=_groups(5.0, 1.0, 6.0)), GATE_COMPLETE)
        mixed = _groups(5.0, 1.0, 6.0)
        mixed[2] = {"age_delta_mean": 1.0, "tag_increment_delta_mean": 5.0, "combined_delta_mean": 6.0}
        self.assertEqual(choose_gate(**common, l2_groups=mixed), GATE_INCONCLUSIVE)
        self.assertEqual(
            choose_gate(integrity_ok=False, budget_ok=True, reproduced=True, decomposition_ok=True, l2_groups=_groups(5.0, 1.0, 6.0)),
            GATE_BLOCKED,
        )
        self.assertEqual(choose_gate(**common, l2_groups=None), GATE_BLOCKED)

    def test_new_sources_do_not_embed_the_forbidden_token(self) -> None:
        root = Path(__file__).resolve().parents[1]
        package = root / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_attribution_full_replay"
        token = "deci" + "de("
        self.assertNotIn(token, Path(__file__).read_text(encoding="utf-8"))
        held_out = ("322" + "048", "322" + "064")
        for path in package.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(token, text)
            for item in held_out:
                self.assertNotIn(item, text)


if __name__ == "__main__":
    unittest.main()
