"""Guards for the direct tagged-service reward. These tests do not train."""
from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_direct_tag_service_reward.criteria import choose_gate, score_l3
from wt.benchmark_v2.phase6i_direct_tag_service_reward.protocol import (
    GATE_BLOCKED,
    GATE_INCONCLUSIVE,
    GATE_PASS,
    LAMBDA_TAG,
    RHO_TAG,
)
from wt.benchmark_v2.phase6i_direct_tag_service_reward.reward import direct_tag_bonus, validate_config


def _row(variant: str, seed: int, scenario: int, *, tagged: float, payload: float) -> dict:
    return {
        "variant": variant,
        "model_seed": seed,
        "scenario_id": scenario,
        "served_tagged_bits": tagged,
        "payload_bits": payload,
        "dropped_bits": 10.0,
        "sinr_mean": 1.0,
        "interference_mean": 0.1,
        "final_queue_bits": 20.0,
        "safety_violations": 0.0,
        "constraint_failures": 0.0,
        "age_service_bits": 5.0,
        "tag_increment_service_bits": 1.0,
    }


def _table(passing_seeds: int) -> list[dict]:
    rows = []
    for offset, scenario in enumerate(range(321096, 321128)):
        del offset
        rows.append(_row("L0", 0, scenario, tagged=1_000.0, payload=5_000.0))
        for seed in (0, 1, 2):
            rows.append(_row("L1", seed, scenario, tagged=900.0, payload=5_000.0))
            rows.append(_row("L2", seed, scenario, tagged=800.0, payload=5_000.0))
            tagged = 1_200.0 if seed < passing_seeds else 900.0
            rows.append(_row("L3", seed, scenario, tagged=tagged, payload=5_100.0))
    return rows


class DirectTagRewardTest(unittest.TestCase):
    def test_bonus_uses_tagged_bits_and_the_cap(self) -> None:
        self.assertEqual(LAMBDA_TAG, 0.002)
        self.assertEqual(RHO_TAG, 0.001)
        self.assertEqual(direct_tag_bonus(0.0, 0.0), 0.0)
        self.assertAlmostEqual(direct_tag_bonus(5.0, 15.0), 0.002 * 5.0 / 1.0e6)
        self.assertAlmostEqual(direct_tag_bonus(10.0, 10.0), 0.001 * 10.0 / 1.0e6)
        self.assertNotIn("urgency", inspect.signature(direct_tag_bonus).parameters)
        with self.assertRaises(RuntimeError):
            direct_tag_bonus(4.0, 1.0)
        with self.assertRaises(RuntimeError):
            validate_config(0.003, 0.001)

    def test_two_seeds_pass_and_one_seed_does_not(self) -> None:
        scored = score_l3(_table(2))
        self.assertEqual(scored["passing_seeds"], 2)
        self.assertTrue(scored["merge"])
        self.assertEqual(
            choose_gate(smoke_ok=True, train_ok=True, replay_ok=True, hashes_ok=True, passing_seeds=2),
            GATE_PASS,
        )
        partial = score_l3(_table(1))
        self.assertEqual(partial["passing_seeds"], 1)
        self.assertEqual(
            choose_gate(smoke_ok=True, train_ok=True, replay_ok=True, hashes_ok=True, passing_seeds=1),
            GATE_INCONCLUSIVE,
        )
        self.assertEqual(
            choose_gate(smoke_ok=True, train_ok=False, replay_ok=True, hashes_ok=True, passing_seeds=2),
            GATE_BLOCKED,
        )

    def test_sources_keep_the_forbidden_fragments_out(self) -> None:
        root = Path(__file__).resolve().parents[1]
        package = root / "src" / "wt" / "benchmark_v2" / "phase6i_direct_tag_service_reward"
        token = "deci" + "de("
        held = ("322" + "048", "322" + "064")
        self.assertNotIn(token, Path(__file__).read_text(encoding="utf-8"))
        for path in package.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(token, text)
            for item in held:
                self.assertNotIn(item, text)


if __name__ == "__main__":
    unittest.main()
