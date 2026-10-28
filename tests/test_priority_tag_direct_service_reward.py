"""Locked checks for the separate age and tagged-service bonus."""
from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag.ledger import TagBucket
from wt.benchmark_v2.phase6i_priority_tag_reward.criteria import paired_interval
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.attribute import allocate_service
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.criteria import choose_gate, leave_two_largest, publish_gate, score_l3
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.protocol import (
    GATE_BLOCKED,
    GATE_FAIL,
    GATE_INCONCLUSIVE,
    GATE_PASS,
    LAMBDA_AGE,
    LAMBDA_TAG,
    RHO_AGE,
    RHO_TAG,
)
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.reward import split_bonus


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_direct_service_reward"


def _row(variant: str, seed: int, scenario: int, **overrides) -> dict:
    item = {
        "variant": variant,
        "model_seed": int(seed),
        "scenario_id": int(scenario),
        "served_tagged_bits": 1_000_000.0,
        "payload_bits": 1_000_000.0,
        "age_service_bits": 100_000.0,
        "served_ordinary_bits": 1_000_000.0,
        "dropped_bits": 10.0,
        "final_queue_bits": 10.0,
        "safety_violations": 0.0,
        "constraint_failures": 0.0,
        "sinr_mean": 1.0,
        "interference_mean": 1.0,
    }
    item.update(overrides)
    return item


def _table(tagged_delta, *, safety_by_seed=None, payload_delta=0.0) -> list[dict]:
    rows = []
    safety_by_seed = {} if safety_by_seed is None else safety_by_seed
    for scenario in range(321096, 321128):
        rows.append(_row("L0", 0, scenario))
        for seed in (0, 1, 2):
            delta = float(tagged_delta[seed] if isinstance(tagged_delta, dict) else tagged_delta)
            rows.append(_row("L2", seed, scenario))
            rows.append(
                _row(
                    "L3",
                    seed,
                    scenario,
                    served_tagged_bits=1_000_000.0 + delta,
                    payload_bits=1_000_000.0 + float(payload_delta),
                    safety_violations=float(safety_by_seed.get(seed, 0.0)),
                )
            )
    return rows


class DirectServiceRewardTest(unittest.TestCase):
    def test_coefficients_and_bounds(self) -> None:
        self.assertEqual((LAMBDA_AGE, RHO_AGE, LAMBDA_TAG, RHO_TAG), (0.0005, 0.00025, 0.0010, 0.00075))
        age, tag = split_bonus(1.0e6, 1.0e6, 1.0e6)
        self.assertAlmostEqual(age, 0.00025)
        self.assertAlmostEqual(tag, 0.00075)
        self.assertLessEqual(age + tag, 0.001 + 1.0e-12)
        self.assertEqual(split_bonus(0.0, 0.0, 0.0), (0.0, 0.0))
        with self.assertRaises(RuntimeError):
            split_bonus(-1.0, 0.0, 1.0)
        with self.assertRaises(RuntimeError):
            split_bonus(0.0, 10.0, 0.0)

    def test_bonus_uses_age_not_combined_urgency(self) -> None:
        bucket = TagBucket(40, "arrival", 1, 10.0, 0)
        before = float(bucket.bits_remaining)
        result = allocate_service([[bucket]], np.asarray([10.0]), 48)
        self.assertAlmostEqual(float(bucket.bits_remaining), before)
        age = float(result["age_service_bits"])
        tagged = float(result["served_tagged_bits"])
        combined = float(result["combined_urgency_served_bits"])
        self.assertAlmostEqual(age + float(result["tag_increment_service_bits"]), combined, delta=1.0e-6)
        self.assertAlmostEqual(tagged + float(result["served_ordinary_bits"]), 10.0, delta=1.0e-6)
        bonus_age, _bonus_tag = split_bonus(age, tagged, 10.0)
        self.assertAlmostEqual(bonus_age, min(0.0005 * age / 1.0e6, 0.00025 * 10.0 / 1.0e6))
        self.assertGreater(abs(bonus_age - (0.0005 * combined / 1.0e6)), 1.0e-12)

    def test_fifo_and_no_future_argument(self) -> None:
        parameters = inspect.signature(allocate_service).parameters
        self.assertNotIn("generated_arrival_bits", parameters)
        older = TagBucket(0, "reset", 0, 10.0, 0)
        younger = TagBucket(36, "arrival", 1, 10.0, 1)
        future = TagBucket(100, "arrival", 1, 10.0, 2)
        result = allocate_service([[older, younger]], np.asarray([10.0]), 48)
        self.assertAlmostEqual(float(result["served_ordinary_bits"]), 10.0, delta=1.0e-6)
        self.assertAlmostEqual(float(result["served_tagged_bits"]), 0.0, delta=1.0e-6)
        self.assertAlmostEqual(float(older.bits_remaining), 10.0)
        self.assertAlmostEqual(float(future.bits_remaining), 10.0)

    def test_leave_two_and_bootstrap_lock(self) -> None:
        self.assertEqual(int(BOOTSTRAP_DRAWS), 10000)
        self.assertEqual(int(BOOTSTRAP_SEED), 0)
        interval = paired_interval(np.ones(32))
        self.assertAlmostEqual(interval["mean"], 1.0)
        self.assertAlmostEqual(interval["ci_low"], 1.0)
        self.assertTrue(leave_two_largest(np.asarray([1.0, 1.0, 5.0, 9.0])))
        self.assertFalse(leave_two_largest(np.asarray([100.0, 100.0, -10.0, -10.0])))

    def test_gate_rules(self) -> None:
        passing = score_l3(_table(1000.0))
        self.assertEqual(passing["passing_seeds"], 3)
        self.assertEqual(
            choose_gate(smoke_ok=True, train_ok=True, replay_ok=True, hashes_ok=True, score=passing),
            GATE_PASS,
        )
        inconclusive = score_l3(_table(0.0))
        self.assertEqual(inconclusive["passing_seeds"], 0)
        self.assertEqual(
            choose_gate(smoke_ok=True, train_ok=True, replay_ok=True, hashes_ok=True, score=inconclusive),
            GATE_INCONCLUSIVE,
        )
        failing = score_l3(_table({0: -1000.0, 1: -1000.0, 2: 1000.0}))
        self.assertEqual(
            choose_gate(smoke_ok=True, train_ok=True, replay_ok=True, hashes_ok=True, score=failing),
            GATE_FAIL,
        )
        unsafe = score_l3(_table(1000.0, safety_by_seed={2: 1.0}))
        self.assertEqual(
            choose_gate(smoke_ok=True, train_ok=True, replay_ok=True, hashes_ok=True, score=unsafe),
            GATE_FAIL,
        )
        self.assertEqual(choose_gate(smoke_ok=False, train_ok=True, replay_ok=True, hashes_ok=True, score=passing), GATE_BLOCKED)
        self.assertEqual(publish_gate(GATE_PASS, False), GATE_INCONCLUSIVE)
        self.assertEqual(publish_gate(GATE_PASS, True), GATE_PASS)
        self.assertEqual(publish_gate(GATE_FAIL, False), GATE_FAIL)

    def test_forbidden_tokens_and_context(self) -> None:
        held = ("322" + "048", "322" + "064")
        for path in PACKAGE.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("deci" + "de(", text)
            for token in held:
                self.assertNotIn(token, text)
        own = Path(__file__).read_text(encoding="utf-8")
        self.assertNotIn("deci" + "de(", own)
        from wt.benchmark_v2.phase6i_priority_tag_attribution_full_replay.context import apply_historical_context, context_is_historical

        self.assertTrue(context_is_historical(apply_historical_context()))

    def test_missing_attribution_is_rejected(self) -> None:
        from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT
        from wt.benchmark_v2.phase6i_priority_tag_reward.actor import load_formal_actor
        from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR
        from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.worker import DirectServiceWorker

        actor = load_formal_actor(FORMAL_ACTOR, LOG_STD_INIT)
        actor.eval()
        worker = DirectServiceWorker(0, 0, actor, None, deterministic=True, scenario_id=321000)
        worker.reset()
        with self.assertRaises(RuntimeError):
            worker.step()

    def test_terminal_reset_keeps_pre_service_attribution(self) -> None:
        from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT
        from wt.benchmark_v2.phase6i_priority_tag_reward.actor import load_formal_actor
        from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR
        from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _install, _restore
        from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.worker import DirectServiceWorker

        actor = load_formal_actor(FORMAL_ACTOR, LOG_STD_INIT)
        actor.eval()
        rollout, original = _install()
        try:
            worker = DirectServiceWorker(0, 0, actor, None, deterministic=True)
            worker.reset()
            worker.env.t = int(worker.env.config.episode_horizon) - 1
            record = worker.step()
        finally:
            _restore(rollout, original)
        self.assertIn("bonus_age", record)
        self.assertGreaterEqual(float(record["bonus_age"]), 0.0)
        self.assertGreaterEqual(float(record["bonus_tag"]), 0.0)


if __name__ == "__main__":
    unittest.main()
