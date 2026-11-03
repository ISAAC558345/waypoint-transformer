"""Synthetic checks for the fixed tag rule. They do not step the registered scenarios."""
from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag_assignment_rule.criteria import mechanism_evidence
from wt.benchmark_v2.phase6i_priority_tag_assignment_rule.identity import contract_problems, history_problems
from wt.benchmark_v2.phase6i_priority_tag_assignment_rule.protocol import (
    SCENARIOS,
    TAG_WEIGHT,
    StepBudget,
    registered_flags,
)
from wt.benchmark_v2.phase6i_priority_tag_assignment_rule.rule import (
    _score_assignment,
    link_components,
    select_registered,
    select_rule,
)
from wt.benchmark_v2.phase6i_priority_tag_assignment_rule.stats import bootstrap_mean

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_assignment_rule"
REPLAY = PACKAGE / "replay.py"
RULE = PACKAGE / "rule.py"


def _bucket(bits: float, tag: int, insert: int, admission: int = 0) -> dict:
    return {
        "admission_step": int(admission),
        "insert_index": int(insert),
        "priority_tag": int(tag),
        "bits_remaining": float(bits),
    }


def _pair(queue, rate_value: float):
    width = len(queue)
    rate = np.full((2, width), float(rate_value), dtype=np.float64)
    eta = np.zeros((2, width), dtype=np.float64)
    return rate, eta


class FixedTagRuleTests(unittest.TestCase):
    def test_ordinary_prefix_blocks_a_tagged_bucket(self) -> None:
        rate, eta = _pair([25.0], 1.0)
        buckets = [[_bucket(20.0, 0, 0), _bucket(5.0, 1, 1)]]
        capacity, tagged = link_components(rate, eta, [25.0], buckets, horizon=20.0, delta_t=1.0, decision_time=15)
        self.assertAlmostEqual(float(capacity[0, 0]), 20.0)
        self.assertAlmostEqual(float(tagged[0, 0]), 0.0)

    def test_partial_tagged_bucket_counts_only_the_taken_bits(self) -> None:
        rate, eta = _pair([25.0], 1.0)
        buckets = [[_bucket(10.0, 0, 0), _bucket(15.0, 1, 1)]]
        capacity, tagged = link_components(rate, eta, [25.0], buckets, horizon=12.0, delta_t=1.0, decision_time=15)
        self.assertAlmostEqual(float(capacity[0, 0]), 12.0)
        self.assertAlmostEqual(float(tagged[0, 0]), 2.0)

    def test_empty_queue_and_future_tags_do_not_enter_the_score(self) -> None:
        rate, eta = _pair([0.0, 10.0], 1.0e6)
        buckets = [[], [_bucket(10.0, 0, 0, 0), _bucket(100.0, 1, 1, 20)]]
        capacity, tagged = link_components(rate, eta, [0.0, 10.0], buckets, horizon=15.0, delta_t=1.0, decision_time=15)
        self.assertEqual(float(capacity[0, 0]), 0.0)
        self.assertEqual(float(tagged[0, 0]), 0.0)
        self.assertAlmostEqual(float(capacity[0, 1]), 10.0)
        self.assertEqual(float(tagged[0, 1]), 0.0)

    def test_ledger_mismatch_is_rejected(self) -> None:
        rate, eta = _pair([10.0], 1.0)
        with self.assertRaises(RuntimeError):
            link_components(rate, eta, [10.0], [[_bucket(20.0, 0, 0)]], horizon=15.0, delta_t=1.0, decision_time=15)

    def test_tag_term_changes_only_the_tagged_choice(self) -> None:
        rate, eta = _pair([10.0, 10.0, 0.0], 1.0e6)
        buckets = [[_bucket(10.0, 0, 0)], [_bucket(10.0, 1, 1)], []]
        pool = [np.array([0, -1]), np.array([1, -1]), np.array([0, 1])]
        chosen = select_rule(
            pool, 0, rate, eta, [10.0, 10.0, 0.0], buckets,
            horizon=15.0, delta_t=1.0, decision_time=15, weight=0.001,
        )
        self.assertEqual(chosen["masked"]["pool_index"], 0)
        self.assertEqual(chosen["tagged"]["pool_index"], 1)
        self.assertTrue(chosen["tagged_differs_from_masked"])
        self.assertNotEqual(chosen["tagged"]["assignment"], [0, 1])
        registered = select_registered(
            pool, 0, rate, eta, [10.0, 10.0, 0.0], buckets,
            horizon=15.0, delta_t=1.0, decision_time=15,
        )
        self.assertEqual(registered["tagged"]["pool_index"], chosen["tagged"]["pool_index"])
        self.assertEqual(registered["weight"], 0.001)

    def test_zero_tags_keep_the_masked_choice(self) -> None:
        rate, eta = _pair([10.0, 4.0], 1.0e6)
        buckets = [[_bucket(10.0, 0, 0)], [_bucket(4.0, 0, 1)]]
        pool = [np.array([1, -1]), np.array([0, -1])]
        chosen = select_registered(
            pool, 0, rate, eta, [10.0, 4.0], buckets, horizon=15.0, delta_t=1.0, decision_time=15,
        )
        self.assertEqual(chosen["masked"]["assignment"], chosen["tagged"]["assignment"])
        self.assertFalse(chosen["tagged_differs_from_masked"])

    def test_exact_tie_uses_the_smallest_original_index(self) -> None:
        rate, eta = _pair([10.0, 10.0], 1.0e6)
        buckets = [[_bucket(10.0, 0, 0)], [_bucket(10.0, 0, 1)]]
        pool = [np.array([1, -1]), np.array([0, -1]), np.array([1, -1])]
        chosen = select_registered(
            pool, 2, rate, eta, [10.0, 10.0], buckets, horizon=15.0, delta_t=1.0, decision_time=15,
        )
        self.assertEqual(chosen["natural"]["pool_index"], 2)
        self.assertEqual(chosen["masked"]["pool_index"], 0)
        self.assertEqual(chosen["tagged"]["pool_index"], 0)
        self.assertEqual(chosen["natural"]["assignment"], chosen["masked"]["assignment"])

    def test_mask_is_itemwise_and_idle_adds_nothing(self) -> None:
        rate, eta = _pair([10.0, 10.0], 1.0e6)
        buckets = [[_bucket(10.0, 1, 0)], [_bucket(10.0, 1, 1)]]
        pool = [np.array([-1, 0]), np.array([0, -1]), np.array([0, 1])]
        chosen = select_registered(
            pool, 1, rate, eta, [10.0, 10.0], buckets, horizon=15.0, delta_t=1.0, decision_time=15,
        )
        self.assertEqual(chosen["eligible_count"], 1)
        self.assertEqual(chosen["natural"]["P"], "10.0")
        self.assertEqual(chosen["natural"]["K"], "10.0")

    def test_same_inputs_do_not_consume_rng(self) -> None:
        rate, eta = _pair([10.0], 1.0e6)
        buckets = [[_bucket(10.0, 1, 0)]]
        pool = [np.array([-1, -1]), np.array([0, -1])]
        np.random.seed(12345)
        before = np.random.get_state()
        first = select_registered(pool, 1, rate, eta, [10.0], buckets, horizon=15.0, delta_t=1.0, decision_time=15)
        second = select_registered(pool, 1, rate, eta, [10.0], buckets, horizon=15.0, delta_t=1.0, decision_time=15)
        after = np.random.get_state()
        self.assertEqual(first["tagged"]["pool_index"], second["tagged"]["pool_index"])
        self.assertEqual(before[2], after[2])
        self.assertTrue(np.array_equal(before[1], after[1]))

    def test_tag_proxy_cannot_exceed_payload_proxy(self) -> None:
        with self.assertRaises(RuntimeError):
            _score_assignment((0,), np.array([[1.0]]), np.array([[2.0]]))

    def test_selector_signature_has_no_scenario_or_environment(self) -> None:
        for function in (select_rule, select_registered):
            names = set(inspect.signature(function).parameters)
            self.assertNotIn("scenario_id", names)
            self.assertNotIn("reference_window", names)
        self.assertNotIn("scenario_id", RULE.read_text(encoding="utf-8"))
        self.assertNotIn("env.step", RULE.read_text(encoding="utf-8"))
        self.assertNotIn("num_uavs", RULE.read_text(encoding="utf-8"))
        self.assertNotIn("num_buoys", RULE.read_text(encoding="utf-8"))

    def test_sources_do_not_train_or_name_held_out_ids(self) -> None:
        text = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE.glob("*.py"))
        text += "\n" + Path(__file__).read_text(encoding="utf-8")
        replay = REPLAY.read_text(encoding="utf-8")
        self.assertEqual(replay.count("session.draw()"), 1)
        self.assertNotIn("assignment_rng" + ".integers", text)
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("optimizer" + ".step", text)
        self.assertNotIn("backward" + "(", text)
        self.assertNotIn("322" + "048", text)
        self.assertNotIn("322" + "064", text)
        self.assertEqual(TAG_WEIGHT, 0.001)
        self.assertEqual(SCENARIOS, tuple(range(321008, 321024)))
        flags = registered_flags()
        self.assertFalse(flags["performance_pass"])
        self.assertFalse(flags["release_eligible"])
        self.assertFalse(flags["merged"])
        self.assertEqual(flags["training_updates"], 0)

    def test_budget_rejects_steps_outside_its_own_ceiling(self) -> None:
        budget = StepBudget()
        budget.add("reference", 768)
        with self.assertRaises(RuntimeError):
            budget.add("reference")
        with self.assertRaises(RuntimeError):
            budget.add("fixture")

    def test_mechanism_order(self) -> None:
        def row(**overrides):
            base = {
                "scenario_id": 0,
                "interpretable": True,
                "tagged_differs_from_masked": False,
                "s0_tm": False,
                "s2_tm": False,
                "s2_tn": False,
                "safety_t": 0,
                "safety_m": 0,
                "safety_n": 0,
                "constraint_t": 0,
                "constraint_m": 0,
                "constraint_n": 0,
                "payload_tm": 0.0,
                "payload_tn": 0.0,
            }
            base.update(overrides)
            return base

        no_effect = [row(s0_tm=True) for _ in range(16)]
        self.assertEqual(mechanism_evidence(no_effect)["label"], "NO_TAG_DECISION_EFFECT")
        replicated = [row(tagged_differs_from_masked=True) for _ in range(16)]
        replicated[0]["s2_tm"] = True
        replicated[0]["s2_tn"] = True
        replicated[1]["s2_tm"] = True
        replicated[1]["s2_tn"] = True
        self.assertEqual(mechanism_evidence(replicated)["label"], "REPLICATED_GUARDED_TAG_SIGNAL")
        local = [row(tagged_differs_from_masked=True, s0_tm=True) for _ in range(16)]
        judged = mechanism_evidence(local)
        self.assertEqual(judged["label"], "LOCAL_OR_UNGUARDED_TAG_EFFECT")
        self.assertTrue(any("S2" in item for item in judged["failed_checks"]))
        unsafe = [row(tagged_differs_from_masked=True, s0_tm=True, s2_tm=True, s2_tn=True, safety_t=1) for _ in range(16)]
        unsafe_judged = mechanism_evidence(unsafe)
        self.assertEqual(unsafe_judged["label"], "LOCAL_OR_UNGUARDED_TAG_EFFECT")
        self.assertTrue(any("safety" in item for item in unsafe_judged["failed_checks"]))
        absent = [row(tagged_differs_from_masked=True) for _ in range(16)]
        self.assertEqual(mechanism_evidence(absent)["label"], "NO_GUARDED_TAG_SIGNAL")
        self.assertIsNone(mechanism_evidence(replicated[:15])["label"])

    def test_bootstrap_resamples_scenarios(self) -> None:
        zeros = bootstrap_mean([0.0] * 16)
        self.assertEqual(zeros["mean"], 0.0)
        self.assertTrue(zeros["crosses_zero"])
        self.assertEqual(zeros["draws"], 10000)
        self.assertEqual(zeros["seed"], 9400019)
        flat = bootstrap_mean([5.0] * 16)
        self.assertEqual(flat["ci_low"], 5.0)
        self.assertFalse(flat["crosses_zero"])
        self.assertEqual(bootstrap_mean([1.0] * 16), bootstrap_mean([1.0] * 16))
        with self.assertRaises(RuntimeError):
            bootstrap_mean([0.0] * 15)

    def test_history_and_contract_remain_registered(self) -> None:
        self.assertEqual(contract_problems(), [])
        self.assertEqual(history_problems(), [])


if __name__ == "__main__":
    unittest.main()
