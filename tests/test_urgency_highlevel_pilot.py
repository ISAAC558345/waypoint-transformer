"""Stage D0 checks. The MAPPO comparison uses one fresh assignment, not the reserved split."""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info
from wt.benchmark_v2.phase6h_executor_conditioned_macro.labels import batch_j15_mappo
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _frozen_actor
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.analyze import reversal_rate, spearman
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.labels import score_assignment
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.protocol import (
    ADAPTER_CONFIG_SHA256,
    GATE_BLOCKED,
    GATE_INCONCLUSIVE,
    GATE_PASS,
    choose_gate,
    config_sha256,
    pilot_scenarios,
)
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.sampling import select_candidates
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.shards import load_manifest, load_npz


FORBIDDEN = ("deci" + "de(", "generate_scenario(", "_store(", "FormalPlanner", "phase6i_urgency_lowlevel")


class HighLevelPilotTest(unittest.TestCase):
    def test_registry_prefix_and_gate_rules(self) -> None:
        chosen = pilot_scenarios()
        self.assertEqual(chosen, (322000, 322001, 322002, 322003))
        self.assertEqual(config_sha256(), ADAPTER_CONFIG_SHA256)
        self.assertEqual(choose_gate(mechanical_ok=False, budget_ok=True, mean_spearman=0.1, mean_reversal=0.5), GATE_BLOCKED)
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=False, mean_spearman=0.1, mean_reversal=0.5), GATE_BLOCKED)
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, mean_spearman=0.99, mean_reversal=0.01), GATE_INCONCLUSIVE)
        self.assertEqual(choose_gate(mechanical_ok=True, budget_ok=True, mean_spearman=0.4, mean_reversal=0.2), GATE_PASS)
        with self.assertRaises(RuntimeError):
            from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.protocol import assert_pilot_scenario

            assert_pilot_scenario(322064)

    def test_sampling_is_deterministic_and_covers_ranks(self) -> None:
        ids = np.arange(200, dtype=np.int64)
        labels = 10000.0 - ids.astype(np.float64)
        ranks = ids + 1
        first = select_candidates(ids, labels, ranks, scenario_id=322000, time_step=0)
        second = select_candidates(ids, labels, ranks, scenario_id=322000, time_step=0)
        self.assertEqual([row["candidate_id"] for row in first["rows"]], [row["candidate_id"] for row in second["rows"]])
        self.assertEqual(first["count"], 128)
        self.assertEqual(len({row["candidate_id"] for row in first["rows"]}), 128)
        top = [row["candidate_id"] for row in first["rows"] if row["selection"] == "top_j15"]
        self.assertEqual(top, list(range(32)))
        stratum_rows = [row for row in first["rows"] if row["selection"] == "stratum"]
        self.assertEqual(len(stratum_rows), 96)
        low = min(row["utility_rank"] for row in stratum_rows if row["stratum"] == 0)
        high = max(row["utility_rank"] for row in stratum_rows if row["stratum"] == 7)
        self.assertGreater(high, low)
        other = select_candidates(ids, labels, ranks, scenario_id=322000, time_step=15)
        self.assertNotEqual([row["candidate_id"] for row in first["rows"]], [row["candidate_id"] for row in other["rows"]])

    def test_rank_reversal_counts_payload_order(self) -> None:
        payload = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
        urgent = np.asarray([30.0, 20.0, 10.0], dtype=np.float64)
        pairs = reversal_rate(payload, urgent)
        self.assertEqual(pairs["payload_separated_pairs"], 3)
        self.assertEqual(pairs["reversed_pairs"], 3)
        self.assertEqual(pairs["reversal_rate"], 1.0)
        self.assertLess(float(spearman(payload, urgent)), 0.0)

    def test_frozen_blob_matches_manifest_checksum(self) -> None:
        manifest = load_manifest(322000, 0)
        _state, state_sha = load_npz(322000, 0, "state")
        candidates, candidate_sha = load_npz(322000, 0, "candidates")
        self.assertEqual(state_sha, manifest["checksum_state"])
        self.assertEqual(candidate_sha, manifest["checksum_candidates"])
        self.assertEqual(int(candidates["assignment"].shape[0]), 13327)
        self.assertEqual(set(candidates), {"candidate_id", "assignment", "j15_bits", "utility_rank", "tie_group_id"})
        disk = Path("src/wt/benchmark_v2/phase6ib_highlevel_dataset/dataset/train/scenario_322000")
        names = {path.name for path in disk.iterdir()}
        self.assertFalse(any(name.endswith(".npz") for name in names))
        from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.shards import ShardError, _refuse_dataset_write

        with self.assertRaises(ShardError):
            _refuse_dataset_write(disk / "t000_state.npz")

    def test_sources_do_not_call_forbidden_writers(self) -> None:
        root = Path("src/wt/benchmark_v2/phase6i_urgency_highlevel_pilot")
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        for token in FORBIDDEN:
            self.assertNotIn(token, text)
        self.assertNotIn("322064", text)

    def test_label_rollout_matches_j15_and_keeps_snapshot(self) -> None:
        env = L3BBitQueueEnvV2()
        install_light_info(env)
        env.reset(seed=322000)
        ledger = AdmissionAgeLedger(int(env.config.num_buoys))
        ledger.reset_from_queue(env.q_bits)
        snap = capture_state(env)
        saved = ledger.capture()
        assignment = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
        actor = _frozen_actor()
        reference, _seconds, _first, error = batch_j15_mappo(env, actor, [assignment], 15)
        first = score_assignment(snap, saved, actor, assignment, scenario_id=322000)
        second = score_assignment(snap, saved, actor, assignment, scenario_id=322000)
        key = (0, 1, 2, 3, 4, 5)
        self.assertLessEqual(abs(float(reference[key]) - float(first["payload_bits"])), float(env.tol_bits))
        self.assertLessEqual(float(error), float(env.tol_bits))
        self.assertLessEqual(float(first["ledger_gap_bits"]), float(env.tol_bits))
        self.assertLessEqual(float(first["generated_bits"]), float(env.tol_bits))
        self.assertEqual(first["payload_bits"], second["payload_bits"])
        self.assertEqual(first["urgent_service_bits"], second["urgent_service_bits"])
        self.assertEqual(int(first["steps"]), 15)
        after = capture_state(env)
        self.assertTrue(all(np.array_equal(snap[name], after[name]) for name in snap if isinstance(snap[name], np.ndarray)))
        self.assertEqual(ledger.capture(), saved)
        digest = hashlib.sha256(str(first["urgent_service_bits"]).encode("utf-8")).hexdigest()
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
