"""Contracts for the additive, diagnostic-only Round-13 completion amendment."""
from __future__ import annotations

from pathlib import Path
import unittest
import inspect
import json

import numpy as np

from wt.round12.contracts import sha256
from wt.round13.amendment import (
    ORIGINAL_OUTPUT,
    _file_hashes,
    _hstep_return_rows,
    _round12_integrity,
    _scenario_mechanisms,
)
from wt.round13.audit import _validation_scenarios
from wt.round13.service_decoder import (
    allocate_payload,
    continuous_priority_from_current_state,
    masked_simplex,
)


ROOT = Path(__file__).resolve().parents[1]


class Round13CompletionAmendmentContracts(unittest.TestCase):
    def test_01_masked_simplex_rejects_padding_mass(self) -> None:
        self.assertTrue(np.allclose(masked_simplex(np.array([1.0, 3.0, 0.0]), np.array([True, True, False])), [0.25, 0.75, 0.0]))
        with self.assertRaises(ValueError):
            masked_simplex(np.array([1.0, 3.0, 2.0]), np.array([True, True, False]))

    def test_02_observable_continuous_priority_and_equal_fallback(self) -> None:
        decoded = continuous_priority_from_current_state(np.array([2.0, 4.0, 3.0]), np.array([1.0, 2.0, 4.0]), np.array([True, True, False]))
        self.assertTrue(np.allclose(decoded.fraction, [0.2, 0.8, 0.0]))
        fallback = continuous_priority_from_current_state(np.zeros(3), np.array([1.0, 3.0, 4.0]), np.array([True, False, True]))
        self.assertTrue(fallback.used_fixed_equal_fallback)
        self.assertTrue(np.allclose(fallback.fraction, [0.5, 0.0, 0.5]))

    def test_03_capacity_and_queue_bound_payload(self) -> None:
        capacity, payload = allocate_payload(np.array([2.0, 12.0]), np.array([10.0, 6.0]), np.array([0.2, 0.8]), 1.0)
        self.assertTrue(np.all(payload <= capacity))
        self.assertTrue(np.all(payload <= np.array([2.0, 12.0])))
        self.assertAlmostEqual(float(capacity.sum()), 6.8)

    def test_04_shadow_reward_reconstructs_current_environment_exactly(self) -> None:
        rows, _, _ = _scenario_mechanisms(_validation_scenarios(ROOT)[0])
        current = next(row for row in rows if row["mechanism"] == "current_fixed_parallel_1_over_K")
        self.assertAlmostEqual(float(current["current_return_reconstruction_error"]), 0.0, places=9)
        self.assertEqual(int(current["resource_decoder_violations"]), 0)
        self.assertTrue(bool(current["same_movement_continuation"]))

    def test_05_hstep_alignment_reconstructs_immutable_raw_rows(self) -> None:
        rows = _hstep_return_rows(ROOT)
        self.assertEqual(len(rows), 560)
        self.assertTrue(all(bool(row["all_actions_legal"]) for row in rows))
        self.assertTrue(all(bool(row["uses_current_state_only"]) for row in rows))
        self.assertTrue(all(np.isfinite(float(row["H_step_return"])) for row in rows))

    def test_06_original_round13_and_round12_frozen_evidence_is_unchanged(self) -> None:
        manifest_path = ROOT / "round_13/amendments/round13_completion_amendment_20260825_0030/amendment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            _file_hashes(ROOT, ORIGINAL_OUTPUT),
            manifest["original_formal_output_hashes_before"],
        )
        self.assertEqual(sha256(ROOT / "round_13/ROUND_LOCK.json"), manifest["original_round13_lock_sha256_before"])
        self.assertTrue(bool(_round12_integrity(ROOT)["policy_checkpoint_hash_audit"]))

    def test_07_diagnostic_contract_has_no_future_input_parameter(self) -> None:
        parameters = tuple(inspect.signature(continuous_priority_from_current_state).parameters)
        self.assertEqual(parameters, ("queue_bits", "full_bandwidth_rate_bps", "valid_mask"))

    def test_08_generated_sidecars_have_identical_movement_and_no_violations(self) -> None:
        rows, _, _ = _scenario_mechanisms(_validation_scenarios(ROOT)[1])
        self.assertEqual({str(row["mechanism"]) for row in rows}, {
            "current_fixed_parallel_1_over_K",
            "discrete_argmax_queue_times_rate",
            "continuous_masked_simplex_queue_times_rate",
        })
        self.assertTrue(all(bool(row["same_movement_continuation"]) for row in rows))
        self.assertTrue(all(int(row["resource_decoder_violations"]) == 0 for row in rows))
        self.assertTrue(all(np.isfinite(float(row["episode_return"])) for row in rows))


if __name__ == "__main__":
    unittest.main()
