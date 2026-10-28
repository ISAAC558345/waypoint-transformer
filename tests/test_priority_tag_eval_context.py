"""Guards for the explicit evaluation-context labels and gate."""
from __future__ import annotations

import unittest
from pathlib import Path

from wt.benchmark_v2.phase6i_priority_tag_eval_context.compare import choose_gate
from wt.benchmark_v2.phase6i_priority_tag_eval_context.context import build_manifest, manifest_is_explicit
from wt.benchmark_v2.phase6i_priority_tag_eval_context.protocol import (
    ACCOUNTING_TOLERANCE_BITS,
    HASHED_RELATIVE,
    PATH_TOLERANCE_M,
)


class EvalContextReproTest(unittest.TestCase):
    def _manifest(self) -> dict:
        return build_manifest(
            versions={"python": "test", "numpy": "test", "torch": "test"},
            commit="abc",
            code_hashes={relative: "hash" for relative in HASHED_RELATIVE},
        )

    def test_historical_seed_is_not_labeled_as_the_eval_protocol(self) -> None:
        manifest = self._manifest()
        self.assertTrue(manifest_is_explicit(manifest))
        self.assertEqual(manifest["reported_model_seed"], 0)
        self.assertEqual(manifest["env_reset_seed"], 321101)
        self.assertEqual(manifest["assignment_rng_seed"], 7_321_101)
        self.assertEqual(manifest["global_rng"]["applied_seed"], 2)
        self.assertEqual(manifest["global_rng"]["role"], "historical_inherited_process_seed")
        self.assertIsNone(manifest["global_rng"]["canonical_explicit_seed"])
        self.assertNotEqual(manifest["global_rng"]["role"], "eval_protocol_seed")

    def test_thresholds_stay_fixed(self) -> None:
        self.assertEqual(ACCOUNTING_TOLERANCE_BITS, 1.0)
        self.assertEqual(PATH_TOLERANCE_M, 1.0e-9)

    def test_gate_separates_repeatability_from_historical_match(self) -> None:
        self.assertEqual(
            choose_gate(
                hashes_ok=True,
                split_ok=True,
                manifest_ok=True,
                processes_consistent=True,
                both_match_saved=True,
                ledger_ok_both=True,
            ),
            "EVAL_CONTEXT_REPRO_PASS",
        )
        self.assertEqual(
            choose_gate(
                hashes_ok=True,
                split_ok=True,
                manifest_ok=True,
                processes_consistent=True,
                both_match_saved=False,
                ledger_ok_both=True,
            ),
            "EVAL_CONTEXT_REPRO_INCONCLUSIVE",
        )
        self.assertEqual(
            choose_gate(
                hashes_ok=True,
                split_ok=True,
                manifest_ok=True,
                processes_consistent=False,
                both_match_saved=True,
                ledger_ok_both=True,
            ),
            "EVAL_CONTEXT_REPRO_BLOCKED",
        )

    def test_package_source_has_no_held_out_id_or_decision_call(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_eval_context"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("322048", text)
        self.assertNotIn("322064", text)


if __name__ == "__main__":
    unittest.main()
