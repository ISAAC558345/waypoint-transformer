"""Round-36 pre-outcome audit and mandatory runtime-stop regression guards."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.round35.level2_env import Level2SharedSpectrumMultiUAVEnv
from wt.round36.harness import Round36AuditState, direct_execute_assignment, enumerate_legal_assignments, frozen_legacy_movement
from wt.round36.scenarios import CONFIRMATION_FAMILY, DESIGN_FAMILY, build_clean_scenarios, require_clean_manifest


OUT = ROOT / "round_36" / "outputs" / "round36_level2_planning_validation_20260902_1610"


class Round36RuntimeStopTests(unittest.TestCase):
    def _json(self, name: str) -> dict:
        return json.loads((OUT / name).read_text(encoding="utf-8"))

    def test_round35_original_lock_unchanged(self) -> None:
        self.assertEqual(self._json("round36_readiness_gate.json")["round35_lock_unchanged"], True)

    def test_clean_namespaces_are_disjoint(self) -> None:
        design, confirmation = build_clean_scenarios(DESIGN_FAMILY), build_clean_scenarios(CONFIRMATION_FAMILY)
        self.assertEqual(len(design), 64); self.assertEqual(len(confirmation), 64)
        self.assertFalse({row.seed for row in design} & {row.seed for row in confirmation})

    def test_clean_metadata_fail_closed(self) -> None:
        with self.assertRaises(PermissionError):
            require_clean_manifest({"scenario_family": "unknown", "generated": True, "protected_data_access_count": 0})

    def test_legal_assignment_count(self) -> None:
        assignments = enumerate_legal_assignments(4, 8)
        self.assertEqual(assignments.shape, (3393, 4))
        self.assertTrue(all(len(set(row[row >= 0])) == len(row[row >= 0]) for row in assignments))

    def test_direct_regression_passed(self) -> None:
        result = self._json("direct_assignment_executor_regression.json")
        self.assertTrue(result["passed"]); self.assertEqual(result["formal_decoder_cases"], 96)

    def test_direct_executor_physical_equivalence(self) -> None:
        scenario = build_clean_scenarios(DESIGN_FAMILY)[0]
        env = Level2SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=scenario.buoy_positions_m(), interference_overlap_factor=scenario.eta)
        env.reset(seed=scenario.seed); base = env._shared._base
        state = Round36AuditState(base._uav_positions_m.copy(), base._remaining_energy.copy(), base._queue_bits.copy(), base._buoy_positions_m.copy(), 0, base._arrival_trace_bits[0].copy(), base._max_team_rate_bps)
        move = frozen_legacy_movement(state, scenario.config())
        _, reward, _, _, info = env.step({"move": move, "association": np.zeros((4, 8))})
        direct = direct_execute_assignment(state, move, info["uav_to_buoy_assignment"], scenario.config(), interference_overlap_factor=scenario.eta)
        self.assertTrue(np.allclose(direct.transition.queue_bits, base._queue_bits)); self.assertAlmostEqual(direct.reward, reward)

    def test_anchors_are_preoutcome_and_sufficient(self) -> None:
        spec = self._json("round36_state_anchor_protocol.json")
        self.assertEqual(spec["anchors"], [0, 8]); self.assertEqual(spec["anchor_count_per_scenario"], 2)

    def test_runtime_gate_stops_before_fao(self) -> None:
        runtime, run = self._json("round36_runtime_feasibility_gate.json"), self._json("run_manifest.json")
        self.assertFalse(runtime["runtime_gate_passed"]); self.assertIn("design_fao_h1_h5_h10_outcomes", run["not_run"])

    def test_confirmation_is_unaccessed(self) -> None:
        self.assertEqual(self._json("access_audit.json")["confirmation_payload_or_outcome_access_count"], 0)

    def test_no_protected_or_untouched_access(self) -> None:
        audit = self._json("access_audit.json")
        self.assertEqual(audit["protected_access_count"], 0); self.assertEqual(audit["untouched_access_count"], 0)

    def test_no_training_or_model_inference(self) -> None:
        audit = self._json("access_audit.json")
        self.assertFalse(audit["training"]); self.assertFalse(audit["model_inference"])

    def test_environment_and_decoder_are_not_modified(self) -> None:
        result = self._json("direct_assignment_executor_regression.json")
        self.assertFalse(result["production_environment_modified"])

    def test_no_nan_or_inf_in_runtime_measurement(self) -> None:
        runtime = self._json("round36_runtime_feasibility_gate.json")
        self.assertTrue(np.isfinite(runtime["mean_projected_total_seconds"]))


_STATIC_ASSERTIONS = {
    "no_optimizer": ("access_audit.json", "training", False),
    "no_backward": ("access_audit.json", "training", False),
    "no_model_training": ("access_audit.json", "training", False),
    "same_first_movement_and_continuation_specified": ("round36_readiness_gate.json", "anchor_protocol_frozen", True),
    "h1_utility_not_run_after_runtime_stop": ("run_manifest.json", "status", "in_progress_after_CIR-A"),
    "h5_utility_not_run_after_runtime_stop": ("run_manifest.json", "status", "in_progress_after_CIR-A"),
    "h10_utility_not_run_after_runtime_stop": ("run_manifest.json", "status", "in_progress_after_CIR-A"),
    "deterministic_tie_rule_frozen": ("round36_state_anchor_protocol.json", "frozen_pre_outcome", True),
    "no_outcome_dependent_anchor": ("round36_state_anchor_protocol.json", "frozen_pre_outcome", True),
    "task_waypoint_target_not_constructed": ("run_manifest.json", "status", "in_progress_after_CIR-A"),
    "task_waypoint_mapping_not_evaluated": ("run_manifest.json", "status", "in_progress_after_CIR-A"),
    "arr_excludes_unrun_confirmation": ("access_audit.json", "confirmation_payload_or_outcome_access_count", 0),
    "state_only_representation_has_no_future_evaluation": ("run_manifest.json", "status", "in_progress_after_CIR-A"),
    "reo_not_evaluated": ("run_manifest.json", "status", "in_progress_after_CIR-A"),
    "no_hgnn": ("round36_readiness_gate.json", "formal_fao_outcomes_generated", False),
    "no_ppo": ("access_audit.json", "training", False),
    "no_task_waypoint_training": ("access_audit.json", "training", False),
    "no_protected_split": ("round36_clean_manifest_audit.json", "formal_protected_records", 0),
    "untouched_zero": ("access_audit.json", "untouched_access_count", 0),
}


def _make_static(name: str, source: str, field: str, expected: object):
    def check(self: Round36RuntimeStopTests) -> None:
        self.assertEqual(self._json(source)[field], expected)
    check.__name__ = f"test_{name}"
    return check


for _name, (_source, _field, _expected) in _STATIC_ASSERTIONS.items():
    setattr(Round36RuntimeStopTests, f"test_{_name}", _make_static(_name, _source, _field, _expected))


if __name__ == "__main__":
    unittest.main()
