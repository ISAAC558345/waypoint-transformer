"""Registered non-learning execution stages for the Round-36 audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from wt.round35.level2_env import Level2SharedSpectrumMultiUAVEnv

from .harness import Round36AuditState, audit_legacy_assignment, direct_execute_assignment, frozen_legacy_movement, formal_legacy_assignment
from .scenarios import CONFIRMATION_FAMILY, DESIGN_FAMILY, build_clean_scenarios, require_clean_manifest


ROOT = Path(__file__).resolve().parents[3]
ROUND = ROOT / "round_36"
OUTPUT = ROUND / "outputs" / "round36_level2_planning_validation_20260902_1610"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(name: str, value: Any) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def clean_manifest_preflight() -> dict[str, Any]:
    """Freeze the two generated metadata manifests before any state payload."""
    designs = build_clean_scenarios(DESIGN_FAMILY)
    confirmations = build_clean_scenarios(CONFIRMATION_FAMILY)
    records = [item.metadata() for item in designs + confirmations]
    for metadata in records:
        require_clean_manifest(metadata)
    if {item.seed for item in designs} & {item.seed for item in confirmations}:
        raise RuntimeError("Round-36 clean namespaces have overlapping seeds")
    design_path = _write_json("round36_design_manifest.json", {
        "schema_version": "round36_clean_manifest_v1", "scenario_family": DESIGN_FAMILY,
        "outcome_accessed": False, "generated": True, "protected_data_access_count": 0,
        "scenarios": [item.metadata() for item in designs],
    })
    confirmation_path = _write_json("round36_confirmation_manifest.json", {
        "schema_version": "round36_clean_manifest_v1", "scenario_family": CONFIRMATION_FAMILY,
        "outcome_accessed": False, "generated": True, "protected_data_access_count": 0,
        "scenarios": [item.metadata() for item in confirmations],
    })
    audit = {
        "schema_version": "round36_clean_manifest_audit_v1",
        "metadata_only_preflight": True,
        "allowlist": [DESIGN_FAMILY, CONFIRMATION_FAMILY, "clean_engineering_metadata"],
        "design_records": len(designs), "confirmation_records": len(confirmations),
        "protected_records_decoded": 0, "unknown_records": 0, "formal_protected_records": 0,
        "untouched_records_decoded": 0, "seed_overlap": False,
        "design_manifest_sha256": _sha(design_path),
        "confirmation_manifest_sha256": _sha(confirmation_path),
        "passed": True,
    }
    _write_json("round36_clean_manifest_audit.json", audit)
    return audit


def _audit_state(env: Level2SharedSpectrumMultiUAVEnv) -> Round36AuditState:
    base = env._shared._base
    return Round36AuditState(
        base._uav_positions_m.copy(), base._remaining_energy.copy(), base._queue_bits.copy(),
        base._buoy_positions_m.copy(), int(base._time_step), base._arrival_trace_bits[base._time_step].copy(),
        float(base._max_team_rate_bps),
    )


def _allclose(left: Any, right: Any, tolerance: float = 1.0e-9) -> bool:
    return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=0.0, atol=tolerance))


def direct_executor_regression() -> dict[str, Any]:
    """Formal decoder path vs audit direct execution, prior to outcomes."""
    rng = np.random.default_rng(360360)
    fixtures: list[dict[str, Any]] = []
    adapter_checks: list[bool] = []
    direct_checks: list[bool] = []
    for scenario in build_clean_scenarios(DESIGN_FAMILY)[:8]:
        require_clean_manifest(scenario.metadata())
        config, buoys = scenario.config(), scenario.buoy_positions_m()
        for trial in range(12):
            env = Level2SharedSpectrumMultiUAVEnv(config, buoy_positions_m=buoys, interference_overlap_factor=scenario.eta)
            env.reset(seed=scenario.seed)
            state = _audit_state(env)
            movement = frozen_legacy_movement(state, config)
            logits = np.zeros((4, 8), dtype=np.float64) if trial == 0 else rng.normal(size=(4, 8))
            # The production decoder is the sole authority for Gamma here.
            _, reward, terminated, truncated, info = env.step({"move": movement, "association": logits})
            assignment = np.asarray(info["uav_to_buoy_assignment"], dtype=np.int64)
            direct = direct_execute_assignment(state, movement, assignment, config, interference_overlap_factor=scenario.eta)
            base = env._shared._base
            checks = {
                "assignment": np.array_equal(direct.transition.assignment, assignment),
                "service": _allclose(direct.transition.payload_bits, info["transmitted_payload_bits"]),
                "rate": _allclose(direct.transition.rates_bps, info["uav_buoy_rates_bps"]),
                "queue": _allclose(direct.transition.queue_bits, base._queue_bits),
                "interference": _allclose(direct.transition.interference_power_w, info["cross_interference_power_w"]),
                "reward": abs(direct.reward - float(reward)) <= 1.0e-9,
                "next_state": _allclose(direct.next_state.positions_m, base._uav_positions_m) and _allclose(direct.next_state.remaining_energy, base._remaining_energy),
                "termination": direct.terminated == bool(terminated) and direct.truncated == bool(truncated),
            }
            direct_checks.append(all(checks.values()))
            # The frozen legacy adapter is separately checked on the exact same moved state.
            actual, _, _ = formal_legacy_assignment(state.queue_bits, direct.transition.legacy_rates_bps, config.queue_capacity_bits)
            adapted, _, _ = audit_legacy_assignment(state.queue_bits, direct.transition.legacy_rates_bps, config.queue_capacity_bits)
            adapter_checks.append(np.array_equal(actual, adapted))
            fixtures.append({"scenario_id": scenario.scenario_id, "trial": trial, "checks": checks, "adapter_equal": adapter_checks[-1]})
    result = {
        "schema_version": "round36_direct_executor_regression_v1",
        "formal_decoder_cases": len(fixtures), "systematic_zero_logit_cases": 8,
        "random_logit_cases": len(fixtures) - 8,
        "formal_assignment_space": 3393,
        "direct_executor_physical_equivalence": bool(all(direct_checks)),
        "audit_legacy_adapter_equivalence": bool(all(adapter_checks)),
        "production_environment_modified": False,
        "fixtures": fixtures,
        "passed": bool(all(direct_checks) and all(adapter_checks)),
    }
    _write_json("direct_assignment_executor_regression.json", result)
    if not result["passed"]:
        raise RuntimeError("Round-36 direct executor equivalence regression failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "direct-regression"))
    args = parser.parse_args()
    if args.stage == "preflight":
        print(json.dumps(clean_manifest_preflight(), sort_keys=True))
    else:
        print(json.dumps(direct_executor_regression(), sort_keys=True))


if __name__ == "__main__":
    main()
