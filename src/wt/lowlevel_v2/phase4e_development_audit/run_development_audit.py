"""Inventory scenario roles. No model fit and no threshold search."""
from __future__ import annotations

import csv
import json

import numpy as np

from wt.lowlevel_v2.data_registry import scan_historical_seeds
from wt.lowlevel_v2.htw_phase4b_b2_shadow.protocol import B2_FEATURE_KEYS, ET
from wt.lowlevel_v2.phase4e_development_audit.protocol import OUTPUT, ROOT, TAU

LLV2 = ROOT / "src" / "wt" / "lowlevel_v2"
REGISTRY = LLV2 / "outputs" / "phase0" / "LLV2_DATA_REGISTRY.json"
PHASE2 = LLV2 / "phase2" / "LLV2_PHASE2_VALIDATION_MANIFEST.json"
PHASE3 = LLV2 / "phase3" / "LLV2_PHASE3_VALIDATION_MANIFEST.json"
HOLDOUT = LLV2 / "phase4d7b_independent_holdout_generation" / "outputs" / "deployment_holdout_manifest.json"
P4A = LLV2 / "htw_phase4a_hindsight_credit" / "outputs" / "phase4a_hindsight_credit.csv"
SHADOW = LLV2 / "htw_phase4b_b2_shadow" / "outputs" / "b2_shadow_decisions.csv"
P4C = LLV2 / "phase4c_independent_validation" / "outputs" / "independent_decisions.csv"
P4E = LLV2 / "phase4e_deployment_verification" / "outputs" / "deployment_verification_results.csv"
EPS = 1e-12


def _dump(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path, rows: list[dict]) -> None:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _ids(start: int, end: int) -> list[int]:
    return list(range(start, end + 1))


def _load_seeds(path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(x) for x in payload["scenario_seeds"]]


def _row(scenario: int, pool: str, role: str, flags: dict, reason: str, source: str) -> dict:
    base = {
        "used_for_B2_fit": 0,
        "used_for_feature_selection": 0,
        "used_for_hyperparameter_selection": 0,
        "used_for_tau_selection": 0,
        "used_for_gate_selection": 0,
        "used_for_criterion_selection": 0,
        "used_for_Phase4C_validation": 0,
        "used_for_Phase4E_holdout": 0,
        "eligible_for_future_development": 0,
    }
    base.update(flags)
    return {
        "scenario_id": int(scenario),
        "pool": pool,
        "historical_role": role,
        **base,
        "reason": reason,
        "source_file": source,
    }


def _corr(path, margin_key: str, delta_key: str) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    margin = np.array([float(row[margin_key]) for row in rows], dtype=np.float64)
    delta = np.array([float(row[delta_key]) for row in rows], dtype=np.float64)
    released = margin >= TAU
    negative = delta < -EPS
    return {
        "n": int(len(rows)),
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        "pearson_margin_delta_R": float(np.corrcoef(margin, delta)[0, 1]),
        "spearman_margin_delta_R": _spearman(margin, delta),
        "n_margin_ge_tau": int(np.sum(released)),
        "n_negative_delta_R_given_margin_ge_tau": int(np.sum(released & negative)),
        "P_delta_R_negative_given_margin_ge_tau": float(np.mean(negative[released])) if np.any(released) else float("nan"),
        "used_to_fit_threshold": False,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    train = [int(x) for x in registry["LLV2_train"]]
    design = [int(x) for x in registry["LLV2_design"]]
    phase2 = _load_seeds(PHASE2)
    phase3 = _load_seeds(PHASE3)
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    consumed = [int(x) for x in holdout["batch_1_scenarios"] + holdout["batch_2_scenarios"]]
    if train != _ids(220000, 220063) or design != _ids(221000, 221015):
        raise RuntimeError("LLV2 registry IDs differ from the registered ranges")
    if phase2 != _ids(222000, 222063) or phase3 != _ids(223000, 223063):
        raise RuntimeError("phase2/phase3 manifests differ from the registered ranges")
    if consumed != _ids(230000, 230127):
        raise RuntimeError("holdout manifest IDs differ from 230000-230127")
    rows = []
    for scenario in train:
        rows.append(
            _row(
                scenario,
                "LLV2_train",
                "VALIDATION",
                {"used_for_Phase4C_validation": 1},
                "Low-Level train pool, then Phase-4C/4D B2 validation. Not a fresh development pool.",
                str(REGISTRY.relative_to(ROOT)).replace("\\", "/"),
            )
        )
    for scenario in design:
        rows.append(
            _row(
                scenario,
                "LLV2_design",
                "DESIGN",
                {
                    "used_for_B2_fit": 1,
                    "used_for_feature_selection": 1,
                    "used_for_tau_selection": 1,
                    "used_for_gate_selection": 1,
                },
                "EXEC labels for the frozen ExtraTrees fit, Phase-3E/4B feature ablation, GroupKFold, and the shadow mean dU_hat that locked tau. Hyperparameters were pre-specified, not searched.",
                "src/wt/lowlevel_v2/htw_phase4b_b2_validation/outputs/HTW_Phase4B_B2_IndependentValidation_Audit_CN.md",
            )
        )
    for scenario in phase2:
        rows.append(
            _row(
                scenario,
                "phase2_val",
                "HISTORICAL_TEST",
                {},
                "Consumed Phase-2 validation. refuse_compat blocks it. No B2-protocol labels.",
                str(PHASE2.relative_to(ROOT)).replace("\\", "/"),
            )
        )
    for scenario in phase3:
        rows.append(
            _row(
                scenario,
                "phase3_val",
                "HISTORICAL_TEST",
                {},
                "Consumed Phase-3 validation. refuse_compat blocks it. No B2-protocol labels.",
                str(PHASE3.relative_to(ROOT)).replace("\\", "/"),
            )
        )
    historical = (
        ("phase104_independent", 104000, 104063, "src/wt/lowlevel_v2/protocol.py"),
        ("phase105_independent", 105000, 105063, "src/wt/lowlevel_v2/protocol.py"),
        ("phase106_final_test", 106000, 106063, "src/wt/lowlevel_v2/outputs/phase0/LLV2_DATA_REGISTRY.json"),
    )
    for pool, start, end, source in historical:
        for scenario in _ids(start, end):
            rows.append(
                _row(
                    scenario,
                    pool,
                    "HISTORICAL_TEST",
                    {},
                    "Historical Phase-10 evaluation pool. refuse_compat blocks it. No B2 first-hold labels.",
                    source,
                )
            )
    for scenario in consumed:
        rows.append(
            _row(
                scenario,
                "deployment_holdout",
                "DEPLOYMENT_HOLDOUT_CONSUMED",
                {"used_for_Phase4E_holdout": 1},
                "Consumed by Phase-4E Candidate-C verification. Permanently closed to training, selection, tuning, and model comparison.",
                str(HOLDOUT.relative_to(ROOT)).replace("\\", "/"),
            )
        )
    named = {int(row["scenario_id"]) for row in rows}
    other = sorted(scan_historical_seeds() - named)
    _csv(OUTPUT / "phase4e_remediation_data_usage_matrix.csv", rows)
    consumed_doc = {
        "status": "DEPLOYMENT_HOLDOUT_CONSUMED",
        "scenario_ids": consumed,
        "n": len(consumed),
        "closed_to": [
            "B2 training",
            "B2 retraining",
            "feature selection",
            "hyperparameter selection",
            "tau selection",
            "Gate-1 selection",
            "criterion selection",
            "candidate design",
            "model comparison",
            "threshold tuning",
            "planner development",
        ],
        "source_manifest": str(HOLDOUT.relative_to(ROOT)).replace("\\", "/"),
    }
    _dump(OUTPUT / "deployment_holdout_consumed_registry.json", consumed_doc)
    design_corr = _corr(SHADOW, "dU_hat", "delta_R")
    design_corr["role"] = "in_sample_design_diagnostic"
    phase4c_corr = _corr(P4C, "dU_hat", "delta_R")
    phase4c_corr["role"] = "consumed_phase4c_validation_diagnostic"
    external = _corr(P4E, "margin", "delta_R")
    external["role"] = "EXTERNAL_RECORD_NOT_USED_FOR_SELECTION"
    external["note"] = "Phase-4E delta_R is the realized Gate-1 outcome. Below tau it is zero because LOCAL was selected. This record is not a tuning set."
    lineage = {
        "model": "frozen Phase-4C B2",
        "fit_scenario_ids": design,
        "fit_family": "EXEC",
        "fit_rows_per_scenario": 24,
        "target": "C_abs = U_traj - U_single",
        "features": list(B2_FEATURE_KEYS),
        "extratrees": ET,
        "random_seed": 0,
        "groupkfold": {"n_splits": 5, "group": "scenario_id", "validation_fold": [], "hyperparameter_search": False},
        "feature_comparison_scenario_ids": design,
        "tau": TAU,
        "tau_source": "Phase-4B shadow changed-state mean dU_hat on LLV2_design",
        "criterion_source": "HUMAN_SPECIFICATION Candidate-C; no scenario pool selected C1-C8",
        "saved_model_artifact": None,
        "reconstruction": "fit ExtraTrees on Phase-4A EXEC rows for 221000-221015",
        "label_file": str(P4A.relative_to(ROOT)).replace("\\", "/"),
        "confirmation_pool": registry.get("LLV2_confirmation"),
        "final_untouched_pool": registry.get("LLV2_final_untouched"),
    }
    eligible = {
        "status": "NO_ELIGIBLE_DEVELOPMENT_DATA",
        "remediation_status": "B2_REMEDIATION_BLOCKED_BY_NO_ELIGIBLE_DEVELOPMENT_DATA",
        "eligible_scenario_ids": [],
        "development_pool": None,
        "validation_pool": None,
        "consumed_holdout_pool": {"ids": "230000-230127", "n": 128, "role": "DEPLOYMENT_HOLDOUT_CONSUMED"},
        "other_blocked_historical_seed_count": len(other),
        "other_blocked_not_a_development_pool": True,
        "absent_pools": ["LLV2_confirmation", "LLV2_final_untouched"],
    }
    audit = {
        "status": "DEPLOYMENT_VERIFICATION_FAILED",
        "deployment_verified": False,
        "planner_ready": False,
        "b2_frozen": True,
        "tau_frozen": TAU,
        "gate1_frozen": True,
        "criterion_frozen": "CONSERVATIVE_WITH_FALLBACK_V1",
        "matrix_rows": len(rows),
        "eligible_count": 0,
        "lineage": lineage,
        "eligible": eligible,
        "margin_diagnostics": {
            "design": design_corr,
            "phase4c_validation": phase4c_corr,
            "phase4e_external_record": external,
        },
        "failure_direction": {
            "A_candidate_ranking_error": "EVIDENCE_FROM_COMPLETED_PHASE4E_ATTRIBUTION",
            "B_U_hat_calibration_error": "NOT ESTABLISHED",
            "C_geometry_long_horizon_mismatch": "NOT ESTABLISHED",
            "D_margin_delta_R_relation": "SEE_DIAGNOSTICS_NOT_A_NEW_THRESHOLD",
            "E_training_coverage": "NOT ESTABLISHED",
            "F_feature_insufficiency": "NOT ESTABLISHED",
        },
        "deployment_rerun": False,
        "model_modified": False,
    }
    _dump(OUTPUT / "phase4e_model_lineage_audit.json", lineage)
    _dump(OUTPUT / "phase4e_eligible_development_pool.json", eligible)
    _dump(OUTPUT / "phase4e_remediation_audit.json", audit)
    print(json.dumps({"rows": len(rows), "eligible": 0, "other_blocked": len(other), "design": design_corr, "phase4c": phase4c_corr, "external": external}, indent=2))


if __name__ == "__main__":
    main()
