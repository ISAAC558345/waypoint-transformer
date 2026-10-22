"""Formal exact-only Round-36 FAO execution and design-stage diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from wt.round35.level2_env import external_preference_matching

from .harness import audit_legacy_execute, direct_execute_assignment, enumerate_legal_assignments, first_action_assessment, frozen_legacy_movement
from .infrastructure import configure_project_tempdir
from .optimization import all_candidate_utilities, prefix_trajectory
from .optimization_run import _state
from .run import OUTPUT, _sha, _write_json
from .scenarios import CONFIRMATION_FAMILY, DESIGN_FAMILY, build_clean_scenarios, require_clean_manifest


FORMAL_WORKERS = 4
ANCHORS = (0, 8)
HORIZONS = (1, 5, 10)


def _csv(name: str, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise RuntimeError(f"cannot write empty formal CSV: {name}")
    path = OUTPUT / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return path


def _winner(values: np.ndarray, catalogue: np.ndarray, valid: np.ndarray | None = None) -> int:
    if valid is not None:
        if not np.any(valid):
            raise RuntimeError("no feasible candidate exists for the required FAO horizon")
        values = np.where(valid, values, -np.inf)
    best = float(np.max(values))
    tied = np.flatnonzero(np.isclose(values, best, rtol=1.0e-12, atol=1.0e-6))
    cardinality = np.sum(catalogue[tied] >= 0, axis=1)
    tied = tied[cardinality == np.max(cardinality)]
    lexical = np.where(catalogue[tied] < 0, catalogue.shape[1] + catalogue.shape[0], catalogue[tied])
    return int(tied[np.lexsort(tuple(lexical[:, col] for col in range(catalogue.shape[1] - 1, -1, -1)))[0]])


def _assignment_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left, right = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    edge = float(np.mean(left == right))
    return {
        "edge_agreement": edge,
        "normalized_hamming": float(1.0 - edge),
        "per_uav_selected_buoy_agreement": edge,
        "idle_agreement": float(np.mean((left < 0) == (right < 0))),
    }


def _manifest_hashes() -> tuple[str, str]:
    audit = json.loads((OUTPUT / "round36_clean_manifest_audit.json").read_text(encoding="utf-8"))
    design, confirmation = OUTPUT / "round36_design_manifest.json", OUTPUT / "round36_confirmation_manifest.json"
    if _sha(design) != audit["design_manifest_sha256"] or _sha(confirmation) != audit["confirmation_manifest_sha256"]:
        raise RuntimeError("frozen Round-36 manifest hash mismatch")
    return _sha(design), _sha(confirmation)


def run_fao(family: str, label: str) -> dict[str, Any]:
    """Run every legal first assignment once per frozen state, with H10 reuse."""
    if family == CONFIRMATION_FAMILY and not (OUTPUT / "round36_prefinal_freeze.json").is_file():
        raise RuntimeError("confirmation access is forbidden before the design prefinal freeze")
    configure_project_tempdir("runtime")
    scenarios = build_clean_scenarios(family)
    for scenario in scenarios:
        require_clean_manifest(scenario.metadata())
    catalogue = enumerate_legal_assignments(4, 8)
    if catalogue.shape != (3393, 4):
        raise RuntimeError("frozen assignment catalogue changed")
    state_rows: list[dict[str, Any]] = []
    utilities = {h: np.empty((len(scenarios) * len(ANCHORS), len(catalogue)), dtype=np.float64) for h in HORIZONS}
    validity = {h: np.empty((len(scenarios) * len(ANCHORS), len(catalogue)), dtype=bool) for h in HORIZONS}
    selected = {h: np.empty((len(scenarios) * len(ANCHORS),), dtype=np.int32) for h in HORIZONS}
    services_h5 = np.empty((len(scenarios) * len(ANCHORS), 4, 8), dtype=np.float64)
    services_h10 = np.empty_like(services_h5)
    started = perf_counter()
    state_index = 0
    for scenario in scenarios:
        for anchor in ANCHORS:
            anchor_started = perf_counter()
            state = _state(scenario, anchor)
            movement = frozen_legacy_movement(state, scenario.config())
            values = all_candidate_utilities(state, movement, scenario.config(), scenario.eta, workers=FORMAL_WORKERS, executor="process", record_continuation_infeasible=True)
            matrix = np.asarray([row[:3] for row in values], dtype=np.float64)
            masks = np.asarray([row[3:6] for row in values], dtype=bool)
            if matrix.shape != (3393, 3) or not np.all(np.isfinite(matrix)):
                raise RuntimeError("formal FAO candidate result is incomplete or non-finite")
            for column, horizon in enumerate(HORIZONS):
                utilities[horizon][state_index] = matrix[:, column]
                validity[horizon][state_index] = masks[:, column]
                selected[horizon][state_index] = _winner(matrix[:, column], catalogue, masks[:, column])
            chosen = prefix_trajectory(state, movement, catalogue[selected[5][state_index]], scenario.config(), scenario.eta)
            chosen10 = prefix_trajectory(state, movement, catalogue[selected[10][state_index]], scenario.config(), scenario.eta)
            services_h5[state_index] = np.asarray(chosen[5]["service"], dtype=np.float64)
            services_h10[state_index] = np.asarray(chosen10[10]["service"], dtype=np.float64)
            state_rows.append({
                "state_index": state_index, "scenario_id": scenario.scenario_id, "seed": scenario.seed,
                "anchor": anchor, "eta": scenario.eta, "queue_pressure": scenario.queue_pressure,
                "geometry": scenario.geometry, "replicate": scenario.replicate,
                "seconds": perf_counter() - anchor_started,
                **{f"gamma_star_{h}_index": int(selected[h][state_index]) for h in HORIZONS},
                **{f"u{h}_star": float(utilities[h][state_index, selected[h][state_index]]) for h in HORIZONS},
            })
            state_index += 1
    elapsed = perf_counter() - started
    prefix = f"round36_{label}_fao"
    npz_path = OUTPUT / f"{prefix}_candidates.npz"
    np.savez_compressed(
        npz_path, scenario_id=np.asarray([row["scenario_id"] for row in state_rows]), anchor=np.asarray([row["anchor"] for row in state_rows]),
        candidate_index=np.arange(3393, dtype=np.int32), assignments=catalogue,
        U1=utilities[1], U5=utilities[5], U10=utilities[10], valid_H1=validity[1], valid_H5=validity[5], valid_H10=validity[10], gamma_star_1=selected[1], gamma_star_5=selected[5], gamma_star_10=selected[10],
    )
    _csv(f"{prefix}_summary.csv", state_rows)
    np.savez_compressed(OUTPUT / f"round36_{label}_service_targets.npz", scenario_id=np.asarray([row["scenario_id"] for row in state_rows]), anchor=np.asarray([row["anchor"] for row in state_rows]), Phi_service_H5=services_h5, Phi_service_H10=services_h10, gamma_star_5=selected[5], gamma_star_10=selected[10])
    integrity = {
        "schema_version": "round36_fao_integrity_v1", "population": label, "formal_state_count": len(state_rows),
        "candidate_count_per_state": 3393, "candidate_rows": int(len(state_rows) * 3393), "expected_candidate_rows": int(len(state_rows) * 3393),
        "all_finite": bool(all(np.all(np.isfinite(utilities[h])) for h in HORIZONS)), "continuation_infeasible_candidate_count": {str(h): int(np.size(validity[h]) - np.sum(validity[h])) for h in HORIZONS}, "missing_states": 0,
        "selected_winners_present": bool(all(np.all((selected[h] >= 0) & (selected[h] < 3393)) for h in HORIZONS)),
        "prefix_reuse": True, "worker_count": FORMAL_WORKERS, "wall_clock_seconds": elapsed, "passed": True,
    }
    _write_json(f"round36_{label}_fao_integrity.json", integrity)
    return {"rows": state_rows, "utilities": utilities, "validity": validity, "selected": selected, "catalogue": catalogue, "h5_service": services_h5, "h10_service": services_h10, "integrity": integrity, "npz_path": npz_path}


def _legacy_and_tw(state_row: dict[str, Any], catalogue: np.ndarray, service: np.ndarray, oracle_index: int) -> dict[str, Any]:
    scenario = next(item for item in build_clean_scenarios(DESIGN_FAMILY) if item.scenario_id == state_row["scenario_id"])
    state = _state(scenario, int(state_row["anchor"])); move = frozen_legacy_movement(state, scenario.config())
    legacy = audit_legacy_execute(state, move, scenario.config(), interference_overlap_factor=scenario.eta).transition.assignment
    reference = direct_execute_assignment(state, move, catalogue[int(oracle_index)], scenario.config(), interference_overlap_factor=scenario.eta)
    total = np.sum(service, axis=1, keepdims=True)
    phi = np.divide(service, total, out=np.zeros_like(service), where=total > 0.0)
    tw, _, _, _, evaluated = external_preference_matching(state.queue_bits, reference.transition.channel_gains, scenario.config(), np.log(phi + 1.0e-12))
    legacy_u5 = float(first_action_assessment(state, move, legacy, scenario.config(), interference_overlap_factor=scenario.eta)[5]["utility"])
    tw_u5 = float(first_action_assessment(state, move, tw, scenario.config(), interference_overlap_factor=scenario.eta)[5]["utility"])
    return {"legacy": legacy, "tw": tw, "legacy_u5": legacy_u5, "tw_u5": tw_u5, "phi": phi, "decoder_evaluated": int(evaluated)}


def design_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    """Compute preregistered design-only headroom, alignment and ARR diagnostics."""
    rows: list[dict[str, Any]] = []
    headroom: list[dict[str, Any]] = []; h51: list[dict[str, Any]] = []; h105: list[dict[str, Any]] = []; alignment: list[dict[str, Any]] = []; utility: list[dict[str, Any]] = []
    phi = np.empty_like(result["h5_service"])
    for index, row in enumerate(result["rows"]):
        oracle_h1, oracle_h5, oracle_h10 = (result["catalogue"][result["selected"][h][index]] for h in HORIZONS)
        tw = _legacy_and_tw(row, result["catalogue"], result["h5_service"][index], int(result["selected"][5][index]))
        phi[index] = tw["phi"]
        legacy = tw["legacy"]
        base = {key: row[key] for key in ("state_index", "scenario_id", "seed", "anchor", "eta", "queue_pressure", "geometry", "replicate")}
        headroom.append({**base, "legacy_u5": tw["legacy_u5"], "oracle_u5": row["u5_star"], "Delta_assoc_headroom": row["u5_star"] - tw["legacy_u5"]})
        h51.append({**base, **_assignment_metrics(oracle_h5, oracle_h1), "Delta_5_1": row["u5_star"] - result["utilities"][5][index, result["selected"][1][index]]})
        h105.append({**base, **_assignment_metrics(oracle_h10, oracle_h5), "Delta_10_5": row["u10_star"] - result["utilities"][10][index, result["selected"][5][index]]})
        alignment.append({**base, "comparison": "task_waypoint_vs_oracle_h5", **_assignment_metrics(tw["tw"], oracle_h5)})
        alignment.append({**base, "comparison": "legacy_vs_oracle_h5", **_assignment_metrics(legacy, oracle_h5)})
        utility.append({**base, "legacy_u5": tw["legacy_u5"], "task_waypoint_u5": tw["tw_u5"], "oracle_u5": row["u5_star"], "Delta_TW_legacy": tw["tw_u5"] - tw["legacy_u5"], "Gap_oracle": row["u5_star"] - tw["tw_u5"]})
        rows.append({**base, "oracle_h5": oracle_h5.tolist(), "legacy": legacy.tolist(), "task_waypoint_h5": tw["tw"].tolist()})
    _csv("association_oracle_headroom.csv", headroom); _csv("h5_vs_h1_association_planning.csv", h51); _csv("h10_vs_h5_association_planning.csv", h105); _csv("task_waypoint_assignment_alignment.csv", alignment); _csv("task_waypoint_direct_control_utility.csv", utility)
    np.savez_compressed(OUTPUT / "round36_design_service_targets.npz", Phi_service_H5=phi, Phi_service_H10=result["h10_service"], scenario_id=np.asarray([row["scenario_id"] for row in result["rows"]]), anchor=np.asarray([row["anchor"] for row in result["rows"]]))
    (OUTPUT / "task_waypoint_service_target_spec.md").write_text("# H5 Oracle Service Task-Waypoint\n\nPhi_service_H5 is the per-UAV H5 actual payload by buoy, normalized by that UAV's H5 payload. A zero-service UAV receives the zero vector. The frozen association preference mapping is `log(Phi_service_H5 + 1e-12)`.\n", encoding="utf-8")
    _write_json("round36_arr_protocol.json", {"schema_version": "round36_arr_protocol_v1", "frozen_before_design_aggregation": True, "k": 5, "distance": "z_scored_euclidean", "exclude_same_scenario": True, "exclude_temporal_neighbor": True, "epsilon": 1e-12, "minimum_practical_arr": 0.05})
    # kNN ambiguity: nearest five states from different scenarios; exact-assignment disagreement is ambiguity.
    state_x = np.asarray([[row["eta"], row["anchor"], row["seed"], *phi[i].ravel()] for i, row in enumerate(result["rows"])], dtype=float)
    z = (state_x - state_x.mean(axis=0)) / np.where(state_x.std(axis=0) > 1e-12, state_x.std(axis=0), 1.0)
    action = np.asarray([result["catalogue"][result["selected"][5][i]] for i in range(len(rows))])
    arr_rows = []
    for i, row in enumerate(result["rows"]):
        permitted = [j for j, other in enumerate(result["rows"]) if other["scenario_id"] != row["scenario_id"]]
        nearest = sorted(permitted, key=lambda j: float(np.sum((z[i] - z[j]) ** 2)))[:5]
        ambiguity = float(np.mean([np.mean(action[i] != action[j]) for j in nearest]))
        # State+Task-WP holds the target action itself only through an allowed target descriptor; same kNN protocol.
        target_distance = np.asarray([np.mean(np.abs(phi[i] - phi[j])) for j in permitted])
        target_nearest = [permitted[int(v)] for v in np.argsort(target_distance)[:5]]
        ambiguity_tw = float(np.mean([np.mean(action[i] != action[j]) for j in target_nearest]))
        arr_rows.append({"state_index": i, "scenario_id": row["scenario_id"], "Amb_state": ambiguity, "Amb_state_tw": ambiguity_tw, "ARR_assoc": 1.0 - ambiguity_tw / (ambiguity + 1.0e-12)})
    _csv("task_waypoint_association_arr.csv", arr_rows)
    return {"headroom": headroom, "h51": h51, "h105": h105, "alignment": alignment, "utility": utility, "arr": arr_rows}


def write_prefinal_freeze(design: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    design_hash, confirmation_hash = _manifest_hashes()
    artifacts = [design["npz_path"], OUTPUT / "round36_design_fao_integrity.json", OUTPUT / "association_oracle_headroom.csv", OUTPUT / "task_waypoint_association_arr.csv"]
    freeze = {"schema_version": "round36_prefinal_freeze_v1", "design_manifest_sha256": design_hash, "confirmation_manifest_sha256": confirmation_hash, "anchors": list(ANCHORS), "candidate_count": 3393, "worker_count": FORMAL_WORKERS, "design_fao_artifact_sha256": {path.name: _sha(path) for path in artifacts}, "confirmation_access_count_before_freeze": 0, "design_state_count": len(design["rows"]), "round37_allowed": False, "passed": True}
    _write_json("round36_prefinal_freeze.json", freeze)
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("design", "confirmation")); args = parser.parse_args()
    if args.stage == "design":
        result = run_fao(DESIGN_FAMILY, "design")
        diagnostics = design_diagnostics(result)
        freeze = write_prefinal_freeze(result, diagnostics)
        _write_json("round36_design_execution.json", {"status": "complete", "integrity": result["integrity"], "prefinal_freeze": freeze})
    else:
        result = run_fao(CONFIRMATION_FAMILY, "confirmation")
        _write_json("round36_confirmation_execution.json", {"status": "complete", "integrity": result["integrity"]})


if __name__ == "__main__":
    main()
