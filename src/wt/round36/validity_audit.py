"""Post-design continuation-validity audit required before any Round-36 gate."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .harness import audit_legacy_execute, direct_execute_assignment, enumerate_legal_assignments, frozen_legacy_movement
from .optimization_run import _state
from .run import OUTPUT, _write_json
from .scenarios import DESIGN_FAMILY, build_clean_scenarios


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv(name: str, rows: list[dict[str, Any]]) -> None:
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _trace(state, movement, gamma, config, eta: float) -> tuple[tuple[bool, bool, bool], int | None, str | None]:
    current = direct_execute_assignment(state, movement, gamma, config, interference_overlap_factor=eta)
    valid = {1: True, 5: False, 10: False}
    for step in range(2, 11):
        try:
            current = audit_legacy_execute(current.next_state, frozen_legacy_movement(current.next_state, config), config, interference_overlap_factor=eta)
        except RuntimeError as error:
            if "hard UAV separation projection did not converge" not in str(error):
                raise
            return (valid[1], valid[5], valid[10]), step, "continuation_invalid_under_frozen_separation_projector"
        if step == 5: valid[5] = True
        if step == 10: valid[10] = True
    return (True, True, True), None, None


def audit_design_validity() -> dict[str, Any]:
    data = np.load(OUTPUT / "round36_design_fao_candidates.npz", allow_pickle=False)
    catalogue = np.asarray(data["assignments"], dtype=np.int64)
    masks = {1: np.asarray(data["valid_H1"], dtype=bool), 5: np.asarray(data["valid_H5"], dtype=bool), 10: np.asarray(data["valid_H10"], dtype=bool)}
    summary = _rows(OUTPUT / "round36_design_fao_summary.csv")
    if any(masks[h].shape != (128, 3393) for h in masks): raise RuntimeError("invalid candidate-mask shape")
    monotonic = bool(np.all(~masks[10] | masks[5]) and np.all(~masks[5] | masks[1]))
    if not monotonic: raise RuntimeError("continuation validity monotonicity failed")
    provenance = []
    scenarios = {item.scenario_id: item for item in build_clean_scenarios(DESIGN_FAMILY)}
    for state_index, row in enumerate(summary):
        scenario = scenarios[row["scenario_id"]]; state = _state(scenario, int(row["anchor"])); move = frozen_legacy_movement(state, scenario.config())
        invalid = np.flatnonzero(~masks[10][state_index])
        for candidate_index in invalid:
            valid, step, reason = _trace(state, move, catalogue[candidate_index], scenario.config(), scenario.eta)
            if tuple(masks[h][state_index, candidate_index] for h in (1, 5, 10)) != valid:
                raise RuntimeError("validity replay does not match pre-ranking mask")
            if step is not None:
                assignment = catalogue[candidate_index]
                provenance.append({"state_index": state_index, "scenario_id": scenario.scenario_id, "anchor": int(row["anchor"]), "candidate_index": int(candidate_index), "valid_H1": valid[0], "valid_H5": valid[1], "valid_H10": valid[2], "failure_step": step, "failure_reason": reason, "failure_component": "frozen_separation_projector", "idle_uav_count": int(np.sum(assignment < 0)), "assignment": assignment.tolist(), "eta": scenario.eta, "queue_pressure": scenario.queue_pressure, "geometry": scenario.geometry})
    np.savez_compressed(OUTPUT / "round36_candidate_validity_mask.npz", scenario_id=data["scenario_id"], anchor=data["anchor"], candidate_index=data["candidate_index"], valid_H1=masks[1], valid_H5=masks[5], valid_H10=masks[10])
    coverage = []
    for state_index, row in enumerate(summary):
        coverage.append({"state_index": state_index, "scenario_id": row["scenario_id"], "anchor": row["anchor"], **{f"N_valid_H{h}": int(masks[h][state_index].sum()) for h in masks}, **{f"coverage_H{h}": float(masks[h][state_index].mean()) for h in masks}})
    _csv("fao_valid_candidate_coverage.csv", coverage)
    _csv("round36_candidate_validity_summary.csv", coverage)
    _write_json("continuation_validity_monotonicity_audit.json", {"passed": monotonic, "rule": "Valid_H10=>Valid_H5=>Valid_H1", "candidate_rows": 128 * 3393})
    if provenance: _csv("continuation_failure_structure.csv", provenance)
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in provenance:
        for key in (("overall", "all"), ("eta", row["eta"]), ("queue_pressure", row["queue_pressure"]), ("geometry", row["geometry"]), ("anchor", row["anchor"]), ("failure_step", row["failure_step"])): grouped[key].append(row)
    rates = [{"dimension": dim, "value": value, "invalid_H1_rate": float(1.0 - masks[1].mean()) if dim == "overall" else "", "invalid_H5_rate": float(1.0 - masks[5].mean()) if dim == "overall" else "", "invalid_H10_rate": float(1.0 - masks[10].mean()) if dim == "overall" else "", "failure_count": len(items)} for (dim, value), items in grouped.items()]
    _csv("continuation_failure_rate_audit.csv", rates)
    zero = any(any(row[f"N_valid_H{h}"] == 0 for h in (1, 5, 10)) for row in coverage)
    h10_rate = float(1.0 - masks[10].mean())
    gate = "CVG-C" if zero or h10_rate > 0.25 else ("CVG-B" if h10_rate > 0.0 else "CVG-A")
    _write_json("continuation_validity_gate.json", {"gate": gate, "continuation_validity_acceptable": gate in {"CVG-A", "CVG-B"}, "monotonicity_passed": True, "zero_valid_state_horizon": zero, "invalid_H10_rate": h10_rate, "failure_term": "continuation_invalid_under_frozen_separation_projector", "possible_physical_infeasibility": "not established"})
    return {"gate": gate, "provenance_count": len(provenance), "invalid_h10_rate": h10_rate}


if __name__ == "__main__":
    print(json.dumps(audit_design_validity(), sort_keys=True))
