"""Write the M0 artifacts. One analytical run, no policy load."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from wt.ocean_observation_m0.accounting import (
    build_cases,
    case_catalog,
    conclude,
    link_properties,
    propulsion_facts,
    wind_illustration,
)
from wt.ocean_observation_m0.codec import encode_product, packet_sizes
from wt.ocean_observation_m0.information import four_timelines, worked_product
from wt.ocean_observation_m0.interfaces import interface_draft
from wt.ocean_observation_m0.parameters import (
    ADOPTED,
    DESIGN_BASELINE,
    E19_RESULT_COMMIT,
    FORMAL_CHECKPOINT,
    FORMAL_SOURCE_COMMIT,
    HANDOFF_COMMIT,
    PERFORMANCE_PASS,
    RAW_RECORD_BITS,
    RELEASE_ELIGIBLE,
    assert_registered_layouts,
    parameter_records,
)
from wt.ocean_observation_m0.report_cn import render_report

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "ocean_observation_m0"
REPORT = ROOT / "docs" / "research" / "Ocean_Observation_M0_Report_CN.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _git(args: list) -> str:
    proc = subprocess.run(
        ["git", "-c", "core.longpaths=true", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _write(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_study() -> dict:
    assert_registered_layouts()
    facts = propulsion_facts()
    parameters = parameter_records(facts)
    sizes = packet_sizes()
    catalog = case_catalog()
    cases = build_cases()
    if len(cases) != 24:
        raise RuntimeError("the analytical run must stay at 24 cases")
    decision = conclude(cases, parameters)
    example = worked_product((500.0, 500.0), (0.0, 0.0))
    blob = encode_product(
        1,
        1,
        example["window_start_s"],
        example["window_end_s"],
        example["mean_x_m"],
        example["mean_y_m"],
        example["mean_hs_m"],
        example["measurement_variance_m2"],
        0,
        example["n_samples"],
    )
    inputs = {
        "solution": "docs/research/Ocean_Observation_Solution_CN.md",
        "instructions": "docs/research/Ocean_Observation_M0_Instructions_CN.md",
        "contract": "docs/research/Ocean_Observation_M0_Contract.json",
        "old_environment_not_used_as_link": "configs/environment/l3b.json",
        "formal_checkpoint": FORMAL_CHECKPOINT,
    }
    input_hashes = {key: _sha256(ROOT / relative) for key, relative in inputs.items()}
    manifest = {
        "design_baseline": DESIGN_BASELINE,
        "formal_source_commit": FORMAL_SOURCE_COMMIT,
        "handoff_commit": HANDOFF_COMMIT,
        "e19_result_commit": E19_RESULT_COMMIT,
        "code_commit": _git(["rev-parse", "HEAD"]),
        "formal_checkpoint_sha256": input_hashes["formal_checkpoint"],
        "input_hashes": input_hashes,
        "case_manifest_sha256": hashlib.sha256(_canonical(catalog).encode("utf-8")).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": "Python standard library only",
        "training_updates": 0,
        "policy_model_loads": 0,
        "policy_inferences": 0,
        "new_environment_episode_steps": 0,
        "historical_holdout_reruns": 0,
        "large_external_dataset_downloads": 0,
        "analytical_cases": 24,
        "analytical_runs": 1,
        "performance_pass": PERFORMANCE_PASS,
        "adopted": ADOPTED,
        "release_eligible": RELEASE_ELIGIBLE,
    }
    return {
        "schema_version": 1,
        "conclusion": decision,
        "facts": facts,
        "parameters": parameters,
        "packets": sizes,
        "raw_record_bits": RAW_RECORD_BITS,
        "links": link_properties(sizes),
        "wind_illustration": wind_illustration(facts["cruise_airspeed_m_s"]),
        "observation_example": example,
        "timelines": four_timelines(blob),
        "interfaces": interface_draft(),
        "case_manifest": catalog,
        "cases": cases,
        "manifest": manifest,
    }


def write_outputs(study: dict, verification: dict | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _write(OUT / "study.json", study)
    _write(OUT / "parameter_table.json", study["parameters"])
    _csv(OUT / "parameter_table.csv", [_flat_parameter(row) for row in study["parameters"]])
    _write(OUT / "packet_sizes.json", study["packets"])
    _write(OUT / "link_budget.json", study["links"])
    _write(OUT / "case_manifest.json", study["case_manifest"])
    _write(OUT / "case_results.json", study["cases"])
    _csv(OUT / "case_results.csv", [_flat_case(case) for case in study["cases"]])
    _write(OUT / "observation_example.json", study["observation_example"])
    _write(OUT / "timelines.json", study["timelines"])
    _write(OUT / "coverage_drift.json", [_coverage_row(case) for case in study["cases"]])
    _write(OUT / "interface_draft.json", study["interfaces"])
    _write(OUT / "wind_illustration.json", study["wind_illustration"])
    _write(OUT / "run_manifest.json", study["manifest"])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_report(study, verification), encoding="utf-8")


def _flat_parameter(row: dict) -> dict:
    copied = dict(row)
    copied["value"] = json.dumps(copied["value"], ensure_ascii=False)
    return copied


def _flat_case(case: dict) -> dict:
    tour = case["tour"]
    far = case["farthest"]
    direct = case["direct"]
    max_uav = None
    if tour.get("defined"):
        max_uav = max(vehicle["uav_J"] for vehicle in tour["vehicles"]) * 2.0
    return {
        "case_id": case["case_id"],
        "n_uav": case["n_uav"],
        "n_buoy": case["n_buoy"],
        "regime": case["regime"],
        "wind_name": case["wind_name"],
        "layout": case["layout"],
        "direct_feasible": direct["delivery_feasible"],
        "direct_J": None if not direct.get("episode") else direct["episode"]["total_J"],
        "direct_margin_db": direct.get("min_margin_db"),
        "farthest_feasible": far["delivery_feasible"],
        "farthest_reasons": "|".join(far["reasons"]),
        "farthest_sortie_s": far["sortie_s"],
        "tour_feasible": tour["delivery_feasible"],
        "tour_reasons": "|".join(tour["reasons"]),
        "tour_max_uav_episode_J": max_uav,
        "tour_dominant": case["tour_dominant"],
        "covered_fraction": case["coverage"]["covered_fraction"],
        "drift_exits": case["drift"]["exited"],
    }


def _coverage_row(case: dict) -> dict:
    return {
        "case_id": case["case_id"],
        "coverage": case["coverage"],
        "drift": case["drift"],
    }


def main() -> None:
    if "--render-only" in sys.argv:
        study = json.loads((OUT / "study.json").read_text(encoding="utf-8"))
        verification_path = OUT / "verification.json"
        verification = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else None
        REPORT.write_text(render_report(study, verification), encoding="utf-8")
        return
    verification_path = OUT / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else None
    write_outputs(build_study(), verification)


if __name__ == "__main__":
    main()
