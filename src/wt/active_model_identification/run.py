"""Run the registered M1 trial and write its manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

from wt.active_model_identification.belief import fixed_model_belief, prior_belief
from wt.active_model_identification.diagnostics import direct_row, fixed_receiver_row
from wt.active_model_identification.policies import select_actions
from wt.active_model_identification.report_cn import write_report
from wt.active_model_identification.scenarios import build_scenarios, observation_noise
from wt.active_model_identification.spec import CONFIG_PATH, ROOT, decision_count, load_spec
from wt.active_model_identification.world import advance, energy_used_j, make_world, on_time_bits, utility

METHODS = ("fixed_model", "passive", "fixed_calibration", "finite_lookahead")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _episode(method: str, scenario: dict, spec: dict, factor: float, stats: dict) -> tuple[dict, list[dict]]:
    world = make_world(spec, scenario["battery_j"], scenario["id"])
    if method == "fixed_model":
        beliefs = [fixed_model_belief(spec), fixed_model_belief(spec)]
    else:
        beliefs = [prior_belief(spec), prior_belief(spec)]
    sigma = float(spec["belief"]["sigma_j"])
    noise = observation_noise(int(scenario["seed"]), sigma)
    decisions: list[dict] = []
    covered = 0
    coverage_n = 0
    first_service = [None, None]
    first_z = [0.0, 0.0]
    epochs = decision_count(spec)
    for epoch in range(epochs):
        requested = select_actions(method, world, beliefs, spec, factor, epoch, stats)
        frozen = [item.copy() for item in beliefs]
        observations = advance(
            world,
            requested,
            list(scenario["theta"]),
            beliefs,
            spec,
            factor,
            stats,
            planning=False,
            interval_index=epoch,
            noise_j=noise,
        )
        if method == "fixed_model":
            beliefs = frozen
        for obs in observations:
            uav = int(obs["uav"])
            if epoch == 0:
                first_z[uav] = float(obs["z_j"])
            executed = obs["executed"]
            if first_service[uav] is None and executed[0] == "service":
                first_service[uav] = executed[1]
            if obs["telemetry_sent"] and float(obs["z_j"]) > 100.0:
                coverage_n += 1
                covered += int(bool(obs["covered_90"]))
            decisions.append(
                {
                    "method": method,
                    "scenario_id": scenario["id"],
                    "epoch": epoch,
                    "uav": uav,
                    "requested": f"{obs['requested'][0]}:{obs['requested'][1]}",
                    "executed": f"{executed[0]}:{executed[1]}",
                    "z_j": obs["z_j"],
                    "y_j": obs["y_j"],
                    "prediction_error_j": obs["prediction_error_j"],
                    "mean_before": obs["mean_before"],
                    "variance_before": obs["variance_before"],
                    "covered_90": obs["covered_90"],
                    "telemetry_sent": obs["telemetry_sent"],
                }
            )
    end_variance = [beliefs[index].variance() for index in range(2)]
    errors = [(beliefs[index].mean() - float(scenario["theta"][index])) ** 2 for index in range(2)]
    row = {
        "method": method,
        "scenario_id": scenario["id"],
        "base": scenario["base"],
        "condition": scenario["condition"],
        "theta_0": scenario["theta"][0],
        "theta_1": scenario["theta"][1],
        "utility": utility(world, spec),
        "on_time_bits": on_time_bits(world),
        "energy_j": sum(energy_used_j(world)),
        "calibration_s": sum(world.calibration_s),
        "calibration_j": sum(world.calibration_j),
        "rmse": (sum(errors) / 2.0) ** 0.5,
        "coverage_90": (covered / coverage_n) if coverage_n else None,
        "coverage_n": coverage_n,
        "return_failures": world.return_failures,
        "interventions": world.interventions,
        "min_separation_m": world.min_separation_m,
        "first_service_0": first_service[0],
        "first_service_1": first_service[1],
        "end_variance_0": end_variance[0],
        "end_variance_1": end_variance[1],
        "first_z_0": first_z[0],
        "first_z_1": first_z[1],
        "failed_0": world.uavs[0].failed,
        "failed_1": world.uavs[1].failed,
    }
    return row, decisions


def _subset(rows: list[dict], method: str, condition: str | None = None) -> list[dict]:
    return [row for row in rows if row["method"] == method and (condition is None or row["condition"] == condition)]


def _answers(rows: list[dict]) -> dict:
    fixed_cal = _subset(rows, "fixed_calibration")
    passive = _subset(rows, "passive")
    lookahead = _subset(rows, "finite_lookahead")

    def first_z(group: list[dict]) -> float:
        return _mean([(float(row["first_z_0"]) + float(row["first_z_1"])) / 2.0 for row in group])

    def end_var(group: list[dict]) -> float:
        return _mean([(float(row["end_variance_0"]) + float(row["end_variance_1"])) / 2.0 for row in group])

    sensitive_pairs = 0
    choice_differences = 0
    gaps = []
    for base in sorted({row["base"] for row in rows}):
        left = next(row for row in lookahead if row["base"] == base and row["condition"] == "sensitive")
        right = next(row for row in passive if row["base"] == base and row["condition"] == "sensitive")
        sensitive_pairs += 1
        if (left["first_service_0"], left["first_service_1"]) != (right["first_service_0"], right["first_service_1"]):
            choice_differences += 1
        gaps.append(float(left["utility"]) - float(right["utility"]))
    look_s = _subset(rows, "finite_lookahead", "sensitive")
    look_i = _subset(rows, "finite_lookahead", "insensitive")
    fixed_i = _subset(rows, "fixed_calibration", "insensitive")
    cal_s = _mean([float(row["calibration_s"]) for row in look_s])
    cal_i = _mean([float(row["calibration_s"]) for row in look_i])
    fixed_cal_i = _mean([float(row["calibration_s"]) for row in fixed_i])
    adds = first_z(fixed_cal) > first_z(passive) + 100.0 and end_var(_subset(rows, "fixed_calibration", "sensitive")) <= end_var(
        _subset(rows, "passive", "sensitive")
    ) * 1.05 + 1.0e-8
    return {
        "calibration_adds_information": bool(adds),
        "fixed_calibration_first_z_j": first_z(fixed_cal),
        "passive_first_z_j": first_z(passive),
        "fixed_calibration_end_variance": end_var(_subset(rows, "fixed_calibration", "sensitive")),
        "passive_end_variance": end_var(_subset(rows, "passive", "sensitive")),
        "information_changes_choice": choice_differences > 0,
        "sensitive_choice_differences": choice_differences,
        "sensitive_pairs": sensitive_pairs,
        "benefit_offsets_cost": _mean(gaps) > 0.0,
        "sensitive_utility_gap": _mean(gaps),
        "insensitive_reduces_calibration": cal_i < cal_s and cal_i < fixed_cal_i,
        "lookahead_sensitive_cal_s": cal_s,
        "lookahead_insensitive_cal_s": cal_i,
        "fixed_calibration_insensitive_cal_s": fixed_cal_i,
    }


def _method_means(rows: list[dict]) -> list[dict]:
    output = []
    for method in METHODS:
        for condition in ("sensitive", "insensitive"):
            group = _subset(rows, method, condition)
            output.append(
                {
                    "method": method,
                    "condition": condition,
                    "utility": _mean([float(row["utility"]) for row in group]),
                    "on_time_bits": _mean([float(row["on_time_bits"]) for row in group]),
                    "energy_kj": _mean([float(row["energy_j"]) for row in group]) / 1000.0,
                    "calibration_s": _mean([float(row["calibration_s"]) for row in group]),
                    "calibration_kj": _mean([float(row["calibration_j"]) for row in group]) / 1000.0,
                    "rmse": _mean([float(row["rmse"]) for row in group]),
                    "coverage": _mean([float(row["coverage_90"]) for row in group if row["coverage_90"] is not None]),
                    "return_failures": int(sum(int(row["return_failures"]) for row in group)),
                }
            )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _tests_status(log_path: str | None) -> dict:
    if not log_path:
        return {"status": "NOT_ATTACHED", "ok": False}
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    ok = "\nOK" in text or text.rstrip().endswith("OK")
    return {"status": "PASS" if ok else "FAIL", "ok": ok, "log": log_path}


def run_trial(out_dir: Path, tests_log: str | None, force: bool) -> dict:
    if (out_dir / "summary.json").exists() and not force:
        raise SystemExit(f"existing summary at {out_dir}; pass --force to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / "RUNNING"
    if lock.exists() and not force:
        raise SystemExit(f"found {lock}; another run may be unfinished")
    lock.write_text("running\n", encoding="utf-8")
    spec = load_spec()
    scenarios, budget, factor = build_scenarios(spec)
    scenario_path = out_dir / "scenario_manifest.json"
    scenario_path.write_text(json.dumps({"scenarios": scenarios, "budget": budget}, indent=2), encoding="utf-8")
    stats = {"physical_steps": 0, "planning_steps": 0, "planning_copies": 0}
    episodes: list[dict] = []
    decisions: list[dict] = []
    for scenario in scenarios:
        for method in METHODS:
            print(f"run {scenario['id']} {method}", flush=True)
            row, decision_rows = _episode(method, scenario, spec, factor, stats)
            episodes.append(row)
            decisions.extend(decision_rows)
            _write_csv(out_dir / "episode_rows.csv", episodes)
            _write_csv(out_dir / "decision_rows.csv", decisions)
    diagnostics = [fixed_receiver_row(spec, scenario) for scenario in scenarios]
    directs = [direct_row(spec, scenario) for scenario in scenarios]
    (out_dir / "diagnostics.json").write_text(json.dumps({"fixed_receiver": diagnostics, "direct": directs}, indent=2), encoding="utf-8")
    tests = _tests_status(tests_log)
    answers = _answers(episodes)
    mechanism_flags = [
        answers["calibration_adds_information"],
        answers["information_changes_choice"],
        answers["benefit_offsets_cost"],
        answers["insensitive_reduces_calibration"],
    ]
    summary = {
        "schema_version": 1,
        "parent_commit": spec["parent_commit"],
        "reference_base": spec["reference_base"],
        "result_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(CONFIG_PATH),
        "scenario_manifest_sha256": _sha256(scenario_path),
        "python": sys.version,
        "platform": platform.platform(),
        "integration_factor": factor,
        "interface_status": "PASS" if tests["ok"] else "PENDING_TESTS",
        "mechanism_status": "SAMPLE_DIFFERENCES_PRESENT" if any(mechanism_flags) else "NO_SAMPLE_BENEFIT",
        "tests": tests,
        "counts": {
            "core_missions": len(episodes),
            "fixed_receiver_diagnostics": len(diagnostics),
            "direct_accountings": len(directs),
            "executed_tasks": len(episodes) + len(diagnostics),
            "physical_steps": stats["physical_steps"],
            "planning_steps": stats["planning_steps"],
            "planning_copies": stats["planning_copies"],
            "return_failures": int(sum(int(row["return_failures"]) for row in episodes)),
            "training_updates": 0,
            "model_forward_passes": 0,
        },
        "answers": answers,
        "method_means": _method_means(episodes),
        "diagnostics_overview": {
            "direct_delivered_bits": sum(float(row["delivered_bits"]) for row in directs),
            "fixed_receiver_delivered_bits": sum(float(row["delivered_bits"]) for row in diagnostics),
            "direct_feasible_scenarios": sum(1 for row in directs if float(row["delivered_bits"]) > 0.0),
        },
        "commands": {
            "tests": "python -m unittest tests.test_active_model_identification_m1 -v",
            "trial": "python -m wt.active_model_identification.run --tests-log results/active_model_identification_m1/m1_20261113/necessary_tests.log",
            "pythonpath": "src",
        },
        "budget": budget,
        "claim_limits": [
            "32 core missions are one paired draw per method, not a significance test",
            "finite lookahead is a one-interval coordinate-wise approximation",
            "no network was trained and no legacy checkpoint was loaded",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "parent_commit": spec["parent_commit"],
                "config_sha256": summary["config_sha256"],
                "scenario_manifest_sha256": summary["scenario_manifest_sha256"],
                "python": sys.version,
                "seed_rule": "observation noise is Random(f'm1-obs-{scenario_seed}-{uav}-{interval}') and does not depend on action order",
                "core_missions": len(episodes),
                "physical_steps": stats["physical_steps"],
                "planning_steps": stats["planning_steps"],
                "planning_copies": stats["planning_copies"],
                "training_updates": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(summary, ROOT / "docs" / "research" / "Active_Model_Identification_M1_Report_CN.md")
    lock.unlink(missing_ok=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/active_model_identification_m1/m1_20261113")
    parser.add_argument("--tests-log", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_trial(ROOT / args.out, args.tests_log or None, args.force)
    print(json.dumps({"interface": summary["interface_status"], "mechanism": summary["mechanism_status"], "missions": summary["counts"]["core_missions"]}, indent=2))


if __name__ == "__main__":
    main()
