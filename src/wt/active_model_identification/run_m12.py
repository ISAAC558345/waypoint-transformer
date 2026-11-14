"""Run the M1.2 checks. Missions start only after accounting, quadrature, and the window pass."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from wt.active_model_identification.belief import prior_belief
from wt.active_model_identification.completion import accounting_cases, completion_quote
from wt.active_model_identification.integral import select_integral
from wt.active_model_identification.policies import pick_action, select_lookahead
from wt.active_model_identification.report_m12 import write_m12_report
from wt.active_model_identification.scenarios import ablate_beliefs, assess_m12, observation_noise
from wt.active_model_identification.spec import M12_CONFIG_PATH, ROOT, decision_count, load_m12_spec
from wt.active_model_identification.world import advance, energy_used_j, make_world, mission_terminal, on_time_bits, utility

METHODS = ("lookahead_calibration_on", "lookahead_calibration_off")
DECISION_FIELDS = [
    "branch",
    "method",
    "scenario_id",
    "seed",
    "epoch",
    "t_end_s",
    "uav",
    "x",
    "y",
    "speed_m_s",
    "pad_distance_m",
    "flight_state",
    "airborne",
    "land_timer_s",
    "energy_j",
    "z_j",
    "aux_j",
    "consumed_j",
    "requested",
    "executed",
    "intervened",
    "terminal_flags",
    "obs_id",
    "y_j",
    "prediction_error_j",
    "mean_before",
    "variance_before",
    "mean_after",
    "variance_after",
    "assimilated",
    "telemetry_sent",
    "tie_abs",
    "planning_copies_delta",
    "chosen_q",
    "table_choice",
    "calibrate_q",
    "best_execute_q",
    "q_gap_calibrate_minus_execute",
    "completion_quote_id",
    "completion_predicted_j",
    "interval_actual_j",
    "delivery_s",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or (list(rows[0].keys()) if rows else [])
    if not names:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError as exc:
        return {"head": "unknown", "error": str(exc)}
    return {"head": commit}


def _tests_status(log_path: str | None) -> str:
    if not log_path:
        return "PENDING_TESTS"
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    ok = "\nOK" in text or text.rstrip().endswith("OK")
    return "PASS" if ok else "FAIL"


def _judge_accounting(spec: dict, rows: list[dict]) -> dict:
    tol = float(spec["energy"]["execution_abs_tol_j"])
    worst = 0.0
    ok = True
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        total = abs(float(row["group_errors_j"]["total"]))
        worst = max(worst, total)
        if total > tol or int(row["interventions"]) != 0:
            ok = False
        if any(abs(float(value)) > tol for value in row["group_errors_j"].values()):
            ok = False
        grouped.setdefault(row["name"], []).append(row)
    for pair in grouped.values():
        if len(pair) >= 2 and abs(float(pair[0]["affine_z_j"]) - float(pair[1]["affine_z_j"])) > 1e-4:
            ok = False
    if not all(row["on_time"] for row in grouped.get("pad_far", [])):
        ok = False
    if any(row["on_time"] for row in grouped.get("near_deadline_cache", [])):
        ok = False
    return {"status": "PASS" if ok else "FAIL", "worst_abs_j": worst, "tolerance_j": tol, "cases": rows}


def _q_fields(table, tau: float) -> dict:
    chosen_score, chosen = pick_action(table, tau)
    calibrate = next((item for item in table if item[1][0] == "calibrate"), None)
    execute = [item for item in table if item[1][0] != "calibrate"]
    best_execute = max(execute, key=lambda item: item[0]) if execute else None
    gap = None if calibrate is None or best_execute is None else calibrate[0] - best_execute[0]
    return {
        "chosen_q": chosen_score,
        "table_choice": f"{chosen[0]}:{chosen[1]}",
        "calibrate_q": "" if calibrate is None else calibrate[0],
        "best_execute_q": "" if best_execute is None else best_execute[0],
        "q_gap_calibrate_minus_execute": "" if gap is None else gap,
    }


def _quotes(world, beliefs, requested, spec) -> dict[int, object]:
    guard = float(spec["belief"]["guard_probability"])
    found = {}
    for index, action in enumerate(requested):
        theta = beliefs[index].quantile(guard)
        if action[0] == "service" and action[1] is not None:
            found[index] = completion_quote(world, index, int(action[1]), theta, spec)
        elif action[0] == "backhaul":
            found[index] = completion_quote(world, index, None, theta, spec)
    return found


def _play(method, scenario, spec, factor, stats, branch, start_epoch, world, beliefs, removed_ids):
    sigma = float(spec["belief"]["sigma_j"])
    noise = observation_noise(int(scenario["seed"]), sigma)
    tau = float(spec["integral"]["tie_abs"])
    decisions = []
    covered = 0
    coverage_n = 0
    first_service = [None, None]
    snapshot = None
    for epoch in range(start_epoch, decision_count(spec)):
        before = [item.copy() for item in beliefs]
        before_copies = int(stats.get("planning_copies", 0))
        requested, tables = select_lookahead(method, world, beliefs, spec, factor, stats)
        copies = int(stats.get("planning_copies", 0)) - before_copies
        predicted = _quotes(world, beliefs, requested, spec)
        spent_before = [uav.consumed_j for uav in world.uavs]
        delivery_before = len(world.deliveries)
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
        fresh = world.deliveries[delivery_before:]
        delivered = {}
        for item in fresh:
            delivered.setdefault(int(item["uav"]), []).append(f"{item['id']}@{float(item['delivered_s']):.3f}")
        for obs in observations:
            if obs["obs_id"] in removed_ids:
                raise RuntimeError(f"removed observation {obs['obs_id']} was assimilated again")
            uav = int(obs["uav"])
            quote = predicted.get(uav)
            scores = _q_fields(tables[uav], tau)
            decisions.append(
                {
                    "branch": branch,
                    "method": method,
                    "scenario_id": scenario["id"],
                    "seed": scenario["seed"],
                    "epoch": obs["interval"],
                    "t_end_s": (int(obs["interval"]) + 1) * int(spec["time"]["decision_s"]),
                    "uav": uav,
                    "x": obs["x"],
                    "y": obs["y"],
                    "speed_m_s": obs["speed_m_s"],
                    "pad_distance_m": obs["pad_distance_m"],
                    "flight_state": obs["flight_state"],
                    "airborne": obs["airborne"],
                    "land_timer_s": obs["land_timer_s"],
                    "energy_j": obs["energy_j"],
                    "z_j": obs["z_j"],
                    "aux_j": obs["aux_j"],
                    "consumed_j": obs["consumed_j"],
                    "requested": f"{obs['requested'][0]}:{obs['requested'][1]}",
                    "executed": f"{obs['executed'][0]}:{obs['executed'][1]}",
                    "intervened": obs["intervened"],
                    "terminal_flags": "",
                    "obs_id": obs["obs_id"],
                    "y_j": obs["y_j"],
                    "prediction_error_j": obs["prediction_error_j"],
                    "mean_before": obs["mean_before"],
                    "variance_before": obs["variance_before"],
                    "mean_after": obs["mean_after"],
                    "variance_after": obs["variance_after"],
                    "assimilated": obs["assimilated"],
                    "telemetry_sent": obs["telemetry_sent"],
                    "tie_abs": tau,
                    "planning_copies_delta": copies,
                    "completion_quote_id": "" if quote is None else quote.quote_id,
                    "completion_predicted_j": "" if quote is None else quote.energy_j,
                    "interval_actual_j": world.uavs[uav].consumed_j - spent_before[uav],
                    "delivery_s": ";".join(delivered.get(uav, [])),
                    **scores,
                }
            )
            executed = obs["executed"]
            if first_service[uav] is None and executed[0] == "service":
                first_service[uav] = executed[1]
            if obs["telemetry_sent"] and float(obs["z_j"]) > 100.0:
                coverage_n += 1
                covered += int(bool(obs["covered_90"]))
        if snapshot is None and branch == "core" and method == "lookahead_calibration_on" and scenario["condition"] == "sensitive":
            if any(obs["executed"][0] == "calibrate" for obs in observations):
                executed = [tuple(obs["executed"]) for obs in observations]
                ablated, removed_uavs = ablate_beliefs(before, beliefs, executed)
                snapshot = {
                    "scenario": scenario,
                    "world": world.snapshot(),
                    "beliefs_updated": [item.copy() for item in beliefs],
                    "beliefs_removed": ablated,
                    "removed_uavs": removed_uavs,
                    "removed_ids": [obs["obs_id"] for obs in observations if obs["executed"][0] == "calibrate"],
                    "next_epoch": epoch + 1,
                }
    terminals = mission_terminal(world, spec)
    for row in decisions:
        if int(row["epoch"]) == decision_count(spec) - 1:
            flag = terminals[int(row["uav"])]
            row["terminal_flags"] = (
                f"returned:{int(flag['returned'])};not_returned:{int(flag['not_returned'])};"
                f"not_landed:{int(flag['not_landed'])};below_reserve:{int(flag['below_reserve'])};"
                f"exhausted:{int(flag['energy_exhausted'])}"
            )
    errors = [(beliefs[index].mean() - float(scenario["theta"][index])) ** 2 for index in range(2)]
    episode = {
        "branch": branch,
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
        "not_returned": int(sum(int(item["not_returned"]) for item in terminals)),
        "not_landed": int(sum(int(item["not_landed"]) for item in terminals)),
        "below_reserve": int(sum(int(item["below_reserve"]) for item in terminals)),
        "energy_exhausted": int(sum(int(item["energy_exhausted"]) for item in terminals)),
        "returned": int(sum(int(item["returned"]) for item in terminals)),
        "first_service_0": first_service[0],
        "first_service_1": first_service[1],
    }
    deliveries = []
    for item in world.deliveries:
        stamped = dict(item)
        stamped.update({"branch": branch, "method": method, "scenario_id": scenario["id"]})
        deliveries.append(stamped)
    return episode, decisions, deliveries, snapshot


def _blank_paired() -> dict:
    return {
        "sensitive_utility_gap": None,
        "insensitive_utility_gap": None,
        "sensitive_bits_gap": None,
        "sensitive_energy_gap_j": None,
        "sensitive_choice_differences": None,
        "sensitive_pairs": 0,
        "calibration_snapshots": 0,
        "ablation_utility_gap": None,
    }


def _paired(episodes: list[dict], ablation: list[dict], snapshots: int) -> dict:
    cores = [row for row in episodes if row["branch"] == "core"]

    def gap(condition: str, key: str):
        values = []
        for base in sorted({row["base"] for row in cores if row["condition"] == condition}):
            left = next(row for row in cores if row["base"] == base and row["condition"] == condition and row["method"] == METHODS[0])
            right = next(row for row in cores if row["base"] == base and row["condition"] == condition and row["method"] == METHODS[1])
            values.append(float(left[key]) - float(right[key]))
        return _mean(values)

    choice = 0
    pairs = 0
    for base in sorted({row["base"] for row in cores if row["condition"] == "sensitive"}):
        left = next(row for row in cores if row["base"] == base and row["condition"] == "sensitive" and row["method"] == METHODS[0])
        right = next(row for row in cores if row["base"] == base and row["condition"] == "sensitive" and row["method"] == METHODS[1])
        pairs += 1
        if (left["first_service_0"], left["first_service_1"]) != (right["first_service_0"], right["first_service_1"]):
            choice += 1
    updated = [row for row in ablation if row["branch"] == "ablation_updated"]
    removed = [row for row in ablation if row["branch"] == "ablation_removed"]
    ablation_gap = _mean(
        [
            float(next(row for row in updated if row["source_scenario"] == item["source_scenario"])["utility"]) - float(item["utility"])
            for item in removed
        ]
    )
    return {
        "sensitive_utility_gap": gap("sensitive", "utility"),
        "insensitive_utility_gap": gap("insensitive", "utility"),
        "sensitive_bits_gap": gap("sensitive", "on_time_bits"),
        "sensitive_energy_gap_j": gap("sensitive", "energy_j"),
        "sensitive_choice_differences": choice,
        "sensitive_pairs": pairs,
        "calibration_snapshots": snapshots,
        "ablation_utility_gap": ablation_gap,
    }


def run_trial(out_dir: Path, tests_log: str | None, force: bool) -> dict:
    posix = out_dir.as_posix()
    if "active_model_identification_m12" not in posix or "active_model_identification_m11" in posix:
        raise SystemExit("refusing to write outside results/active_model_identification_m12")
    if (out_dir / "summary.json").exists() and not force:
        raise SystemExit(f"existing summary at {out_dir}; pass --force to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spec = copy.deepcopy(load_m12_spec())
    print("accounting", flush=True)
    accounting = _judge_accounting(spec, accounting_cases(spec))
    print("scenario window", flush=True)
    assessed = assess_m12(spec)
    print("integral", flush=True)
    integral = select_integral(spec)
    numerical = str(integral["numerical_status"])
    if numerical == "PASS":
        spec["belief"]["quadrature_nodes"] = int(integral["theta_nodes"])
        spec["integral"]["noise_nodes"] = int(integral["noise_nodes"])
    frozen = copy.deepcopy(spec)
    frozen["integral_result"] = {
        "numerical_status": numerical,
        "theta_nodes": integral["theta_nodes"],
        "noise_nodes": integral["noise_nodes"],
        "tie_abs": integral["tie_abs"],
    }
    frozen_path = out_dir / "frozen_config.json"
    frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (out_dir / "integral_check.json").write_text(json.dumps(integral, indent=2), encoding="utf-8")
    (out_dir / "accounting_check.json").write_text(json.dumps(accounting, indent=2), encoding="utf-8")
    scenario_status = assessed["scenario_status"]
    comparison = "NOT_RUN"
    run_missions = accounting["status"] == "PASS" and numerical == "PASS" and bool(assessed["window_valid"])
    manifest = {
        "scenario_status": scenario_status,
        "comparison_status": "RAN" if run_missions else "NOT_RUN",
        "window_valid": assessed["window_valid"],
        "limiting_condition": assessed["limiting_condition"],
        "budget": {key: value for key, value in assessed.items() if key != "scenarios"},
        "scenarios": assessed["scenarios"],
        "missions_scheduled": run_missions,
    }
    scenario_path = out_dir / "scenario_manifest.json"
    scenario_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stats = {"physical_steps": 0, "planning_steps": 0, "planning_copies": 0}
    ablation_stats = {"physical_steps": 0, "planning_steps": 0, "planning_copies": 0}
    episodes: list[dict] = []
    decisions: list[dict] = []
    deliveries: list[dict] = []
    snapshots = []
    if run_missions:
        comparison = "RAN"
        factor = float(assessed["integration_factor"])
        for scenario in assessed["scenarios"]:
            for method in METHODS:
                print(f"run {scenario['id']} {method}", flush=True)
                world = make_world(spec, scenario["battery_j"], f"{scenario['id']}-{method}")
                beliefs = [prior_belief(spec), prior_belief(spec)]
                episode, decision_rows, delivery_rows, snapshot = _play(
                    method, scenario, spec, factor, stats, "core", 0, world, beliefs, []
                )
                episodes.append(episode)
                decisions.extend(decision_rows)
                deliveries.extend(delivery_rows)
                if snapshot is not None:
                    snapshots.append(snapshot)
        for snapshot in snapshots[:4]:
            print(f"ablate {snapshot['scenario']['id']}", flush=True)
            for label, beliefs in (
                ("ablation_updated", snapshot["beliefs_updated"]),
                ("ablation_removed", snapshot["beliefs_removed"]),
            ):
                continued = snapshot["world"].snapshot()
                for obs_id in snapshot["removed_ids"]:
                    if obs_id not in continued.assimilated_ids:
                        continued.assimilated_ids.append(obs_id)
                episode, decision_rows, delivery_rows, _again = _play(
                    "lookahead_calibration_off",
                    snapshot["scenario"],
                    spec,
                    factor,
                    ablation_stats,
                    label,
                    int(snapshot["next_epoch"]),
                    continued,
                    [item.copy() for item in beliefs],
                    list(snapshot["removed_ids"]),
                )
                episode["removed_uavs"] = ",".join(str(item) for item in snapshot["removed_uavs"])
                episode["removed_ids"] = ",".join(snapshot["removed_ids"])
                episode["source_scenario"] = snapshot["scenario"]["id"]
                episodes.append(episode)
                decisions.extend(decision_rows)
                deliveries.extend(delivery_rows)
    cores = [row for row in episodes if row["branch"] == "core"]
    ablation = [row for row in episodes if str(row["branch"]).startswith("ablation")]
    _write_csv(out_dir / "episode_rows.csv", cores)
    _write_csv(out_dir / "decision_rows.csv", decisions, DECISION_FIELDS)
    _write_csv(out_dir / "deliveries.csv", deliveries)
    _write_csv(out_dir / "ablation_rows.csv", ablation)
    paired = _paired(episodes, ablation, len(snapshots[:4])) if cores else _blank_paired()
    if not cores:
        paired["calibration_snapshots"] = 0
    elapsed = time.perf_counter() - started
    budget = {key: value for key, value in assessed.items() if key != "scenarios"}
    budget["execution_abs_tol_j"] = float(spec["energy"]["execution_abs_tol_j"])
    if assessed["window_valid"]:
        next_step = "窗口成立后的对照已经写入本目录。不要用这 16 条样本做显著性或外推，也不要接着搜索新电量。"
    else:
        next_step = (
            "本轮停止。原电量不下调，也不另搜一档电池。若以后要让保守后验放行远航程，"
            "需要一条新的预注册指令，单独规定能盖住后验分位抬升的裕量，或改为在普通服务航段里辨识。"
            "不要把这次未运行的对照解释成校准收益为零。"
        )
    summary = {
        "schema_version": 1,
        "parent_commit": spec["parent_commit"],
        "reviewed_m11_commit": spec["reviewed_m11_commit"],
        "result_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(M12_CONFIG_PATH),
        "frozen_config_sha256": _sha256(frozen_path),
        "scenario_manifest_sha256": _sha256(scenario_path),
        "python": sys.version,
        "platform": platform.platform(),
        "unit_tests_status": _tests_status(tests_log),
        "accounting_status": accounting["status"],
        "numerical_status": numerical,
        "scenario_status": scenario_status,
        "comparison_status": comparison,
        "accounting": {"status": accounting["status"], "worst_abs_j": accounting["worst_abs_j"], "tolerance_j": accounting["tolerance_j"]},
        "integral": integral,
        "budget": budget,
        "paired": paired if cores else None,
        "counts": {
            "core_missions": len(cores),
            "ablation_branches": len(ablation),
            "calibration_snapshots": 0 if not cores else len(snapshots[:4]),
            "physical_steps": int(stats["physical_steps"]),
            "ablation_physical_steps": int(ablation_stats["physical_steps"]),
            "planning_steps": int(stats["planning_steps"] + ablation_stats["planning_steps"]),
            "planning_copies": int(stats["planning_copies"] + ablation_stats["planning_copies"]),
            "development_planning_steps": int(integral["development_planning_steps"]),
            "development_planning_copies": int(integral["development_planning_copies"]),
            "return_failures": int(sum(int(row["return_failures"]) for row in cores)) if cores else 0,
            "training_updates": 0,
            "elapsed_s": elapsed,
        },
        "next_step": next_step,
        "commands": {
            "tests": "py -3 -m unittest tests.test_active_model_identification_m12 tests.test_active_model_identification_m11 tests.test_active_model_identification_m1",
            "trial": "py -3 -m wt.active_model_identification.run_m12 --tests-log results/active_model_identification_m12/m12_20261114/necessary_tests.log",
            "pythonpath": "src",
        },
        "git": _git(),
    }
    if not cores:
        summary["paired"] = None
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "parent_commit": summary["parent_commit"],
                "head_at_run": summary["git"]["head"],
                "unit_tests_status": summary["unit_tests_status"],
                "accounting_status": summary["accounting_status"],
                "numerical_status": summary["numerical_status"],
                "scenario_status": summary["scenario_status"],
                "comparison_status": summary["comparison_status"],
                "core_missions": len(cores),
                "ablation_branches": len(ablation),
                "config_sha256": summary["config_sha256"],
                "frozen_config_sha256": summary["frozen_config_sha256"],
                "scenario_manifest_sha256": summary["scenario_manifest_sha256"],
                "elapsed_s": elapsed,
                "training_updates": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_m12_report(summary, ROOT / "docs" / "research" / "Active_Model_Identification_M12_Report_CN.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/active_model_identification_m12/m12_20261114")
    parser.add_argument("--tests-log", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_trial(ROOT / args.out, args.tests_log or None, args.force)
    print(
        json.dumps(
            {
                "unit_tests": summary["unit_tests_status"],
                "accounting": summary["accounting_status"],
                "numerical": summary["numerical_status"],
                "scenario": summary["scenario_status"],
                "comparison": summary["comparison_status"],
                "missions": summary["counts"]["core_missions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
