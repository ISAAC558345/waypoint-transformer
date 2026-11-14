"""Run the M1.1 matched-planner trial. Does not write the M1 result directory."""

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
from wt.active_model_identification.integral import select_integral
from wt.active_model_identification.policies import pick_action, select_lookahead
from wt.active_model_identification.report_m11 import write_m11_report
from wt.active_model_identification.scenarios import (
    _chain_status,
    ablate_beliefs,
    assess_budgets,
    observation_noise,
    predicate_rows,
)
from wt.active_model_identification.spec import M11_CONFIG_PATH, ROOT, decision_count, load_m11_spec
from wt.active_model_identification.world import (
    Action,
    advance,
    energy_used_j,
    make_world,
    mission_terminal,
    on_time_bits,
    utility,
)

METHODS = ("lookahead_calibration_on", "lookahead_calibration_off")
M1_MANIFEST = ROOT / "results" / "active_model_identification_m1" / "m1_20261113" / "scenario_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _git() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(
            ["git", "-c", "core.longpaths=true", "status", "--porcelain", "--", "src/wt/active_model_identification", "configs/experiments", "tests", "docs/research", "CURSOR_NEXT_TASK.md", "README_BRANCH_SCOPE.md"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        return {"head": "unknown", "error": str(exc.output)}
    return {"head": commit, "m11_status_lines": status.splitlines()[:40]}


def _tests_status(log_path: str | None) -> dict:
    if not log_path:
        return {"status": "NOT_ATTACHED", "ok": False}
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    ok = "\nOK" in text or text.rstrip().endswith("OK")
    return {"status": "PASS" if ok else "FAIL", "ok": ok, "log": log_path}


def _q_fields(table: list[tuple[float, Action]], tau: float) -> dict:
    chosen_score, chosen = pick_action(table, tau)
    calibrate = next((item for item in table if item[1][0] == "calibrate"), None)
    execute = [item for item in table if item[1][0] != "calibrate"]
    best_execute = max(execute, key=lambda item: item[0]) if execute else None
    gap = None
    if calibrate is not None and best_execute is not None:
        gap = calibrate[0] - best_execute[0]
    return {
        "chosen_q": chosen_score,
        "table_choice": f"{chosen[0]}:{chosen[1]}",
        "calibrate_q": "" if calibrate is None else calibrate[0],
        "best_execute_q": "" if best_execute is None else best_execute[0],
        "q_gap_calibrate_minus_execute": "" if gap is None else gap,
    }


def _decision_row(method: str, scenario: dict, branch: str, obs: dict, scores: dict, tau: float, copies: int) -> dict:
    terminal = ""
    return {
        "branch": branch,
        "method": method,
        "scenario_id": scenario["id"],
        "seed": scenario["seed"],
        "epoch": obs["interval"],
        "t_end_s": (int(obs["interval"]) + 1) * 30,
        "uav": obs["uav"],
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
        "terminal_flags": terminal,
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
        **scores,
    }


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
        for obs in observations:
            if obs["obs_id"] in removed_ids:
                raise RuntimeError(f"removed observation {obs['obs_id']} was assimilated again")
            scores = _q_fields(tables[int(obs["uav"])], tau)
            decisions.append(_decision_row(method, scenario, branch, obs, scores, tau, copies))
            executed = obs["executed"]
            uav = int(obs["uav"])
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
        "end_variance_0": beliefs[0].variance(),
        "end_variance_1": beliefs[1].variance(),
    }
    deliveries = []
    for item in world.deliveries:
        stamped = dict(item)
        stamped["branch"] = branch
        stamped["method"] = method
        stamped["scenario_id"] = scenario["id"]
        deliveries.append(stamped)
    return episode, decisions, deliveries, snapshot


def _core_episode(method, scenario, spec, factor, stats):
    world = make_world(spec, scenario["battery_j"], f"{scenario['id']}-{method}")
    beliefs = [prior_belief(spec), prior_belief(spec)]
    return _play(method, scenario, spec, factor, stats, "core", 0, world, beliefs, [])


def _continue(label, snapshot, beliefs, spec, factor, stats):
    world = snapshot["world"].snapshot()
    for obs_id in snapshot["removed_ids"]:
        if obs_id not in world.assimilated_ids:
            world.assimilated_ids.append(obs_id)
    episode, decisions, deliveries, _again = _play(
        "lookahead_calibration_off",
        snapshot["scenario"],
        spec,
        factor,
        stats,
        label,
        int(snapshot["next_epoch"]),
        world,
        [item.copy() for item in beliefs],
        list(snapshot["removed_ids"]),
    )
    episode["removed_uavs"] = ",".join(str(item) for item in snapshot["removed_uavs"])
    episode["removed_ids"] = ",".join(snapshot["removed_ids"])
    episode["source_scenario"] = snapshot["scenario"]["id"]
    return episode, decisions, deliveries


def _pair_rows(episodes: list[dict]) -> list[dict]:
    rows = []
    cores = [row for row in episodes if row["branch"] == "core"]
    for scenario_id in sorted({row["scenario_id"] for row in cores}):
        left = next(row for row in cores if row["scenario_id"] == scenario_id and row["method"] == "lookahead_calibration_on")
        right = next(row for row in cores if row["scenario_id"] == scenario_id and row["method"] == "lookahead_calibration_off")
        rows.append(
            {
                "scenario_id": scenario_id,
                "condition": left["condition"],
                "utility_on_minus_off": float(left["utility"]) - float(right["utility"]),
                "bits_on_minus_off": float(left["on_time_bits"]) - float(right["on_time_bits"]),
                "energy_j_on_minus_off": float(left["energy_j"]) - float(right["energy_j"]),
                "calibration_s_on": float(left["calibration_s"]),
                "calibration_s_off": float(right["calibration_s"]),
                "service_differs": int((left["first_service_0"], left["first_service_1"]) != (right["first_service_0"], right["first_service_1"])),
                "on_first_service": f"{left['first_service_0']},{left['first_service_1']}",
                "off_first_service": f"{right['first_service_0']},{right['first_service_1']}",
            }
        )
    return rows


def _answers(episodes: list[dict], ablation: list[dict], snapshots: int) -> dict:
    cores = [row for row in episodes if row["branch"] == "core"]

    def gap(condition: str, key: str) -> float:
        values = []
        for base in sorted({row["base"] for row in cores if row["condition"] == condition}):
            left = next(row for row in cores if row["base"] == base and row["condition"] == condition and row["method"] == "lookahead_calibration_on")
            right = next(row for row in cores if row["base"] == base and row["condition"] == condition and row["method"] == "lookahead_calibration_off")
            values.append(float(left[key]) - float(right[key]))
        return _mean(values)

    sensitive = [row for row in cores if row["condition"] == "sensitive" and row["method"] == "lookahead_calibration_on"]
    insensitive = [row for row in cores if row["condition"] == "insensitive" and row["method"] == "lookahead_calibration_on"]
    off = [row for row in cores if row["method"] == "lookahead_calibration_off"]
    choice = 0
    pairs = 0
    for base in sorted({row["base"] for row in cores if row["condition"] == "sensitive"}):
        left = next(row for row in sensitive if row["base"] == base)
        right = next(row for row in cores if row["base"] == base and row["condition"] == "sensitive" and row["method"] == "lookahead_calibration_off")
        pairs += 1
        if (left["first_service_0"], left["first_service_1"]) != (right["first_service_0"], right["first_service_1"]):
            choice += 1
    updated = [row for row in ablation if row["branch"] == "ablation_updated"]
    removed = [row for row in ablation if row["branch"] == "ablation_removed"]
    ablation_gap = _mean(
        [
            float(next(row for row in updated if row["source_scenario"] == item["source_scenario"])["utility"])
            - float(item["utility"])
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
        "on_sensitive_cal_s": _mean([float(row["calibration_s"]) for row in sensitive]),
        "on_insensitive_cal_s": _mean([float(row["calibration_s"]) for row in insensitive]),
        "off_cal_s": _mean([float(row["calibration_s"]) for row in off]),
        "calibration_snapshots": snapshots,
        "ablation_utility_gap": ablation_gap,
    }


def _method_means(episodes: list[dict]) -> list[dict]:
    output = []
    cores = [row for row in episodes if row["branch"] == "core"]
    for method in METHODS:
        for condition in ("sensitive", "insensitive"):
            group = [row for row in cores if row["method"] == method and row["condition"] == condition]
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
                    "not_returned": int(sum(int(row["not_returned"]) for row in group)),
                    "not_landed": int(sum(int(row["not_landed"]) for row in group)),
                    "below_reserve": int(sum(int(row["below_reserve"]) for row in group)),
                    "energy_exhausted": int(sum(int(row["energy_exhausted"]) for row in group)),
                }
            )
    return output


def _mechanism_note(answers: dict) -> str:
    parts = []
    if answers["sensitive_utility_gap"] > 0.0:
        parts.append("敏感场景上，允许校准的平均净效用高于关闭校准。")
    else:
        parts.append("敏感场景上，允许校准的平均净效用没有高于关闭校准。")
    if answers["sensitive_bits_gap"] > 0.0:
        parts.append("按时交付比特有增加。")
    else:
        parts.append("按时交付比特没有增加。")
    if answers["sensitive_choice_differences"] > 0:
        parts.append("有配对的首个服务浮标不同。")
    else:
        parts.append("敏感配对的首个服务浮标相同。")
    if answers["calibration_snapshots"] == 0:
        parts.append("开校准方法没有实际执行校准，信息消融为 0 条。")
    elif answers["ablation_utility_gap"] > 0.0:
        parts.append("撤去校准观测后的续跑净效用更低，差值和轨迹见消融表。")
    else:
        parts.append("撤去校准观测后的续跑没有更低的净效用，信息本身没有在这些快照上表现为正贡献。")
    if answers["on_insensitive_cal_s"] < answers["on_sensitive_cal_s"]:
        parts.append("不敏感场景的校准时间更短。")
    else:
        parts.append("不敏感场景没有减少校准时间。")
    parts.append("这些是 8 对样本上的差值，不是置信区间。")
    return "".join(parts)


def _next_step(answers: dict) -> str:
    if answers["on_sensitive_cal_s"] <= 0.0 or answers["sensitive_utility_gap"] <= 0.0:
        return (
            "这组样本里，专门校准没有在同规划器对照下提高净效用，或根本没有被选中。"
            "后续若继续，应把辨识价值并进普通服务航段，而不是再加一种强制校准。"
            "本轮不切换方案，也不自动重跑。"
        )
    return (
        "允许校准在这组敏感样本上提高了净效用。后续若继续，先用新的预注册样本核对消融差是否仍由被撤去的那条观测解释。"
        "本轮不扩大节点、不训练、不自动重跑。"
    )


def _audit_old(spec: dict, factor: float) -> dict:
    if not M1_MANIFEST.exists():
        return {"present": False}
    old = json.loads(M1_MANIFEST.read_text(encoding="utf-8"))
    budget = old["budget"]
    return {
        "present": True,
        "source": str(M1_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "sha256": _sha256(M1_MANIFEST),
        "old_sensitive_battery_j": budget["sensitive_battery_j"],
        "old_insensitive_battery_j": budget["insensitive_battery_j"],
        "old_sensitive_predicates_under_corrected_physics": predicate_rows(spec, factor, budget["sensitive_battery_j"]),
        "old_insensitive_predicates_under_corrected_physics": predicate_rows(spec, factor, budget["insensitive_battery_j"]),
        "old_sensitive_chain_under_corrected_physics": [
            _chain_status(spec, factor, budget["sensitive_battery_j"], index, None) for index in range(2)
        ],
        "note": "Old batteries are audited under the corrected model and are not reused as the M1.1 initial energy.",
    }


def _effects(spec: dict, factor: float, scenarios: list[dict]) -> list[dict]:
    from wt.active_model_identification.world import action_allowed

    rows = []
    guard = prior_belief(spec).quantile(float(spec["belief"]["guard_probability"]))
    for scenario in scenarios:
        per_uav = []
        world = make_world(spec, scenario["battery_j"], scenario["id"])
        for index, theta in enumerate(scenario["theta"]):
            home = spec["geometry"]["uavs"][index]
            far = ("service", int(home["far_buoy"]))
            near = ("service", int(home["near_buoy"]))
            per_uav.append(
                {
                    "uav": index,
                    "true_theta": theta,
                    "far_physical": action_allowed(world.uavs[index], far, float(theta), spec, factor, world),
                    "near_physical": action_allowed(world.uavs[index], near, float(theta), spec, factor, world),
                    "far_under_prior_guard": action_allowed(world.uavs[index], far, guard, spec, factor, world),
                    "near_under_prior_guard": action_allowed(world.uavs[index], near, guard, spec, factor, world),
                }
            )
        rows.append({"id": scenario["id"], "condition": scenario["condition"], "uavs": per_uav})
    return rows


def run_trial(out_dir: Path, tests_log: str | None, force: bool) -> dict:
    if "active_model_identification_m1" in out_dir.as_posix() and "m11" not in out_dir.as_posix():
        raise SystemExit("refusing to write into the M1 result directory")
    if (out_dir / "summary.json").exists() and not force:
        raise SystemExit(f"existing summary at {out_dir}; pass --force to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / "RUNNING"
    if lock.exists() and not force:
        raise SystemExit(f"found {lock}; another run may be unfinished")
    lock.write_text("running\n", encoding="utf-8")
    started = time.perf_counter()
    spec = copy.deepcopy(load_m11_spec())
    integral = select_integral(spec)
    spec["belief"]["quadrature_nodes"] = int(integral["theta_nodes"])
    spec["integral"]["noise_nodes"] = int(integral["noise_nodes"])
    spec["integral"]["tie_abs"] = float(integral["tie_abs"])
    frozen_path = out_dir / "frozen_config.json"
    frozen_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (out_dir / "integral_check.json").write_text(json.dumps(integral, indent=2), encoding="utf-8")
    scenario_status = "FROZEN"
    failure = ""
    assessed = assess_budgets(spec)
    factor = float(assessed["integration_factor"])
    budget = assessed["budget"]
    if assessed["separates"]:
        scenarios = assessed["scenarios"]
    else:
        scenario_status = "INFEASIBLE"
        failure = str(assessed["reason"])
        scenarios = []
    audit = _audit_old(spec, factor) if factor else {"present": M1_MANIFEST.exists(), "note": failure}
    manifest = {
        "scenario_status": scenario_status,
        "failure": failure,
        "scenarios": scenarios,
        "budget": budget,
        "old_budget_audit": audit,
        "task_effects": _effects(spec, factor, assessed["scenarios"]),
        "diagnosis_scenarios_not_run": [] if assessed["separates"] else [item["id"] for item in assessed["scenarios"]],
        "integral_selection": {
            "theta_nodes": integral["theta_nodes"],
            "noise_nodes": integral["noise_nodes"],
            "tie_abs": integral["tie_abs"],
            "measured_q_gap": integral["measured_q_gap"],
        },
    }
    scenario_path = out_dir / "scenario_manifest.json"
    scenario_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stats = {"physical_steps": 0, "planning_steps": 0, "planning_copies": 0}
    ablation_stats = {"physical_steps": 0, "planning_steps": 0, "planning_copies": 0}
    episodes: list[dict] = []
    decisions: list[dict] = []
    deliveries: list[dict] = []
    snapshots = []
    if scenarios:
        for scenario in scenarios:
            for method in METHODS:
                print(f"run {scenario['id']} {method}", flush=True)
                episode, decision_rows, delivery_rows, snapshot = _core_episode(method, scenario, spec, factor, stats)
                episodes.append(episode)
                decisions.extend(decision_rows)
                deliveries.extend(delivery_rows)
                if snapshot is not None:
                    snapshots.append(snapshot)
                _write_csv(out_dir / "episode_rows.csv", episodes)
                _write_csv(out_dir / "decision_rows.csv", decisions)
        for snapshot in snapshots[:4]:
            print(f"ablate {snapshot['scenario']['id']}", flush=True)
            for label, beliefs in (
                ("ablation_updated", snapshot["beliefs_updated"]),
                ("ablation_removed", snapshot["beliefs_removed"]),
            ):
                episode, decision_rows, delivery_rows = _continue(label, snapshot, beliefs, spec, factor, ablation_stats)
                episodes.append(episode)
                decisions.extend(decision_rows)
                deliveries.extend(delivery_rows)
    _write_csv(out_dir / "episode_rows.csv", [row for row in episodes if row["branch"] == "core"])
    _write_csv(out_dir / "decision_rows.csv", decisions)
    _write_csv(out_dir / "deliveries.csv", deliveries)
    pairs = _pair_rows(episodes) if episodes else []
    _write_csv(out_dir / "pair_rows.csv", pairs)
    ablation = [row for row in episodes if str(row["branch"]).startswith("ablation")]
    _write_csv(out_dir / "ablation_rows.csv", ablation)
    answers = _answers(episodes, ablation, len(snapshots[:4])) if episodes else {
        "sensitive_utility_gap": 0.0,
        "insensitive_utility_gap": 0.0,
        "sensitive_bits_gap": 0.0,
        "sensitive_energy_gap_j": 0.0,
        "sensitive_choice_differences": 0,
        "sensitive_pairs": 0,
        "on_sensitive_cal_s": 0.0,
        "on_insensitive_cal_s": 0.0,
        "off_cal_s": 0.0,
        "calibration_snapshots": 0,
        "ablation_utility_gap": 0.0,
    }
    tests = _tests_status(tests_log)
    cores = [row for row in episodes if row["branch"] == "core"]
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "parent_commit": spec["parent_commit"],
        "reviewed_m1_commit": spec["reviewed_m1_commit"],
        "result_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(M11_CONFIG_PATH),
        "frozen_config_sha256": _sha256(frozen_path),
        "scenario_manifest_sha256": _sha256(scenario_path),
        "python": sys.version,
        "platform": platform.platform(),
        "interface_status": "PASS" if tests["ok"] else ("PENDING_TESTS" if not tests_log else "FAIL"),
        "scenario_status": scenario_status,
        "tests": tests,
        "integral": {
            "theta_nodes": integral["theta_nodes"],
            "noise_nodes": integral["noise_nodes"],
            "tie_abs": integral["tie_abs"],
            "measured_q_gap": integral["measured_q_gap"],
            "selection": integral["selection"],
        },
        "counts": {
            "core_missions": len(cores),
            "ablation_branches": len(ablation),
            "physical_steps": int(stats["physical_steps"]),
            "ablation_physical_steps": int(ablation_stats["physical_steps"]),
            "planning_steps": int(stats["planning_steps"] + ablation_stats["planning_steps"]),
            "planning_copies": int(stats["planning_copies"] + ablation_stats["planning_copies"]),
            "development_planning_steps": int(integral["development_planning_steps"]),
            "development_planning_copies": int(integral["development_planning_copies"]),
            "probe_physical_steps": int(budget.get("probe_physical_steps", 0)) if budget else 0,
            "return_failures": int(sum(int(row["return_failures"]) for row in cores)),
            "training_updates": 0,
            "model_forward_passes": 0,
            "elapsed_s": elapsed,
        },
        "answers": answers,
        "method_means": _method_means(episodes) if cores else [],
        "accounting_note": (
            "远端回传和等待保持空中并积分悬停推进。只有到达登记机坪、速度低于机坪门限并完成 5 秒降落悬停后才转地面待机。"
            "回传成本按当前飞行状态的决策间隔悬停或地面待机能耗计算，不再使用 80 J 常数。"
            "完整包的到岸时刻是该分片 LoRa 结束时刻。任务终态分开记录未返航、未降落、低于余量和能量耗尽。"
        ),
        "terminal_note": "单决策间隔物理展开，加只在预计回传、返航、降落和余量都放得下时计入的缓存比特。未计两机信道争用。不是精确贝叶斯最优。",
        "mechanism_note": _mechanism_note(answers) if cores else f"场景没有冻结为可运行的两类任务：{failure}",
        "next_step": _next_step(answers) if cores else "先有能区分物理可行和先验允许的两类任务，再做同规划器对照。本轮没有为了制造对照去搜索新几何或新电量。",
        "commands": {
            "tests": "py -3 -m unittest tests.test_active_model_identification_m11 tests.test_active_model_identification_m1 -v",
            "trial": "py -3 -m wt.active_model_identification.run_m11 --tests-log results/active_model_identification_m11/m11_20261114/necessary_tests.log",
            "pythonpath": "src",
        },
        "failure": failure,
        "git": _git(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "parent_commit": summary["parent_commit"],
                "reviewed_m1_commit": summary["reviewed_m1_commit"],
                "head_at_run": summary["git"]["head"],
                "config_sha256": summary["config_sha256"],
                "frozen_config_sha256": summary["frozen_config_sha256"],
                "scenario_manifest_sha256": summary["scenario_manifest_sha256"],
                "python": sys.version,
                "seed_rule": "observation noise is Random(f'm1-obs-{scenario_seed}-{uav}-{interval}') and does not depend on action order; planning noise is the registered deterministic quadrature",
                "core_missions": len(cores),
                "ablation_branches": len(ablation),
                "physical_steps": summary["counts"]["physical_steps"],
                "ablation_physical_steps": summary["counts"]["ablation_physical_steps"],
                "planning_steps": summary["counts"]["planning_steps"],
                "planning_copies": summary["counts"]["planning_copies"],
                "development_planning_steps": summary["counts"]["development_planning_steps"],
                "probe_physical_steps": summary["counts"]["probe_physical_steps"],
                "elapsed_s": elapsed,
                "training_updates": 0,
                "scenario_status": scenario_status,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_m11_report(summary, ROOT / "docs" / "research" / "Active_Model_Identification_M11_Report_CN.md")
    lock.unlink(missing_ok=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/active_model_identification_m11/m11_20261114")
    parser.add_argument("--tests-log", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_trial(ROOT / args.out, args.tests_log or None, args.force)
    print(
        json.dumps(
            {
                "interface": summary["interface_status"],
                "scenario": summary["scenario_status"],
                "missions": summary["counts"]["core_missions"],
                "ablation": summary["counts"]["ablation_branches"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
