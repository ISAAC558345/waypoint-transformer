"""Run the M1.3 three-budget pilot.

Missions start after the ledger check and the frozen scoring rule. They are not
gated on posterior_zero_admits or on a passing quadrature gap. A NOT_MET gap
uses the pre-registered 15-point / 5-point fallback and keeps that label.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import subprocess
import time
from pathlib import Path

from wt.active_model_identification.belief import prior_belief
from wt.active_model_identification.completion import accounting_cases, clear_completion_cache
from wt.active_model_identification.policies import select_lookahead
from wt.active_model_identification.report_m13 import write_m13_report
from wt.active_model_identification.run_m12 import (
    DECISION_FIELDS,
    _judge_accounting,
    _mean,
    _q_fields,
    _quotes,
    _sha256,
    _tests_status,
    _write_csv,
)
from wt.active_model_identification.scenarios import ablate_beliefs, assess_m13, observation_noise
from wt.active_model_identification.segment import BudgetExceeded, clear_rollout_cache, select_integral_m13
from wt.active_model_identification.spec import M13_CONFIG_PATH, ROOT, decision_count, load_m13_spec
from wt.active_model_identification.world import advance, energy_used_j, make_world, mission_terminal, on_time_bits, utility

METHODS = ("lookahead_calibration_off", "lookahead_calibration_on")
ON = "lookahead_calibration_on"
M13_FIELDS = DECISION_FIELDS + ["scoring_mode", "numerically_sensitive"]
TIERS = ("tight", "window", "loose")


def _git() -> dict:
    try:
        head = subprocess.check_output(["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(
            ["git", "-c", "core.longpaths=true", "status", "--porcelain=v1", "-b"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        return {"head": "unknown", "status": "", "error": str(exc)}
    return {"head": head, "status": status}


def _empty_tier() -> dict:
    return {
        "utility_gap": None,
        "utility_on": None,
        "utility_off": None,
        "bits_gap": None,
        "energy_gap_j": None,
        "calibration_s_on": None,
        "calibration_s_off": None,
        "choice_differences": None,
        "pairs": 0,
        "rows": [],
    }


def _paired(episodes: list[dict], ablation: list[dict]) -> dict:
    cores = [row for row in episodes if row["branch"] == "core"]

    def tier(condition: str) -> dict:
        rows = [row for row in cores if row["condition"] == condition]
        bases = sorted({row["base"] for row in rows})
        if not bases:
            return _empty_tier()
        gaps_u = []
        gaps_b = []
        gaps_e = []
        on_u = []
        off_u = []
        on_c = []
        off_c = []
        choice = 0
        detail = []
        for base in bases:
            left = next(row for row in rows if row["base"] == base and row["method"] == ON)
            right = next(row for row in rows if row["base"] == base and row["method"] == METHODS[0])
            gaps_u.append(float(left["utility"]) - float(right["utility"]))
            gaps_b.append(float(left["on_time_bits"]) - float(right["on_time_bits"]))
            gaps_e.append(float(left["energy_j"]) - float(right["energy_j"]))
            on_u.append(float(left["utility"]))
            off_u.append(float(right["utility"]))
            on_c.append(float(left["calibration_s"]))
            off_c.append(float(right["calibration_s"]))
            same = (left["first_service_0"], left["first_service_1"]) == (right["first_service_0"], right["first_service_1"])
            if not same:
                choice += 1
            detail.append(
                {
                    "base": base,
                    "utility_on": left["utility"],
                    "utility_off": right["utility"],
                    "bits_on": left["on_time_bits"],
                    "bits_off": right["on_time_bits"],
                    "energy_on": left["energy_j"],
                    "energy_off": right["energy_j"],
                    "calibration_s_on": left["calibration_s"],
                    "calibration_s_off": right["calibration_s"],
                    "rmse_on": left["rmse"],
                    "rmse_off": right["rmse"],
                    "returned_on": left["returned"],
                    "returned_off": right["returned"],
                    "choice_changed": not same,
                }
            )
        found = _empty_tier()
        found.update(
            {
                "utility_gap": _mean(gaps_u),
                "utility_on": _mean(on_u),
                "utility_off": _mean(off_u),
                "bits_gap": _mean(gaps_b),
                "energy_gap_j": _mean(gaps_e),
                "calibration_s_on": _mean(on_c),
                "calibration_s_off": _mean(off_c),
                "choice_differences": choice,
                "pairs": len(bases),
                "rows": detail,
            }
        )
        return found

    updated = [row for row in ablation if row["branch"] == "ablation_updated"]
    removed = [row for row in ablation if row["branch"] == "ablation_removed"]
    ablation_gap = _mean(
        [
            float(next(row for row in updated if row["source_scenario"] == item["source_scenario"])["utility"]) - float(item["utility"])
            for item in removed
        ]
    )
    return {"tight": tier("tight"), "window": tier("window"), "loose": tier("loose"), "ablation_utility_gap": ablation_gap}


def _apply_scoring(spec: dict, integral: dict) -> dict:
    spec = copy.deepcopy(spec)
    if integral["scoring"] == "segment" and integral["theta_nodes"] is not None:
        spec["integral"]["scoring"] = "segment"
        spec["belief"]["quadrature_nodes"] = int(integral["theta_nodes"])
        spec["integral"]["segment_inner_nodes"] = int(integral["segment_inner_nodes"])
    else:
        spec["integral"]["scoring"] = "frozen"
        spec["belief"]["quadrature_nodes"] = int(integral["frozen_theta_nodes"])
        spec["integral"]["noise_nodes"] = int(integral["frozen_noise_nodes"])
    spec["integral"]["numerically_sensitive"] = bool(integral["numerically_sensitive"])
    return spec


def _scoring_label(spec: dict) -> str:
    if spec["integral"]["scoring"] == "frozen":
        return "frozen_approximation"
    return "segment"


def _play(method, scenario, spec, factor, stats, branch, start_epoch, world, beliefs, removed_ids, capture=None):
    sigma = float(spec["belief"]["sigma_j"])
    noise = observation_noise(int(scenario["seed"]), sigma)
    tau = float(spec["integral"]["tie_abs"])
    label = _scoring_label(spec)
    sensitive = bool(spec["integral"].get("numerically_sensitive"))
    calls_before = int(stats.get("score_calls", 0))
    fallback_before = int(stats.get("fallback_score_calls", 0))
    decisions = []
    covered = 0
    coverage_n = 0
    first_service = [None, None]
    snapshot = None
    for epoch in range(start_epoch, decision_count(spec)):
        before = [item.copy() for item in beliefs]
        before_copies = int(stats.get("planning_copies", 0))
        requested, tables = select_lookahead(method, world, beliefs, spec, factor, stats)
        if capture is not None:
            capture(epoch, world, beliefs, requested)
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
                    "scoring_mode": label,
                    "numerically_sensitive": sensitive,
                    **scores,
                }
            )
            executed = obs["executed"]
            if first_service[uav] is None and executed[0] == "service":
                first_service[uav] = executed[1]
            if obs["telemetry_sent"] and float(obs["z_j"]) > 100.0:
                coverage_n += 1
                covered += int(bool(obs["covered_90"]))
        if snapshot is None and branch == "core" and method == ON and scenario["condition"] == "window":
            if any(obs["executed"][0] == "calibrate" for obs in observations):
                executed_actions = [tuple(obs["executed"]) for obs in observations]
                ablated, removed_uavs = ablate_beliefs(before, beliefs, executed_actions)
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
    calls = int(stats.get("score_calls", 0)) - calls_before
    fallback = int(stats.get("fallback_score_calls", 0)) - fallback_before
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
        "score_calls": calls,
        "fallback_score_calls": fallback,
        "unconverged_score_calls": fallback if sensitive and label == "frozen_approximation" else 0,
        "scoring_mode": label,
    }
    deliveries = []
    for item in world.deliveries:
        stamped = dict(item)
        stamped.update({"branch": branch, "method": method, "scenario_id": scenario["id"]})
        deliveries.append(stamped)
    return episode, decisions, deliveries, snapshot


def _load_progress(path: Path) -> tuple[set[str], list[dict], list[dict], list[dict]]:
    done: set[str] = set()
    episodes: list[dict] = []
    decisions: list[dict] = []
    deliveries: list[dict] = []
    if not path.exists():
        return done, episodes, decisions, deliveries
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        done.add(record["key"])
        episodes.append(record["episode"])
        decisions.extend(record["decisions"])
        deliveries.extend(record["deliveries"])
    return done, episodes, decisions, deliveries


def _append_progress(path: Path, key: str, episode: dict, decisions: list[dict], deliveries: list[dict]) -> None:
    record = {"key": key, "episode": episode, "decisions": decisions, "deliveries": deliveries}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _stats() -> dict:
    return {
        "physical_steps": 0,
        "planning_steps": 0,
        "planning_copies": 0,
        "rollouts": 0,
        "cache_hits": 0,
        "score_calls": 0,
        "fallback_score_calls": 0,
        "segment_score_calls": 0,
        "root_solves": 0,
        "algebraic_integrals": 0,
        "algebraic_seconds": 0.0,
    }


def _load_stats(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return _stats()


def _save_stats(path: Path, stats: dict) -> None:
    path.write_text(json.dumps(stats), encoding="utf-8")


def _refuse(out_dir: Path) -> None:
    parts = set(out_dir.parts)
    if "active_model_identification_m13" not in parts:
        raise SystemExit("refusing to write outside results/active_model_identification_m13")
    banned = {"active_model_identification_m1", "active_model_identification_m11", "active_model_identification_m12"}
    if parts & banned:
        raise SystemExit("refusing to write into an earlier result directory")


def _prepare(out_dir: Path, tests_log: str | None, force: bool) -> tuple[dict, dict, dict, dict, bool]:
    frozen_path = out_dir / "frozen_config.json"
    resume = (not force) and frozen_path.exists() and (out_dir / "scenario_manifest.json").exists() and (out_dir / "integral_check.json").exists()
    if resume:
        spec = json.loads(frozen_path.read_text(encoding="utf-8"))
        manifest = json.loads((out_dir / "scenario_manifest.json").read_text(encoding="utf-8"))
        integral = json.loads((out_dir / "integral_check.json").read_text(encoding="utf-8"))
        accounting = json.loads((out_dir / "accounting_check.json").read_text(encoding="utf-8"))
        return spec, manifest, integral, accounting, True
    spec = copy.deepcopy(load_m13_spec())
    print("accounting", flush=True)
    accounting = _judge_accounting(spec, accounting_cases(spec))
    print("three budgets", flush=True)
    assessed = assess_m13(spec)
    factor = float(assessed["integration_factor"])
    print("segmented integral", flush=True)
    integral = select_integral_m13(spec, factor)
    clear_rollout_cache()
    mission_spec = _apply_scoring(spec, integral)
    mission_spec["integral_result"] = {
        "numerical_status": integral["numerical_status"],
        "scoring": mission_spec["integral"]["scoring"],
        "theta_nodes": integral["theta_nodes"],
        "segment_inner_nodes": integral["segment_inner_nodes"],
        "frozen_theta_nodes": integral["frozen_theta_nodes"],
        "frozen_noise_nodes": integral["frozen_noise_nodes"],
        "measured_q_gap": integral["measured_q_gap"],
        "tie_abs": integral["tie_abs"],
    }
    frozen_path.write_text(json.dumps(mission_spec, indent=2), encoding="utf-8")
    (out_dir / "integral_check.json").write_text(json.dumps(integral, indent=2), encoding="utf-8")
    (out_dir / "accounting_check.json").write_text(json.dumps(accounting, indent=2), encoding="utf-8")
    manifest = {
        "scenario_status": assessed["scenario_status"],
        "window_valid": assessed["window_valid"],
        "window_status": assessed["window_status"],
        "posterior_zero_admits": assessed["posterior_zero_admits"],
        "limiting_condition": assessed["limiting_condition"],
        "budget": {key: value for key, value in assessed.items() if key != "scenarios"},
        "scenarios": assessed["scenarios"],
        "config_sha256": _sha256(M13_CONFIG_PATH),
        "frozen_config_sha256": _sha256(frozen_path),
        "unit_tests_status": _tests_status(tests_log),
    }
    scenario_path = out_dir / "scenario_manifest.json"
    scenario_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["scenario_manifest_sha256"] = _sha256(scenario_path)
    scenario_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["scenario_manifest_sha256"] = _sha256(scenario_path)
    return mission_spec, manifest, integral, accounting, False


def _rescore(spec: dict, factor: float, saved: dict, out_dir: Path) -> dict:
    if not saved:
        return {"snapshots": 0, "rank_changes": 0, "cap_hit": False, "rollouts": 0, "rows": []}
    fine = copy.deepcopy(spec)
    fine["integral"]["scoring"] = "segment"
    fine["belief"]["quadrature_nodes"] = int(spec["integral"]["rescore_theta_nodes"])
    fine["integral"]["segment_inner_nodes"] = int(spec["integral"]["rescore_inner_nodes"])
    per_cap = int(spec["integral"]["rescore_snapshot_rollout_cap"])
    total_cap = int(spec["integral"]["rescore_total_rollout_cap"])
    clear_rollout_cache()
    used = 0
    changes = 0
    cap_hit = False
    rows = []
    for key, item in list(saved.items())[:12]:
        remaining = total_cap - used
        if remaining <= 0:
            cap_hit = True
            break
        local = _stats()
        local["rollout_cap"] = min(per_cap, remaining)
        try:
            actions, _tables = select_lookahead(ON, item["world"], item["beliefs"], fine, factor, local)
            fine_actions = [tuple(action) for action in actions]
            stored = [tuple(action) for action in item["actions_on"]]
            changed = fine_actions != stored
        except BudgetExceeded:
            cap_hit = True
            changed = None
            fine_actions = None
        used += int(local.get("rollouts", 0))
        if changed:
            changes += 1
        rows.append(
            {
                "key": f"{key[0]}_{key[1]}",
                "epoch": item["epoch"],
                "reason": item["reason"],
                "stored_actions": [f"{action[0]}:{action[1]}" for action in item["actions_on"]],
                "rescored_actions": None if fine_actions is None else [f"{action[0]}:{action[1]}" for action in fine_actions],
                "rank_changed": changed,
                "rollouts": int(local.get("rollouts", 0)),
            }
        )
        if cap_hit and fine_actions is None:
            break
    result = {"snapshots": len(rows), "rank_changes": changes, "cap_hit": cap_hit, "rollouts": used, "rows": rows}
    (out_dir / "rescore.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_trial(out_dir: Path, tests_log: str | None, force: bool) -> dict:
    _refuse(out_dir)
    if (out_dir / "summary.json").exists() and not force:
        raise SystemExit(f"existing summary at {out_dir}; pass --force to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "completed_missions.jsonl"
    if force:
        for name in ("completed_missions.jsonl", "mission_stats.json", "ablation_stats.json", "summary.json"):
            target = out_dir / name
            if target.exists():
                target.unlink()
    started = time.perf_counter()
    spec, manifest, integral, accounting, resumed = _prepare(out_dir, tests_log, force)
    run_missions = accounting["status"] == "PASS"
    done, episodes, decisions, deliveries = _load_progress(progress_path)
    stats = _load_stats(out_dir / "mission_stats.json")
    ablation_stats = _load_stats(out_dir / "ablation_stats.json")
    snapshots = []
    rescore_inputs = {}
    snap_dir = out_dir / "snapshots"
    if snap_dir.exists():
        for path in sorted(snap_dir.glob("*.pkl")):
            snapshots.append(pickle.loads(path.read_bytes()))
    input_dir = out_dir / "rescore_inputs"
    if input_dir.exists():
        for path in sorted(input_dir.glob("*.pkl")):
            key, payload = pickle.loads(path.read_bytes())
            rescore_inputs[tuple(key)] = payload
    factor = float(manifest["budget"]["integration_factor"])
    off_traces: dict[str, list] = {}
    if run_missions:
        for scenario in manifest["scenarios"]:
            clear_rollout_cache()
            off_traces[scenario["id"]] = []

            def capture_off(epoch, world, beliefs, requested, _scenario=scenario):
                _ = epoch, world, beliefs
                off_traces[_scenario["id"]].append([tuple(action) for action in requested])

            def capture_on(epoch, world, beliefs, requested, _scenario=scenario):
                key = (_scenario["base"], _scenario["condition"])
                if key in rescore_inputs:
                    return
                previous = off_traces[_scenario["id"]]
                stored_off = previous[epoch] if epoch < len(previous) else None
                differ = stored_off is not None and [tuple(action) for action in requested] != stored_off
                calibrates = any(action[0] == "calibrate" for action in requested)
                if not differ and not calibrates:
                    return
                payload = {
                    "epoch": epoch,
                    "world": world.snapshot(),
                    "beliefs": [item.copy() for item in beliefs],
                    "actions_on": [tuple(action) for action in requested],
                    "actions_off": stored_off,
                    "reason": "disagreement" if differ else "calibration",
                }
                rescore_inputs[key] = payload
                folder = out_dir / "rescore_inputs"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{key[0]}_{key[1]}.pkl").write_bytes(pickle.dumps((key, payload)))

            for method in METHODS:
                key = f"core|{scenario['id']}|{method}"
                if key in done:
                    print(f"resume skip {key}", flush=True)
                    continue
                print(f"run {scenario['id']} {method}", flush=True)
                world = make_world(spec, scenario["battery_j"], f"{scenario['id']}-{method}")
                beliefs = [prior_belief(spec), prior_belief(spec)]
                capture = capture_on if method == ON else capture_off
                episode, decision_rows, delivery_rows, snapshot = _play(
                    method, scenario, spec, factor, stats, "core", 0, world, beliefs, [], capture
                )
                episodes.append(episode)
                decisions.extend(decision_rows)
                deliveries.extend(delivery_rows)
                _append_progress(progress_path, key, episode, decision_rows, delivery_rows)
                done.add(key)
                _save_stats(out_dir / "mission_stats.json", stats)
                if snapshot is not None:
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    (snap_dir / f"{scenario['id']}.pkl").write_bytes(pickle.dumps(snapshot))
                    snapshots.append(snapshot)
        for snapshot in snapshots[:4]:
            for label, beliefs in (
                ("ablation_updated", snapshot["beliefs_updated"]),
                ("ablation_removed", snapshot["beliefs_removed"]),
            ):
                key = f"{label}|{snapshot['scenario']['id']}|lookahead_calibration_off"
                if key in done:
                    continue
                print(f"ablate {snapshot['scenario']['id']} {label}", flush=True)
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
                episode["source_scenario"] = snapshot["scenario"]["id"]
                episode["removed_uavs"] = snapshot["removed_uavs"]
                episodes.append(episode)
                decisions.extend(decision_rows)
                deliveries.extend(delivery_rows)
                _append_progress(progress_path, key, episode, decision_rows, delivery_rows)
                done.add(key)
                _save_stats(out_dir / "ablation_stats.json", ablation_stats)
    comparison = "RAN" if run_missions else "NOT_RUN"
    if (out_dir / "rescore.json").exists() and not force:
        rescore = json.loads((out_dir / "rescore.json").read_text(encoding="utf-8"))
    elif run_missions:
        print("rescore", flush=True)
        rescore = _rescore(spec, factor, rescore_inputs, out_dir)
    else:
        rescore = {"snapshots": 0, "rank_changes": 0, "cap_hit": False, "rollouts": 0, "rows": []}
    cores = [row for row in episodes if row["branch"] == "core"]
    ablation_rows = [row for row in episodes if str(row["branch"]).startswith("ablation")]
    paired = _paired(episodes, ablation_rows)
    counts = {
        "core_missions": len(cores),
        "ablation_branches": len(ablation_rows),
        "design_probe_intervals": int(manifest["budget"]["design_probe_intervals"]),
        "design_probe_steps": int(manifest["budget"]["design_probe_steps"]),
        "development_rollouts": int(integral.get("development_rollouts", 0)),
        "development_planning_steps": int(integral.get("development_planning_steps", 0)),
        "development_cache_hits": int(integral.get("cache_hits", 0)),
        "development_root_solves": int(integral.get("root_solves", 0)),
        "development_algebraic_integrals": int(integral.get("algebraic_integrals", 0)),
        "development_algebraic_seconds": float(integral.get("algebraic_seconds", 0.0)),
        "mission_rollouts": int(stats.get("rollouts", 0)),
        "mission_planning_steps": int(stats.get("planning_steps", 0)),
        "mission_physical_steps": int(stats.get("physical_steps", 0)),
        "mission_cache_hits": int(stats.get("cache_hits", 0)),
        "ablation_rollouts": int(ablation_stats.get("rollouts", 0)),
        "ablation_planning_steps": int(ablation_stats.get("planning_steps", 0)),
        "ablation_physical_steps": int(ablation_stats.get("physical_steps", 0)),
        "rescore_rollouts": int(rescore.get("rollouts", 0)),
        "rescore_snapshots": int(rescore.get("snapshots", 0)),
    }
    elapsed = time.perf_counter() - started
    sensitive = bool(spec["integral"].get("numerically_sensitive")) or int(rescore.get("rank_changes") or 0) > 0
    summary = {
        "run_id": out_dir.name,
        "result_dir": out_dir.as_posix(),
        "parent_commit": load_m13_spec()["parent_commit"],
        "command": "py -3 -m wt.active_model_identification.run_m13 --tests-log results/active_model_identification_m13/m13_20261114/necessary_tests.log",
        "resumed": resumed,
        "unit_tests_status": _tests_status(tests_log),
        "accounting_status": accounting["status"],
        "numerical_status": integral["numerical_status"],
        "scenario_status": manifest["scenario_status"],
        "comparison_status": comparison,
        "scoring_mode": _scoring_label(spec),
        "numerically_sensitive": sensitive,
        "budget": manifest["budget"],
        "integral": {key: value for key, value in integral.items() if key != "states"},
        "paired": paired,
        "rescore": rescore,
        "counts": counts,
        "git": _git(),
        "config_sha256": manifest.get("config_sha256"),
        "frozen_config_sha256": manifest.get("frozen_config_sha256"),
        "scenario_manifest_sha256": manifest.get("scenario_manifest_sha256"),
        "elapsed_s": elapsed,
        "training_updates": 0,
    }
    _write_csv(out_dir / "episodes.csv", episodes)
    _write_csv(out_dir / "decisions.csv", decisions, M13_FIELDS)
    _write_csv(out_dir / "deliveries.csv", deliveries)
    _write_csv(out_dir / "pairs.csv", [row for tier in TIERS for row in paired[tier]["rows"]])
    _write_csv(out_dir / "ablation.csv", ablation_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "result_dir": summary["result_dir"],
                "numerical_status": summary["numerical_status"],
                "scenario_status": summary["scenario_status"],
                "comparison_status": summary["comparison_status"],
                "scoring_mode": summary["scoring_mode"],
                "core_missions": counts["core_missions"],
                "ablation_branches": counts["ablation_branches"],
                "window_valid": manifest["budget"]["window_valid"],
                "config_sha256": summary["config_sha256"],
                "frozen_config_sha256": summary["frozen_config_sha256"],
                "elapsed_s": elapsed,
                "training_updates": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_m13_report(summary, ROOT / "docs" / "research" / "Active_Model_Identification_M13_Report_CN.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/active_model_identification_m13/m13_20261114")
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
                "scoring": summary["scoring_mode"],
                "missions": summary["counts"]["core_missions"],
                "window": summary["budget"]["window_valid"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
