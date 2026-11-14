"""Attribute M1.3 calibration choices without changing the planner or the batteries.

Logs and the six registered snapshots are read only. A continuation runs only
when the one-step split still mixes the terminal value with the information
premium. The rollout caps in the M1.4 instruction are not raised.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import pickle
import subprocess
import time
from pathlib import Path

from wt.active_model_identification.attribution import (
    SNAPSHOTS,
    attribute,
    choose_alternative,
    classify_requests,
    decompose_snapshot,
    duration_of,
    identity_terms,
    judge_sign,
    motion_notes,
    needs_continuation,
    origin_of,
    pair_rows,
    split_action,
)
from wt.active_model_identification.completion import clear_completion_cache
from wt.active_model_identification.report_m14 import write_m14_report
from wt.active_model_identification.run_m12 import _tests_status, _write_csv
from wt.active_model_identification.run_m13 import _play, _stats
from wt.active_model_identification.scenarios import ablate_beliefs, observation_noise
from wt.active_model_identification.segment import BudgetExceeded, clear_rollout_cache
from wt.active_model_identification.spec import ROOT
from wt.active_model_identification.world import advance, energy_used_j, on_time_bits, utility

M13_DIR = ROOT / "results" / "active_model_identification_m13" / "m13_20261114"
REVIEWED_RESULT = "57d3da36969e017d044025e90f90e4e43f0a38e8"
Q_ROLLOUT_CAP = 6000
CONTINUATION_ROLLOUT_CAP = 200000
ON = "lookahead_calibration_on"


def _refuse(out_dir: Path) -> None:
    parts = set(out_dir.parts)
    if "active_model_identification_m14" not in parts:
        raise SystemExit("refusing to write outside results/active_model_identification_m14")
    banned = {
        "active_model_identification_m1",
        "active_model_identification_m11",
        "active_model_identification_m12",
        "active_model_identification_m13",
    }
    if parts & banned:
        raise SystemExit("refusing to write into an earlier result directory")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> dict:
    try:
        head = subprocess.check_output(["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError as exc:
        return {"head": "unknown", "error": str(exc)}
    return {"head": head}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _origin_summary(rows: list[dict]) -> dict:
    def _count(items, key):
        found = {}
        for item in items:
            found[str(item[key])] = found.get(str(item[key]), 0) + 1
        return found

    single = [row for row in rows if row["origin"] == "single_score"]
    repair = [row for row in rows if row["origin"] == "conflict_repair"]
    gaps = [row["Q_gap"] for row in single if row["Q_gap"] is not None]
    return {
        "requests": len(rows),
        "single_score": len(single),
        "conflict_repair": len(repair),
        "return_guard_replacements": sum(1 for row in rows if row["intervention"]),
        "full_30s": sum(1 for row in rows if row["duration_s"] == 30.0),
        "duration_unknown": sum(1 for row in rows if row["duration_s"] is None),
        "single_score_gap_min": min(gaps) if gaps else None,
        "single_score_gap_max": max(gaps) if gaps else None,
        "by_scenario": _count(rows, "scenario_id"),
        "by_condition": _count(rows, "condition"),
        "by_origin_condition": {
            "single_score": _count(single, "condition"),
            "conflict_repair": _count(repair, "condition"),
        },
    }


def _core_terminals(episodes: list[dict]) -> dict:
    cores = [row for row in episodes if row["branch"] == "core"]
    on = [row for row in cores if row["method"] == ON]
    off = [row for row in cores if row["method"] != ON]

    def _sum(rows, key):
        return int(sum(int(row[key]) for row in rows))

    keys = ("not_returned", "not_landed", "below_reserve", "energy_exhausted", "return_failures", "returned")
    return {
        "missions": len(cores),
        "on": {key: _sum(on, key) for key in keys},
        "off": {key: _sum(off, key) for key in keys},
    }


def _blank_split() -> dict:
    return {
        "q_plus": None,
        "q_minus": None,
        "r30": None,
        "terminal_plus": None,
        "terminal_minus": None,
        "propulsion_j": None,
        "aux_j": None,
        "radio_j": None,
        "compute_j": None,
        "prior_q95": None,
        "posterior_q95_eps0": None,
        "predicted_bits_prior": None,
        "predicted_bits_posterior": None,
        "active_prior": "",
        "active_posterior": "",
        "plans_json": "",
        "executed": "",
        "intervened": "",
        "end_x": None,
        "end_y": None,
        "calibration_s": None,
        "table_q": None,
    }


def _from_split(split: dict, table_q: float) -> dict:
    detail = split["detail"]
    return {
        "q_plus": split["q_plus"],
        "q_minus": split["q_minus"],
        "r30": split["r30"],
        "terminal_plus": split["terminal_plus"],
        "terminal_minus": split["terminal_minus"],
        "propulsion_j": split["propulsion_j"],
        "aux_j": split["aux_j"],
        "radio_j": split["radio_j"],
        "compute_j": split["compute_j"],
        "prior_q95": detail["prior_q95"],
        "posterior_q95_eps0": detail["posterior_q95_eps0"],
        "predicted_bits_prior": detail["predicted_bits_prior"],
        "predicted_bits_posterior": detail["predicted_bits_posterior"],
        "active_prior": detail["active_prior"],
        "active_posterior": detail["active_posterior"],
        "plans_json": json.dumps(detail["plans"]),
        "executed": f"{detail['executed'][0]}:{detail['executed'][1]}",
        "intervened": detail["intervened"],
        "end_x": detail["x"],
        "end_y": detail["y"],
        "calibration_s": detail["calibration_s"],
        "table_q": table_q,
    }


def _row(snapshot: str, registered: str, common: dict, role: str, action: str, values: dict) -> dict:
    item = {
        "snapshot": snapshot,
        "registered_class": registered,
        "role": role,
        "action": action,
        "state_cost": "",
        "info_premium_c": "",
        "info_premium_e": "",
        "gap_plus": "",
        "identity_residual": "",
        "immediate_gap": "",
        "terminal_minus_gap": "",
        "attribution": "",
        "sign_status": "",
        "fine_gap": "",
        "exclusion_json": "",
        "motion_notes": "",
        "context_other_is_wait": "",
        "other_uav_belief_enters_focus_terminal": False,
    }
    item.update(common)
    item.update(values)
    return item


def _analyze_one(name, registered, payload, spec, factor, stats, log_rows) -> dict:
    world = payload["world"]
    beliefs = payload["beliefs"]
    base = decompose_snapshot(world, beliefs, spec, factor, stats, payload["actions_on"])
    epoch = int(payload["epoch"])
    logged = [row for row in log_rows if int(row["epoch"]) == epoch]
    by_uav = {int(row["uav"]): row for row in logged}
    status = "COMPLETE" if base["stored_match"] else "log_mismatch"
    if not base["focus_uavs"]:
        status = "no_focus_calibration"
    common = {
        "epoch": epoch,
        "stored_match": base["stored_match"],
        "repair_applied": base["repair_applied"],
        "weaker_uav": "" if base["weaker_uav"] is None else base["weaker_uav"],
        "table_choice_0": base["table_choice"][0],
        "table_choice_1": base["table_choice"][1],
        "joint_0": base["joint_action"][0],
        "joint_1": base["joint_action"][1],
        "stored_0": base["stored_action"][0],
        "stored_1": base["stored_action"][1],
        "status": status,
        "log_origin": "",
        "focus_uav": "",
        "partner_uav": "",
        "partner_joint_action": "",
    }
    rows = []
    for uav, table in enumerate(base["tables"]):
        for score, action in table:
            role = "partner_table" if uav not in base["focus_uavs"] else "table"
            values = _blank_split()
            values["table_q"] = score
            values["q_plus"] = score
            rows.append(_row(name, registered, common, role, f"{action[0]}:{action[1]}", values))
    focus_records = []
    if status != "COMPLETE":
        return {"status": status, "rows": rows, "focuses": focus_records, "continuation": None}
    tie = float(spec["integral"]["tie_abs"])
    residual = float(spec["integral_result"]["measured_q_gap"])
    for focus in base["focus_uavs"]:
        partner = 1 - focus
        partner_action = base["actions"][partner]
        table = base["tables"][focus]
        table_q = {f"{action[0]}:{action[1]}": score for score, action in table}
        e_score, e_action, notes = choose_alternative(table, partner_action, world, beliefs[focus], spec, factor, focus)
        log = by_uav.get(focus)
        log_origin = "" if log is None else origin_of(str(log["table_choice"]))
        focus_common = dict(common)
        focus_common.update(
            {
                "log_origin": log_origin,
                "focus_uav": focus,
                "partner_uav": partner,
                "partner_joint_action": f"{partner_action[0]}:{partner_action[1]}",
                "context_other_is_wait": partner_action == ("wait", None),
            }
        )
        if e_action is None:
            focus_records.append({"status": "no_legal_alternative", "attribution": "sign_unresolved", "snapshot": name})
            rows.append(_row(name, registered, focus_common, "E", "", {**_blank_split(), "exclusion_json": json.dumps(notes)}))
            continue
        before = int(stats.get("rollouts", 0))
        calibration = split_action(world, beliefs, spec, factor, focus, ("calibrate", None), stats)
        alternative = split_action(world, beliefs, spec, factor, focus, e_action, stats)
        split_rollouts = int(stats.get("rollouts", 0)) - before
        terms = identity_terms(calibration["q_plus"], calibration["q_minus"], alternative["q_plus"], alternative["q_minus"])
        fine_gap = None
        if abs(terms["gap_plus"]) > tie and abs(terms["gap_plus"]) <= residual:
            fine = copy.deepcopy(spec)
            fine["belief"]["quadrature_nodes"] = int(spec["integral"]["rescore_theta_nodes"])
            fine["integral"]["segment_inner_nodes"] = int(spec["integral"]["rescore_inner_nodes"])
            try:
                fine_c = split_action(world, beliefs, spec=fine, factor=factor, uav_index=focus, action=("calibrate", None), stats=stats)
                fine_e = split_action(world, beliefs, fine, factor, focus, e_action, stats)
                fine_gap = fine_c["q_plus"] - fine_e["q_plus"]
            except BudgetExceeded:
                stats["budget_exhausted"] = True
                fine_gap = None
        sign_status = judge_sign(terms["gap_plus"], tie, residual, fine_gap)
        terminal_gap = calibration["terminal_minus"] - alternative["terminal_minus"]
        label = attribute(terms, terminal_gap, tie, sign_status)
        notes_text = motion_notes(calibration, alternative, tie)
        identity = {
            "state_cost": terms["state_cost"],
            "info_premium_c": terms["info_premium_c"],
            "info_premium_e": terms["info_premium_e"],
            "gap_plus": terms["gap_plus"],
            "identity_residual": terms["residual"],
            "immediate_gap": calibration["r30"] - alternative["r30"],
            "terminal_minus_gap": terminal_gap,
            "attribution": label,
            "sign_status": sign_status,
            "fine_gap": "" if fine_gap is None else fine_gap,
            "exclusion_json": json.dumps(notes),
            "motion_notes": ";".join(notes_text),
            "context_other_is_wait": partner_action == ("wait", None),
        }
        for role, action, split in (
            ("C", ("calibrate", None), calibration),
            ("E", e_action, alternative),
        ):
            label_action = f"{action[0]}:{action[1]}"
            values = _from_split(split, table_q.get(label_action))
            values.update(identity if role == "C" else {})
            if role == "E":
                values["exclusion_json"] = json.dumps(notes)
            rows.append(_row(name, registered, focus_common, role, label_action, values))
        table_gap = None
        if "calibrate:None" in table_q and f"{e_action[0]}:{e_action[1]}" in table_q:
            table_gap = table_q["calibrate:None"] - table_q[f"{e_action[0]}:{e_action[1]}"]
        focus_records.append(
            {
                "status": "COMPLETE",
                "snapshot": name,
                "registered_class": registered,
                "log_origin": log_origin,
                "class_matches_log": log_origin == registered,
                "epoch": epoch,
                "focus_uav": focus,
                "partner_uav": partner,
                "partner_action": [partner_action[0], partner_action[1]],
                "alternative": [e_action[0], e_action[1]],
                "alternative_table_q": e_score,
                "attribution": label,
                "sign_status": sign_status,
                "gap_plus": terms["gap_plus"],
                "table_gap": table_gap,
                "table_split_abs": None if table_gap is None else abs(table_gap - terms["gap_plus"]),
                "state_cost": terms["state_cost"],
                "info_premium_c": terms["info_premium_c"],
                "info_premium_e": terms["info_premium_e"],
                "immediate_gap": calibration["r30"] - alternative["r30"],
                "terminal_minus_gap": terminal_gap,
                "identity_residual": terms["residual"],
                "fine_gap": fine_gap,
                "motion_notes": notes_text,
                "context_other_is_wait": partner_action == ("wait", None),
                "split_rollouts": split_rollouts,
                "calibration_end": [calibration["detail"]["x"], calibration["detail"]["y"]],
                "alternative_end": [alternative["detail"]["x"], alternative["detail"]["y"]],
                "calibration_propulsion_j": calibration["propulsion_j"],
                "alternative_propulsion_j": alternative["propulsion_j"],
                "calibration_r30": calibration["r30"],
                "alternative_r30": alternative["r30"],
                "predicted_bits_c_posterior": calibration["detail"]["predicted_bits_posterior"],
                "predicted_bits_e_posterior": alternative["detail"]["predicted_bits_posterior"],
                "active_posterior_c": calibration["detail"]["active_posterior"],
                "active_posterior_e": alternative["detail"]["active_posterior"],
            }
        )
    return {"status": status, "rows": rows, "focuses": focus_records}


def _world_match(left, right) -> bool:
    if abs(left.t - right.t) > 1e-6:
        return False
    for first, second in zip(left.uavs, right.uavs):
        if abs(first.x - second.x) > 1e-4 or abs(first.y - second.y) > 1e-4:
            return False
        if abs(first.energy_j - second.energy_j) > 1e-3:
            return False
        if first.airborne != second.airborne or tuple(first.cache_ids) != tuple(second.cache_ids):
            return False
    if len(left.packets) != len(right.packets):
        return False
    for first, second in zip(left.packets, right.packets):
        if first.holder != second.holder or abs(first.remaining_bits - second.remaining_bits) > 1e-4:
            return False
    return True


def _belief_match(left, right) -> bool:
    return all(abs(a.mean() - b.mean()) < 1e-6 and abs(a.variance() - b.variance()) < 1e-8 for a, b in zip(left, right))


def _prefix(world, spec) -> dict:
    return {
        "utility": utility(world, spec),
        "bits": on_time_bits(world),
        "energy_j": sum(energy_used_j(world)),
    }


def _num(episode: dict | None, key: str) -> float | None:
    if episode is None or episode.get(key) in (None, ""):
        return None
    return float(episode[key])


def _branch_record(snapshot, arm, status, source, prefix, episode, decisions, forced, rollouts) -> dict:
    utility = _num(episode, "utility")
    bits = _num(episode, "on_time_bits")
    energy = _num(episode, "energy_j")
    delta_u = None if utility is None else utility - prefix["utility"]
    delta_b = None if bits is None else bits - prefix["bits"]
    delta_e = None if energy is None else energy - prefix["energy_j"]
    return {
        "snapshot": snapshot,
        "arm": arm,
        "status": status,
        "source": source,
        "utility": utility,
        "on_time_bits": bits,
        "energy_j": energy,
        "delta_utility": delta_u,
        "delta_bits": delta_b,
        "delta_energy_j": delta_e,
        "returned": None if episode is None else episode["returned"],
        "not_returned": None if episode is None else episode["not_returned"],
        "not_landed": None if episode is None else episode["not_landed"],
        "below_reserve": None if episode is None else episode["below_reserve"],
        "energy_exhausted": None if episode is None else episode["energy_exhausted"],
        "return_failures": None if episode is None else episode["return_failures"],
        "forced_action": forced.get("action", ""),
        "partner_action": forced.get("partner", ""),
        "forced_executed": forced.get("executed", ""),
        "rollouts": rollouts,
        "decisions": decisions,
    }


def _try_reuse(name, focus, partner_action, epoch, world, beliefs, scenario, spec, factor, episodes, decisions) -> dict | None:
    if scenario["condition"] != "window":
        return None
    path = M13_DIR / "snapshots" / f"{name}.pkl"
    if not path.exists():
        return None
    saved = pickle.loads(path.read_bytes())
    if not saved.get("removed_uavs"):
        return None
    if int(saved["next_epoch"]) != epoch + 1 or int(saved["removed_uavs"][0]) != focus:
        return None
    snap = world.snapshot()
    copied = [item.copy() for item in beliefs]
    noise = observation_noise(int(scenario["seed"]), float(spec["belief"]["sigma_j"]))
    joint = [("wait", None), ("wait", None)]
    joint[focus] = ("calibrate", None)
    joint[1 - focus] = tuple(partner_action)
    observations = advance(
        snap,
        joint,
        list(scenario["theta"]),
        copied,
        spec,
        factor,
        None,
        planning=False,
        interval_index=epoch,
        noise_j=noise,
    )
    executed = [tuple(obs["executed"]) for obs in observations]
    removed_beliefs, removed_uavs = ablate_beliefs([item.copy() for item in beliefs], copied, executed)
    if removed_uavs != [focus]:
        return None
    if not _world_match(snap, saved["world"]) or not _belief_match(copied, saved["beliefs_updated"]):
        return None
    if not _belief_match(removed_beliefs, saved["beliefs_removed"]):
        return None
    prefix = _prefix(world, spec)
    arms = {}
    for arm, branch in (("C_plus", "ablation_updated"), ("C_minus", "ablation_removed")):
        episode = next(row for row in episodes if row["branch"] == branch and row["scenario_id"] == name)
        arm_decisions = [row for row in decisions if row["branch"] == branch and row["scenario_id"] == name]
        arms[arm] = _branch_record(
            name,
            arm,
            "REUSED",
            "m13_window_ablation",
            prefix,
            episode,
            arm_decisions,
            {"action": "calibrate:None", "partner": f"{partner_action[0]}:{partner_action[1]}", "executed": f"{executed[focus][0]}:{executed[focus][1]}"},
            0,
        )
    return arms


def _run_new_branch(name, arm, focus, focus_action, partner, partner_action, remove, epoch, world, beliefs, scenario, spec, factor, stats) -> dict:
    prefix = _prefix(world, spec)
    pre_beliefs = [item.copy() for item in beliefs]
    noise = observation_noise(int(scenario["seed"]), float(spec["belief"]["sigma_j"]))
    joint = [("wait", None), ("wait", None)]
    joint[focus] = tuple(focus_action)
    joint[partner] = tuple(partner_action)
    before = int(stats.get("rollouts", 0))
    observations = advance(
        world,
        joint,
        list(scenario["theta"]),
        beliefs,
        spec,
        factor,
        stats,
        planning=False,
        interval_index=epoch,
        noise_j=noise,
    )
    executed = [tuple(obs["executed"]) for obs in observations]
    obs_id = f"u{focus}-i{epoch}"
    removed = []
    if remove:
        beliefs[focus] = pre_beliefs[focus].copy()
        if obs_id not in world.assimilated_ids:
            world.assimilated_ids.append(obs_id)
        removed = [obs_id]
    episode, decisions, _deliveries, _snapshot = _play(
        "lookahead_calibration_off",
        scenario,
        spec,
        factor,
        stats,
        arm,
        epoch + 1,
        world,
        beliefs,
        removed,
    )
    return _branch_record(
        name,
        arm,
        "RAN",
        "new",
        prefix,
        episode,
        decisions,
        {
            "action": f"{focus_action[0]}:{focus_action[1]}",
            "partner": f"{partner_action[0]}:{partner_action[1]}",
            "executed": f"{executed[focus][0]}:{executed[focus][1]}",
        },
        int(stats.get("rollouts", 0)) - before,
    )


def _load_progress(path: Path) -> dict:
    found = {}
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            found[record["key"]] = record["branch"]
    return found


def _append_progress(path: Path, key: str, branch: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "branch": branch}) + "\n")


def _forced_decision(record: dict, epoch: int, uav: int) -> dict:
    executed = record.get("forced_executed") or record.get("forced_action") or ""
    return {"epoch": epoch, "uav": uav, "requested": record.get("forced_action") or "", "executed": executed}


def _first_divergence(left: list[dict], right: list[dict], skip_epoch: int | None = None) -> int | None:
    def keyed(rows):
        return {(int(row["epoch"]), int(row["uav"])): row for row in rows}

    left_map = keyed(left)
    right_map = keyed(right)
    for epoch, uav in sorted(set(left_map) | set(right_map)):
        if skip_epoch is not None and epoch == skip_epoch:
            continue
        a = left_map.get((epoch, uav))
        b = right_map.get((epoch, uav))
        if a is None or b is None:
            return epoch
        if a.get("requested") != b.get("requested") or a.get("executed") != b.get("executed"):
            return epoch
    return None


def _materialized(q_gap: float, realized: float | None, tie: float, sign_status: str) -> str:
    if realized is None:
        return "not_run"
    if sign_status != "resolved" or abs(q_gap) <= tie:
        if abs(q_gap) <= tie and abs(realized) <= tie:
            return "materialized_no_difference"
        if abs(q_gap) <= tie:
            return "not_materialized"
        return "q_sign_unresolved"
    if abs(realized) <= tie or (realized > 0) != (q_gap > 0):
        return "not_materialized"
    return "materialized"


def _continuations(out_dir, spec, factor, scenarios, payloads, focuses, episodes, decisions) -> dict:
    tie = float(spec["integral"]["tie_abs"])
    labels = [item["attribution"] for item in focuses if item.get("status") == "COMPLETE"]
    enabled = needs_continuation(labels)
    reason = "terminal_or_information_not_separated" if enabled else "one_step_gap_is_immediate_or_unsigned"
    progress_path = out_dir / "continuation_progress.jsonl"
    done = _load_progress(progress_path)
    stats = _stats()
    stats["rollout_cap"] = CONTINUATION_ROLLOUT_CAP
    branches = []
    cap_hit = False
    if not enabled:
        return {"enabled": False, "reason": reason, "branches": [], "pairs": [], "rollouts": 0, "cap_hit": False, "reused": 0, "new": 0}
    by_snapshot = {}
    for item in focuses:
        by_snapshot.setdefault(item["snapshot"], []).append(item)
    for name, _registered in SNAPSHOTS:
        item = next((row for row in by_snapshot.get(name, []) if row.get("status") == "COMPLETE"), None)
        if item is None:
            branches.append({"snapshot": name, "arm": "C_plus", "status": "NOT_RUN", "source": "snapshot_not_complete"})
            continue
        payload = payloads[name]
        scenario = scenarios[name]
        world = payload["world"]
        beliefs = payload["beliefs"]
        epoch = int(item["epoch"])
        focus = int(item["focus_uav"])
        partner = int(item["partner_uav"])
        partner_action = (item["partner_action"][0], None if item["partner_action"][1] is None else int(item["partner_action"][1]))
        alternative = (item["alternative"][0], None if item["alternative"][1] is None else int(item["alternative"][1]))
        reused = _try_reuse(name, focus, partner_action, epoch, world, beliefs, scenario, spec, factor, episodes, decisions)
        plan = (
            ("C_plus", ("calibrate", None), False, reused.get("C_plus") if reused else None),
            ("C_minus", ("calibrate", None), True, reused.get("C_minus") if reused else None),
            ("E_plus", alternative, False, None),
            ("E_minus", alternative, True, None),
        )
        for arm, action, remove, reused_branch in plan:
            key = f"{name}|{arm}"
            if key in done:
                branches.append(done[key])
                continue
            if cap_hit:
                record = {"snapshot": name, "arm": arm, "status": "NOT_RUN", "source": "rollout_cap"}
                branches.append(record)
                continue
            if reused_branch is not None:
                _append_progress(progress_path, key, reused_branch)
                branches.append(reused_branch)
                print(f"reuse {key}", flush=True)
                continue
            if int(stats.get("rollouts", 0)) >= CONTINUATION_ROLLOUT_CAP:
                cap_hit = True
                record = {"snapshot": name, "arm": arm, "status": "NOT_RUN", "source": "rollout_cap"}
                branches.append(record)
                continue
            print(f"continue {key} rollouts={stats.get('rollouts', 0)}", flush=True)
            clear_rollout_cache()
            try:
                record = _run_new_branch(
                    name,
                    arm,
                    focus,
                    action,
                    partner,
                    partner_action,
                    remove,
                    epoch,
                    world.snapshot(),
                    [belief.copy() for belief in beliefs],
                    scenario,
                    spec,
                    factor,
                    stats,
                )
            except BudgetExceeded:
                cap_hit = True
                record = {"snapshot": name, "arm": arm, "status": "PARTIAL", "source": "rollout_cap", "rollouts": int(stats.get("rollouts", 0))}
            _append_progress(progress_path, key, record)
            branches.append(record)
    pairs = []
    for name, _registered in SNAPSHOTS:
        arms = {row["arm"]: row for row in branches if row.get("snapshot") == name and row.get("status") in ("RAN", "REUSED")}
        focus = next((row for row in by_snapshot.get(name, []) if row.get("status") == "COMPLETE"), None)
        if focus is None:
            continue
        epoch = int(focus["epoch"])
        comparisons = (
            ("C_plus_minus_E_plus", "C_plus", "E_plus", focus["gap_plus"], focus["sign_status"]),
            ("C_plus_minus_C_minus", "C_plus", "C_minus", focus["info_premium_c"], "resolved" if abs(focus["info_premium_c"]) > tie else "unresolved"),
            ("E_plus_minus_E_minus", "E_plus", "E_minus", focus["info_premium_e"], "resolved" if abs(focus["info_premium_e"]) > tie else "unresolved"),
            ("C_minus_minus_E_minus", "C_minus", "E_minus", focus["state_cost"], "resolved" if abs(focus["state_cost"]) > tie else "unresolved"),
        )
        for label, left_name, right_name, q_gap, sign_status in comparisons:
            left = arms.get(left_name)
            right = arms.get(right_name)
            realized = None
            bits = None
            energy = None
            divergence = None
            continuation_divergence = None
            if left is not None and right is not None and left.get("delta_utility") is not None and right.get("delta_utility") is not None:
                realized = float(left["delta_utility"]) - float(right["delta_utility"])
                bits = float(left["delta_bits"]) - float(right["delta_bits"])
                energy = float(left["delta_energy_j"]) - float(right["delta_energy_j"])
                left_rows = [_forced_decision(left, epoch, focus["focus_uav"])] + list(left.get("decisions") or [])
                right_rows = [_forced_decision(right, epoch, focus["focus_uav"])] + list(right.get("decisions") or [])
                divergence = _first_divergence(left_rows, right_rows)
                continuation_divergence = _first_divergence(left_rows, right_rows, skip_epoch=epoch)
            pairs.append(
                {
                    "snapshot": name,
                    "comparison": label,
                    "q_gap": q_gap,
                    "realized_utility_gap": realized,
                    "realized_bits_gap": bits,
                    "realized_energy_gap_j": energy,
                    "materialized": _materialized(float(q_gap), realized, tie, sign_status),
                    "first_divergence_epoch": divergence,
                    "first_continuation_divergence_epoch": continuation_divergence,
                }
            )
    reused = sum(1 for row in branches if row.get("status") == "REUSED")
    new = sum(1 for row in branches if row.get("status") == "RAN")
    return {
        "enabled": True,
        "reason": reason,
        "branches": branches,
        "pairs": pairs,
        "rollouts": int(stats.get("rollouts", 0)),
        "planning_steps": int(stats.get("planning_steps", 0)),
        "physical_steps": int(stats.get("physical_steps", 0)),
        "cache_hits": int(stats.get("cache_hits", 0)),
        "algebraic_integrals": int(stats.get("algebraic_integrals", 0)),
        "cap_hit": cap_hit or bool(stats.get("budget_exhausted")),
        "reused": reused,
        "new": new,
        "partial": sum(1 for row in branches if row.get("status") == "PARTIAL"),
        "not_run": sum(1 for row in branches if row.get("status") == "NOT_RUN"),
    }


def _public_branch(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "decisions"}


def run_diagnosis(out_dir: Path, tests_log: str | None, force: bool) -> dict:
    _refuse(out_dir)
    if (out_dir / "summary.json").exists() and not force:
        raise SystemExit(f"existing summary at {out_dir}; pass --force to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for name in ("decomposition.json", "continuation_progress.jsonl", "summary.json"):
            target = out_dir / name
            if target.exists():
                target.unlink()
    started = time.perf_counter()
    spec = json.loads((M13_DIR / "frozen_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((M13_DIR / "scenario_manifest.json").read_text(encoding="utf-8"))
    factor = float(manifest["budget"]["integration_factor"])
    scenarios = {row["id"]: row for row in manifest["scenarios"]}
    decisions = _read_csv(M13_DIR / "decisions.csv")
    episodes = _read_csv(M13_DIR / "episodes.csv")
    labeled = classify_requests(decisions)
    origin = _origin_summary(labeled)
    if origin["single_score"] != 67 or origin["conflict_repair"] != 9 or origin["requests"] != 76:
        raise SystemExit(f"calibration origin count changed: {origin['single_score']} {origin['conflict_repair']} {origin['requests']}")
    paired = pair_rows(episodes)
    decomp_path = out_dir / "decomposition.json"
    if decomp_path.exists():
        saved = json.loads(decomp_path.read_text(encoding="utf-8"))
        flat_rows = saved["rows"]
        focuses = saved["focuses"]
        q_stats = saved["stats"]
    else:
        clear_rollout_cache()
        clear_completion_cache()
        stats = _stats()
        stats["rollout_cap"] = Q_ROLLOUT_CAP
        flat_rows = []
        focuses = []
        payloads = {}
        for name, registered in SNAPSHOTS:
            print(f"decompose {name}", flush=True)
            _key, payload = pickle.loads((M13_DIR / "rescore_inputs" / f"{name}.pkl").read_bytes())
            payloads[name] = payload
            log_rows = [
                row
                for row in decisions
                if row["branch"] == "core" and row["method"] == ON and row["scenario_id"] == name
            ]
            try:
                result = _analyze_one(name, registered, payload, spec, factor, stats, log_rows)
            except BudgetExceeded:
                stats["budget_exhausted"] = True
                result = {"status": "budget_exhausted", "rows": [], "focuses": []}
            flat_rows.extend(result["rows"])
            focuses.extend(result["focuses"])
            if stats.get("budget_exhausted"):
                break
        q_stats = {
            "rollouts": int(stats.get("rollouts", 0)),
            "cache_hits": int(stats.get("cache_hits", 0)),
            "planning_steps": int(stats.get("planning_steps", 0)),
            "algebraic_integrals": int(stats.get("algebraic_integrals", 0)),
            "algebraic_seconds": float(stats.get("algebraic_seconds", 0.0)),
            "root_solves": int(stats.get("root_solves", 0)),
            "budget_exhausted": bool(stats.get("budget_exhausted")),
        }
        decomp_path.write_text(json.dumps({"rows": flat_rows, "focuses": focuses, "stats": q_stats, "payload_epochs": {name: int(payloads[name]["epoch"]) for name in payloads}}, indent=2), encoding="utf-8")
    payloads = {}
    for name, _registered in SNAPSHOTS:
        _key, payload = pickle.loads((M13_DIR / "rescore_inputs" / f"{name}.pkl").read_bytes())
        payloads[name] = payload
    print("continuations", flush=True)
    continuation = _continuations(out_dir, spec, factor, scenarios, payloads, focuses, episodes, decisions)
    elapsed = time.perf_counter() - started
    counts = {
        "single_score": origin["single_score"],
        "conflict_repair": origin["conflict_repair"],
        "explained_by_immediate_motion": sum(1 for item in focuses if item.get("attribution") == "immediate_motion_cost"),
        "explained_by_state_cost": sum(1 for item in focuses if item.get("attribution") == "state_cost_without_update"),
        "information_premium": sum(1 for item in focuses if item.get("attribution") == "information_premium"),
        "mixed": sum(1 for item in focuses if item.get("attribution") == "mixed"),
        "sign_unresolved": sum(1 for item in focuses if item.get("attribution") == "sign_unresolved"),
        "focuses": len(focuses),
        "q_rollouts": q_stats["rollouts"],
        "q_cache_hits": q_stats["cache_hits"],
        "q_planning_steps": q_stats["planning_steps"],
        "q_algebraic_integrals": q_stats["algebraic_integrals"],
        "q_algebraic_seconds": q_stats["algebraic_seconds"],
        "continuation_enabled": continuation["enabled"],
        "continuation_rollouts": continuation.get("rollouts", 0),
        "continuation_reused": continuation.get("reused", 0),
        "continuation_new": continuation.get("new", 0),
        "continuation_not_run": continuation.get("not_run", 0),
    }
    inputs = {
        "frozen_config_sha256": _sha256(M13_DIR / "frozen_config.json"),
        "scenario_manifest_sha256": _sha256(M13_DIR / "scenario_manifest.json"),
        "decisions_sha256": _sha256(M13_DIR / "decisions.csv"),
        "episodes_sha256": _sha256(M13_DIR / "episodes.csv"),
        "snapshots": {name: _sha256(M13_DIR / "rescore_inputs" / f"{name}.pkl") for name, _registered in SNAPSHOTS},
    }
    summary = {
        "run_id": out_dir.name,
        "result_dir": "results/active_model_identification_m14/" + out_dir.name,
        "instruction_commit": _git()["head"],
        "reviewed_result_commit": REVIEWED_RESULT,
        "command": "py -3 -m wt.active_model_identification.run_m14 --tests-log results/active_model_identification_m14/m14_20261114/necessary_tests.log",
        "unit_tests_status": _tests_status(tests_log),
        "origin": origin,
        "focuses": focuses,
        "continuation": {key: value for key, value in continuation.items() if key != "branches"},
        "continuation_branches": [_public_branch(row) for row in continuation.get("branches", [])],
        "counts": counts,
        "q_stats": q_stats,
        "inputs": inputs,
        "snapshot_list": [{"snapshot": name, "registered_class": registered} for name, registered in SNAPSHOTS],
        "elapsed_s": elapsed,
        "training_updates": 0,
        "duration_rule": duration_of("calibrate:None")[1],
        "core_terminals": _core_terminals(episodes),
    }
    summary["continuation"]["pairs"] = continuation.get("pairs", [])
    _write_csv(out_dir / "pair_rows.csv", paired)
    _write_csv(out_dir / "calibration_origin_rows.csv", labeled)
    _write_csv(out_dir / "q_decomposition_rows.csv", flat_rows)
    if continuation["enabled"]:
        _write_csv(out_dir / "continuation_rows.csv", [_public_branch(row) for row in continuation["branches"]])
        _write_csv(out_dir / "continuation_pairs.csv", continuation["pairs"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "result_dir": summary["result_dir"],
                "reviewed_result_commit": REVIEWED_RESULT,
                "instruction_commit": summary["instruction_commit"],
                "snapshot_list": summary["snapshot_list"],
                "inputs": inputs,
                "counts": counts,
                "continuation_enabled": continuation["enabled"],
                "continuation_reason": continuation["reason"],
                "training_updates": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_m14_report(summary, ROOT / "docs" / "research" / "Active_Model_Identification_M14_Report_CN.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/active_model_identification_m14/m14_20261114")
    parser.add_argument("--tests-log", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_diagnosis(ROOT / args.out, args.tests_log or None, args.force)
    print(
        json.dumps(
            {
                "tests": summary["unit_tests_status"],
                "requests": summary["origin"]["requests"],
                "repair": summary["origin"]["conflict_repair"],
                "single": summary["origin"]["single_score"],
                "focuses": summary["counts"],
                "continuation": summary["continuation"]["enabled"],
                "q_rollouts": summary["q_stats"]["rollouts"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
