"""Run the M1.5 return-tail candidate.

Legacy scoring stays the default. The new mode is opt-in for this trial only.
M1.3 core missions are the baseline and are not rerun.
"""

from __future__ import annotations

import copy
import csv
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

from wt.active_model_identification.attribution import SNAPSHOTS, split_action
from wt.active_model_identification.belief import prior_belief, quadrature
from wt.active_model_identification.completion import (
    TAIL_MODE,
    clear_completion_cache,
    clear_return_tail_cache,
    reset_tail_stats,
    tail_stats,
)
from wt.active_model_identification.policies import pick_action, select_lookahead, tie_abs
from wt.active_model_identification.run_m12 import _sha256, _write_csv
from wt.active_model_identification.run_m13 import METHODS, _append_progress, _load_progress, _play, _stats
from wt.active_model_identification.segment import (
    BudgetExceeded,
    clear_rollout_cache,
    describe,
    negative_legal_business,
    posterior_quantile,
    roll_action,
    value_at,
)
from wt.active_model_identification.spec import ROOT
from wt.active_model_identification.world import make_world

M13_DIR = ROOT / "results" / "active_model_identification_m13" / "m13_20261114"
M14_DIR = ROOT / "results" / "active_model_identification_m14" / "m14_20261114"
CONFIG_PATH = ROOT / "configs" / "experiments" / "active_model_identification_m15.json"
UTILITY_TOL = 1e-9
PHYSICAL_KEYS = ("not_returned", "not_landed", "energy_exhausted", "return_failures")
CHECKS = [
    {
        "change": "新增 return_tail_when_no_business。仅当旧终端没有正收益业务时，用同一执行器的返航、落地和剩余地面负载替换零值。",
        "impact": "缺省 legacy 的报价、排序和已保存 Q 可能被误改。",
        "check": "空中与落地状态对确定性 RETURN 回放；六个快照的 legacy Q、排序和修复结果对 M1.4 保存值。",
    },
    {
        "change": "尾效用保留负号，并在电量截断处分段，进入 terminal_value 与分段积分。",
        "impact": "none 分支的正式评分可能仍停在 0，或把截断点积错。",
        "check": "单测比较分段积分与独立参考；三条 single_score 快照列出 R30、旧终端、新尾、新 Q 和分量，确认尾项进入 Q。",
    },
    {
        "change": "正收益业务分支不加第二份返航费，冲突修复代码不改。",
        "impact": "冲突快照的原终端或修复结果可能漂移。",
        "check": "旧终端为正的节点上新旧值一致；legacy 修复动作与保存的联合动作一致。",
    },
    {
        "change": "24 条新航次显式打开尾模式，基线只读 M1.3 核心 24 条。",
        "impact": "性能可能回退，或被均值掩盖。",
        "check": "四组事前比较逐场景报告 bit 与 J；回退则判定未满足保留标准，不改默认模式。",
    },
]


def _refuse(out_dir: Path) -> None:
    if "active_model_identification_m15" not in out_dir.parts:
        raise SystemExit("refusing to write outside results/active_model_identification_m15")
    banned = {
        "active_model_identification_m1",
        "active_model_identification_m11",
        "active_model_identification_m12",
        "active_model_identification_m13",
        "active_model_identification_m14",
    }
    if set(out_dir.parts) & banned:
        raise SystemExit("refusing to write into an earlier result directory")


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _label(action: tuple) -> str:
    return f"{action[0]}:{action[1]}"


def _parse_action(label: str) -> tuple:
    kind, target = str(label).split(":")
    return kind, None if target == "None" else int(target)


def _git(paths: list[str] | None = None) -> dict:
    command = ["git", "-c", "core.longpaths=true"]
    try:
        head = subprocess.check_output(command + ["rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(command + ["status", "--porcelain=v1", "-b"], cwd=ROOT, text=True).strip()
        diff = subprocess.check_output(command + ["diff", "--stat"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError as exc:
        return {"head": "unknown", "error": str(exc)}
    found = {}
    for path in paths or []:
        try:
            found[path] = subprocess.check_output(
                command + ["log", "-1", "--format=%H", "--", path], cwd=ROOT, text=True
            ).strip()
        except subprocess.CalledProcessError:
            found[path] = ""
    return {"head": head, "status": status, "diff_stat": diff, "path_commits": found}


def _with_mode(spec: dict, mode: str) -> dict:
    cloned = copy.deepcopy(spec)
    cloned["terminal"] = {"mode": mode}
    return cloned


def _public_stats(stats: dict) -> dict:
    keys = (
        "rollouts",
        "cache_hits",
        "planning_steps",
        "planning_copies",
        "physical_steps",
        "score_calls",
        "segment_score_calls",
        "algebraic_integrals",
        "algebraic_seconds",
        "root_solves",
        "tail_piece_evals",
        "budget_exhausted",
    )
    return {key: stats.get(key, 0) for key in keys}


def _tests_ok(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\nOK" in text or text.rstrip().endswith("OK")


def _run_tests(log_path: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_active_model_identification_m13",
        "tests.test_active_model_identification_m14",
        "tests.test_active_model_identification_m15",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return int(completed.returncode)


def _saved_snapshot(rows: list[dict], name: str) -> dict:
    subset = [row for row in rows if row["snapshot"] == name]
    focus_ids = {row["focus_uav"] for row in subset if row["role"] == "C" and row["focus_uav"] != ""}
    if len(focus_ids) != 1:
        raise RuntimeError(f"{name} focus set is {sorted(focus_ids)}")
    focus = int(next(iter(focus_ids)))
    scores = {}
    for row in subset:
        if row["role"] == "table" and row["table_q"] != "":
            scores[(focus, row["action"])] = float(row["table_q"])
        elif row["role"] == "partner_table" and row["table_q"] != "":
            scores[(1 - focus, row["action"])] = float(row["table_q"])
    meta = next(row for row in subset if row["role"] == "C" and row["focus_uav"] != "")
    alternative = next(row["action"] for row in subset if row["role"] == "E" and row["action"])
    return {
        "focus": focus,
        "scores": scores,
        "table_choice": (meta["table_choice_0"], meta["table_choice_1"]),
        "joint": (meta["joint_0"], meta["joint_1"]),
        "stored": (meta["stored_0"], meta["stored_1"]),
        "alternative": alternative,
        "epoch": int(meta["epoch"]),
    }


def _table_map(tables: list) -> dict:
    found = {}
    for uav, table in enumerate(tables):
        for score, action in table:
            found[(uav, _label(action))] = float(score)
    return found


def _choice_labels(tables: list, spec: dict) -> tuple[str, str]:
    tau = tie_abs(spec)
    return tuple(_label(pick_action(table, tau)[1]) for table in tables)  # type: ignore[return-value]


def _positive_mismatches(tables: list, world, beliefs, tail_spec: dict, legacy_spec: dict, factor: float, stats: dict, tolerance: float) -> list[dict]:
    guard = float(tail_spec["belief"]["guard_probability"])
    nodes = int(tail_spec["belief"]["quadrature_nodes"])
    failures = []
    for uav, table in enumerate(tables):
        grid = quadrature(beliefs[uav], nodes)
        for _score, action in table:
            for theta, _weight in grid:
                roll = roll_action(world, beliefs, action, uav, float(theta), tail_spec, factor, stats)
                posterior = posterior_quantile(roll.mu, roll.lam, roll.lo, roll.hi, roll.z_j, roll.y0, roll.sigma, guard)
                business = tuple(plan for plan in roll.plans if plan.kind != "return_tail" and plan.label != "return_tail")
                old = value_at(posterior, roll.energy_j, business, legacy_spec, roll.failed)
                new = value_at(posterior, roll.energy_j, roll.plans, tail_spec, roll.failed)
                if old > 1e-8 and abs(old - new) > tolerance:
                    failures.append(
                        {
                            "uav": uav,
                            "action": _label(action),
                            "theta": float(theta),
                            "legacy_terminal": old,
                            "new_terminal": new,
                        }
                    )
    return failures


def _component_row(world, beliefs, legacy_spec, tail_spec, factor, stats, uav: int, action: tuple) -> dict:
    legacy = split_action(world, beliefs, legacy_spec, factor, uav, action, stats)
    updated = split_action(world, beliefs, tail_spec, factor, uav, action, stats)
    theta = float(updated["representative_theta"])
    roll = roll_action(world, beliefs, action, uav, theta, tail_spec, factor, stats)
    tail = next(plan for plan in roll.plans if plan.label == "return_tail")
    posterior = float(updated["detail"]["posterior_q95_eps0"])
    plan_id, _intercept, _slope, value = describe(posterior, roll.energy_j, roll.plans, tail_spec, roll.failed)
    q_gap = float(updated["q_plus"]) - float(legacy["q_plus"])
    terminal_gap = float(updated["terminal_plus"]) - float(legacy["terminal_plus"])
    return {
        "uav": uav,
        "action": _label(action),
        "r30": updated["r30"],
        "legacy_r30": legacy["r30"],
        "legacy_terminal": legacy["terminal_plus"],
        "new_terminal": updated["terminal_plus"],
        "legacy_q": legacy["q_plus"],
        "new_q": updated["q_plus"],
        "q_gap_minus_terminal_gap": q_gap - terminal_gap,
        "posterior_q95": posterior,
        "landing_s": "" if tail.landing_s < 0.0 else tail.landing_s,
        "landed": bool(tail.landed),
        "energy_truncated_at_posterior": plan_id == -3,
        "terminal_source": "return_tail" if plan_id in (-2, -3) else updated["detail"]["active_posterior"],
        "flight_z_j": tail.flight_z_j,
        "flight_aux_j": tail.flight_aux_j,
        "ground_aux_j": tail.ground_aux_j,
        "radio_j": tail.radio_j,
        "propulsion_at_posterior_j": posterior * tail.flight_z_j,
        "demand_at_posterior_j": posterior * tail.z_j + tail.aux_j,
        "tail_value_at_posterior": value,
        "active_posterior": updated["detail"]["active_posterior"],
    }


def _rank_changes(legacy_tables: list, new_tables: list) -> list[dict]:
    changes = []
    for uav, (legacy, updated) in enumerate(zip(legacy_tables, new_tables)):
        old = [_label(action) for _score, action in legacy]
        new = [_label(action) for _score, action in updated]
        for label in old:
            if label in new and old.index(label) != new.index(label):
                changes.append(
                    {
                        "uav": uav,
                        "action": label,
                        "legacy_rank": old.index(label),
                        "new_rank": new.index(label),
                    }
                )
    return changes


def _snapshots(out_dir: Path, legacy_spec: dict, tail_spec: dict, factor: float, cfg: dict) -> dict:
    cap = int(cfg["budget"]["snapshot_rollouts"])
    q_tol = float(cfg["tolerances"]["legacy_q_abs"])
    branch_tol = float(cfg["tolerances"]["positive_branch_abs"])
    enter_tol = float(cfg["tolerances"]["tail_enters_abs"])
    stats = _stats()
    stats["rollout_cap"] = cap
    stats["tail_piece_evals"] = 0
    rows = _read_csv(M14_DIR / "q_decomposition_rows.csv")
    errors = []
    snapshot_rows = []
    components = []
    rank_rows = []
    print("snapshots", flush=True)
    for name, registered in SNAPSHOTS:
        if stats.get("budget_exhausted"):
            break
        print(f"snapshot {name}", flush=True)
        _key, payload = pickle.loads((M13_DIR / "rescore_inputs" / f"{name}.pkl").read_bytes())
        world = payload["world"]
        beliefs = payload["beliefs"]
        saved = _saved_snapshot(rows, name)
        try:
            legacy_joint, legacy_tables = select_lookahead(
                "lookahead_calibration_on", world, beliefs, legacy_spec, factor, stats
            )
            new_joint, new_tables = select_lookahead(
                "lookahead_calibration_on", world, beliefs, tail_spec, factor, stats
            )
        except BudgetExceeded:
            stats["budget_exhausted"] = True
            errors.append(f"{name}: snapshot rollout cap")
            break
        got = _table_map(legacy_tables)
        if set(got) != set(saved["scores"]):
            errors.append(f"{name}: legacy action set changed")
        worst = 0.0
        for key, score in got.items():
            if key not in saved["scores"]:
                continue
            worst = max(worst, abs(score - saved["scores"][key]))
        if worst > q_tol:
            errors.append(f"{name}: legacy Q abs {worst}")
        picks = _choice_labels(legacy_tables, legacy_spec)
        joint = tuple(_label(action) for action in legacy_joint)
        if picks != saved["table_choice"]:
            errors.append(f"{name}: legacy pick {picks} != {saved['table_choice']}")
        if joint != saved["joint"] or joint != saved["stored"]:
            errors.append(f"{name}: legacy joint {joint} != stored {saved['stored']}")
        mismatches = _positive_mismatches(new_tables, world, beliefs, tail_spec, legacy_spec, factor, stats, branch_tol)
        if mismatches:
            errors.append(f"{name}: {len(mismatches)} positive-branch terminal changes")
        changes = _rank_changes(legacy_tables, new_tables)
        for change in changes:
            change["snapshot"] = name
            change["registered_class"] = registered
            rank_rows.append(change)
        new_picks = _choice_labels(new_tables, tail_spec)
        new_joint = tuple(_label(action) for action in new_joint)
        snapshot_rows.append(
            {
                "snapshot": name,
                "registered_class": registered,
                "epoch": saved["epoch"],
                "focus_uav": saved["focus"],
                "legacy_q_abs": worst,
                "legacy_pick": "|".join(picks),
                "legacy_joint": "|".join(joint),
                "new_pick": "|".join(new_picks),
                "new_joint": "|".join(new_joint),
                "positive_branch_mismatches": len(mismatches),
                "rank_changes": len(changes),
                "repair_still_requests_calibrate": int(
                    any(label == "calibrate:None" and pick != "calibrate:None" for pick, label in zip(new_picks, new_joint))
                ),
            }
        )
        if registered == "single_score":
            focus = int(saved["focus"])
            for label in ("calibrate:None", saved["alternative"]):
                row = _component_row(
                    world, beliefs, legacy_spec, tail_spec, factor, stats, focus, _parse_action(label)
                )
                row["snapshot"] = name
                components.append(row)
                if abs(float(row["q_gap_minus_terminal_gap"])) > enter_tol or float(row["new_terminal"]) >= -enter_tol:
                    errors.append(f"{name} {label}: tail did not enter the score")
                if abs(float(row["legacy_terminal"])) > enter_tol:
                    errors.append(f"{name} {label}: legacy terminal was not the zero branch")
    fine_rows = []
    fine_spec = copy.deepcopy(tail_spec)
    fine_spec["belief"]["quadrature_nodes"] = int(cfg["scoring"]["fine_theta_nodes"])
    fine_spec["integral"]["segment_inner_nodes"] = int(cfg["scoring"]["fine_inner_nodes"])
    for name, registered in SNAPSHOTS:
        if registered != "single_score" or stats.get("budget_exhausted"):
            continue
        saved = _saved_snapshot(rows, name)
        _key, payload = pickle.loads((M13_DIR / "rescore_inputs" / f"{name}.pkl").read_bytes())
        focus = int(saved["focus"])
        pair = (("calibrate", None), _parse_action(saved["alternative"]))
        fine = {}
        coarse = {}
        unresolved_reason = ""
        for action in pair:
            if int(stats.get("rollouts", 0)) + int(cfg["scoring"]["fine_theta_nodes"]) > cap:
                stats["budget_exhausted"] = True
                unresolved_reason = "snapshot_rollout_cap"
                break
            try:
                coarse[action] = split_action(payload["world"], payload["beliefs"], tail_spec, factor, focus, action, stats)
                fine[action] = split_action(payload["world"], payload["beliefs"], fine_spec, factor, focus, action, stats)
            except BudgetExceeded:
                stats["budget_exhausted"] = True
                unresolved_reason = "snapshot_rollout_cap"
                break
        if unresolved_reason or len(fine) != 2:
            fine_rows.append(
                {
                    "snapshot": name,
                    "status": "unresolved",
                    "reason": unresolved_reason or "incomplete",
                    "tie_rule": "unchanged",
                }
            )
            continue
        left, right = pair
        gap15 = float(coarse[left]["q_plus"]) - float(coarse[right]["q_plus"])
        gap31 = float(fine[left]["q_plus"]) - float(fine[right]["q_plus"])
        bound = abs(float(coarse[left]["q_plus"]) - float(fine[left]["q_plus"])) + abs(
            float(coarse[right]["q_plus"]) - float(fine[right]["q_plus"])
        )
        fine_rows.append(
            {
                "snapshot": name,
                "focus_uav": focus,
                "left": _label(left),
                "right": _label(right),
                "gap15": gap15,
                "gap31": gap31,
                "combined_abs_error": bound,
                "resolved": bool(abs(gap15) > bound),
                "status": "resolved" if abs(gap15) > bound else "unresolved",
                "reason": "" if abs(gap15) > bound else "combined_error_bound_covers_gap",
                "tie_rule": "unchanged",
                "q15_left": coarse[left]["q_plus"],
                "q31_left": fine[left]["q_plus"],
                "q15_right": coarse[right]["q_plus"],
                "q31_right": fine[right]["q_plus"],
            }
        )
    _write_csv(out_dir / "snapshot_scores.csv", snapshot_rows)
    _write_csv(out_dir / "tail_components.csv", components)
    _write_csv(out_dir / "snapshot_rank_changes.csv", rank_rows)
    _write_csv(out_dir / "tail_gap_check.csv", fine_rows)
    result = {
        "passed": not errors and {row["snapshot"] for row in snapshot_rows} == {name for name, _registered in SNAPSHOTS},
        "errors": errors,
        "snapshots": snapshot_rows,
        "components": components,
        "rank_changes": rank_rows,
        "fine": fine_rows,
        "stats": _public_stats(stats),
        "tail": tail_stats(),
    }
    (out_dir / "snapshot_gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _probe(world, beliefs, action, uav: int, spec: dict, factor: float, stats: dict) -> dict:
    guard = float(spec["belief"]["guard_probability"])
    grid = quadrature(beliefs[uav], int(spec["belief"]["quadrature_nodes"]))
    tail_weight = 0.0
    best_negative = None
    best_label = ""
    mean = beliefs[uav].mean()
    closest = None
    closest_theta = None
    for theta, weight in grid:
        roll = roll_action(world, beliefs, action, uav, float(theta), spec, factor, stats)
        posterior = posterior_quantile(roll.mu, roll.lam, roll.lo, roll.hi, roll.z_j, roll.y0, roll.sigma, guard)
        plan_id = describe(posterior, roll.energy_j, roll.plans, spec, roll.failed)[0]
        if plan_id in (-2, -3):
            tail_weight += float(weight)
        info = negative_legal_business(posterior, roll.energy_j, roll.plans, spec)
        if info["exists"] and (best_negative is None or float(info["max_net"]) > best_negative):
            best_negative = float(info["max_net"])
            best_label = str(info["label"])
        if closest_theta is None or abs(float(theta) - mean) < abs(float(closest_theta) - mean):
            closest = roll
            closest_theta = float(theta)
    _ = closest
    return {
        "tail_triggered": int(tail_weight > 0.0),
        "tail_weight": tail_weight,
        "negative_legal_exists": int(best_negative is not None),
        "negative_legal_max": "" if best_negative is None else best_negative,
        "negative_legal_label": best_label,
        "terminal_mode": TAIL_MODE,
    }


def _enrich(episode: dict, decisions: list[dict], legacy_actions: dict) -> None:
    episode["terminal_mode"] = TAIL_MODE
    episode["tail_trigger_decisions"] = sum(int(row["tail_triggered"]) for row in decisions)
    episode["tail_trigger_weight"] = sum(float(row["tail_weight"]) for row in decisions)
    episode["negative_legal_decisions"] = sum(int(row["negative_legal_exists"]) for row in decisions)
    negatives = [float(row["negative_legal_max"]) for row in decisions if row["negative_legal_max"] != ""]
    episode["negative_legal_max"] = "" if not negatives else max(negatives)
    episode["calibrate_requested"] = sum(row["requested"] == "calibrate:None" for row in decisions)
    episode["calibrate_executed"] = sum(row["executed"] == "calibrate:None" for row in decisions)
    episode["repair_calibrate_requests"] = sum(
        row["requested"] == "calibrate:None" and row["table_choice"] != "calibrate:None" for row in decisions
    )
    first = {0: None, 1: None}
    for row in decisions:
        uav = int(row["uav"])
        if first[uav] is None and str(row["executed"]).startswith("service:"):
            first[uav] = {"target": str(row["executed"]).split(":", 1)[1], "t_s": int(row["epoch"]) * 30}
    for uav in (0, 1):
        episode[f"first_service_{uav}_s"] = "" if first[uav] is None else first[uav]["t_s"]
        episode[f"first_service_{uav}_target"] = "" if first[uav] is None else first[uav]["target"]
    epochs = sorted({int(row["epoch"]) for row in decisions})
    first_epoch = None
    differed = 0
    for epoch in epochs:
        current = {int(row["uav"]): row["requested"] for row in decisions if int(row["epoch"]) == epoch}
        previous = legacy_actions.get((episode["scenario_id"], episode["method"], epoch))
        if previous is None:
            continue
        if current.get(0) != previous.get(0) or current.get(1) != previous.get(1):
            differed += 1
            if first_epoch is None:
                first_epoch = epoch
                episode["first_divergence_legacy"] = f"{previous.get(0)}|{previous.get(1)}"
                episode["first_divergence_new"] = f"{current.get(0)}|{current.get(1)}"
    episode["first_divergence_epoch"] = "" if first_epoch is None else first_epoch
    episode["diverged_epochs"] = differed
    if first_epoch is None:
        episode["first_divergence_legacy"] = ""
        episode["first_divergence_new"] = ""


def _legacy_actions(rows: list[dict]) -> dict:
    found: dict[tuple, dict] = {}
    for row in rows:
        if row.get("branch") != "core":
            continue
        key = (row["scenario_id"], row["method"], int(row["epoch"]))
        found.setdefault(key, {})[int(row["uav"])] = row["requested"]
    return found


def _missions(out_dir: Path, spec: dict, manifest: dict, cfg: dict, factor: float, legacy_decisions: list[dict]) -> dict:
    cap = int(cfg["budget"]["mission_rollouts"])
    progress = out_dir / "completed_missions.jsonl"
    stats_path = out_dir / "mission_stats.json"
    done, episodes, decisions, deliveries = _load_progress(progress)
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else _stats()
    stats["rollout_cap"] = cap
    stats.setdefault("tail_piece_evals", 0)
    by_id = {row["id"]: row for row in manifest["scenarios"]}
    previous = _legacy_actions(legacy_decisions)
    reason = ""
    for scenario_id in cfg["scenarios"]:
        scenario = by_id[scenario_id]
        for method in METHODS:
            key = f"core|{scenario_id}|{method}"
            if key in done:
                print(f"resume skip {key}", flush=True)
                continue
            if int(stats.get("rollouts", 0)) >= cap or stats.get("budget_exhausted"):
                reason = "mission_rollout_cap"
                stats["budget_exhausted"] = True
                break
            print(f"run {scenario_id} {method} rollouts={stats.get('rollouts', 0)}", flush=True)
            world = make_world(spec, scenario["battery_j"], f"{scenario_id}-{method}")
            beliefs = [prior_belief(spec), prior_belief(spec)]
            probes: list[dict] = []

            def capture(epoch, captured_world, captured_beliefs, requested, _spec=spec, _factor=factor, _stats=stats, _probes=probes):
                _ = epoch
                for uav, action in enumerate(requested):
                    _probes.append(_probe(captured_world, captured_beliefs, action, uav, _spec, _factor, _stats))

            try:
                episode, decision_rows, delivery_rows, _snapshot = _play(
                    method, scenario, spec, factor, stats, "core", 0, world, beliefs, [], capture
                )
            except BudgetExceeded:
                stats["budget_exhausted"] = True
                reason = f"mission_rollout_cap_during_{scenario_id}_{method}"
                break
            if len(probes) != len(decision_rows):
                raise RuntimeError(f"{key} probe rows {len(probes)} != decisions {len(decision_rows)}")
            for row, probe in zip(decision_rows, probes):
                row.update(probe)
            _enrich(episode, decision_rows, previous)
            episodes.append(episode)
            decisions.extend(decision_rows)
            deliveries.extend(delivery_rows)
            done.add(key)
            _append_progress(progress, key, episode, decision_rows, delivery_rows)
            stats_path.write_text(json.dumps(stats), encoding="utf-8")
        if reason:
            break
    expected = {f"core|{scenario_id}|{method}" for scenario_id in cfg["scenarios"] for method in METHODS}
    return {
        "episodes": episodes,
        "decisions": decisions,
        "deliveries": deliveries,
        "stats": _public_stats(stats),
        "complete": expected <= done and not reason,
        "reason": reason,
        "completed_keys": sorted(done),
    }


def _f(row: dict, key: str) -> float:
    return float(row[key])


def _i(row: dict, key: str) -> int:
    return int(round(float(row[key])))


def _pair(name: str, new_row: dict, old_row: dict) -> dict:
    bits_delta = _i(new_row, "on_time_bits") - _i(old_row, "on_time_bits")
    utility_delta = _f(new_row, "utility") - _f(old_row, "utility")
    physical = [key for key in PHYSICAL_KEYS if _i(new_row, key) > _i(old_row, key)]
    reserve = _i(new_row, "below_reserve") > _i(old_row, "below_reserve")
    return {
        "comparison": name,
        "scenario_id": new_row["scenario_id"],
        "base": new_row["base"],
        "condition": new_row["condition"],
        "bits_new": _i(new_row, "on_time_bits"),
        "bits_old": _i(old_row, "on_time_bits"),
        "bits_delta": bits_delta,
        "utility_new": _f(new_row, "utility"),
        "utility_old": _f(old_row, "utility"),
        "utility_delta": utility_delta,
        "energy_new": _f(new_row, "energy_j"),
        "energy_old": _f(old_row, "energy_j"),
        "energy_delta": _f(new_row, "energy_j") - _f(old_row, "energy_j"),
        "bits_regressed": bits_delta < 0,
        "utility_regressed": utility_delta < -UTILITY_TOL,
        "regressed": bits_delta < 0 or utility_delta < -UTILITY_TOL,
        "utility_improved": utility_delta > UTILITY_TOL,
        "physical_worse": physical,
        "reserve_worse": reserve,
        "not_returned_new": _i(new_row, "not_returned"),
        "not_returned_old": _i(old_row, "not_returned"),
        "not_landed_new": _i(new_row, "not_landed"),
        "not_landed_old": _i(old_row, "not_landed"),
        "energy_exhausted_new": _i(new_row, "energy_exhausted"),
        "energy_exhausted_old": _i(old_row, "energy_exhausted"),
        "return_failures_new": _i(new_row, "return_failures"),
        "return_failures_old": _i(old_row, "return_failures"),
        "below_reserve_new": _i(new_row, "below_reserve"),
        "below_reserve_old": _i(old_row, "below_reserve"),
    }


def _comparisons(episodes: list[dict], legacy: list[dict], cfg: dict) -> list[dict]:
    fresh = {(row["scenario_id"], row["method"]): row for row in episodes}
    base = {(row["scenario_id"], row["method"]): row for row in legacy if row.get("branch") == "core"}
    on = cfg["methods"]["corrected_ON"]
    off = cfg["methods"]["corrected_OFF"]
    rows = []
    for scenario_id in cfg["scenarios"]:
        corrected_on = fresh[(scenario_id, on)]
        corrected_off = fresh[(scenario_id, off)]
        legacy_on = base[(scenario_id, on)]
        legacy_off = base[(scenario_id, off)]
        rows.append(_pair("corrected_ON-legacy_ON", corrected_on, legacy_on))
        rows.append(_pair("corrected_OFF-legacy_OFF", corrected_off, legacy_off))
        rows.append(_pair("corrected_ON-corrected_OFF", corrected_on, corrected_off))
        rows.append(_pair("corrected_ON-legacy_OFF", corrected_on, legacy_off))
    return rows


def _tier_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list] = {}
    for row in rows:
        grouped.setdefault((row["comparison"], row["condition"]), []).append(row)
    summary = []
    for (comparison, condition), group in grouped.items():
        summary.append(
            {
                "comparison": comparison,
                "condition": condition,
                "n": len(group),
                "utility_delta_min": min(row["utility_delta"] for row in group),
                "utility_delta_sum": sum(row["utility_delta"] for row in group),
                "bits_delta_min": min(row["bits_delta"] for row in group),
                "regressions": sum(bool(row["regressed"]) for row in group),
                "physical_regressions": sum(bool(row["physical_worse"]) for row in group),
                "reserve_regressions": sum(bool(row["reserve_worse"]) for row in group),
                "utility_improvements": sum(bool(row["utility_improved"]) for row in group),
            }
        )
    return summary


def _judge(rows: list[dict], complete: bool) -> dict:
    same_switch = [row for row in rows if row["comparison"] in ("corrected_ON-legacy_ON", "corrected_OFF-legacy_OFF")]
    on_pairs = [row for row in same_switch if row["comparison"] == "corrected_ON-legacy_ON"]
    off_pairs = [row for row in same_switch if row["comparison"] == "corrected_OFF-legacy_OFF"]
    no_regression = bool(same_switch) and all(not row["regressed"] for row in same_switch)
    improved = any(row["utility_improved"] for row in same_switch)
    physical_ok = all(not row["physical_worse"] for row in same_switch)
    reserve_ok = all(not row["reserve_worse"] for row in same_switch)
    standard = complete and len(on_pairs) == 12 and len(off_pairs) == 12 and no_regression and improved and physical_ok and reserve_ok
    versus_legacy_off = [row for row in rows if row["comparison"] == "corrected_ON-legacy_OFF"]
    versus_off = [row for row in rows if row["comparison"] == "corrected_ON-corrected_OFF"]
    loses_legacy_off = any(row["regressed"] for row in versus_legacy_off)
    loses_off = any(row["regressed"] for row in versus_off)
    primary = standard and not loses_legacy_off and not loses_off
    if not standard:
        conclusion = "本轮未满足性能保留标准"
    elif not primary:
        conclusion = "同开关比较满足性能保留标准，但 corrected_ON 不获得主策略性能保留资格"
    else:
        conclusion = "同开关比较满足性能保留标准，且 corrected_ON 相对 legacy_OFF 与 corrected_OFF 没有回退"
    return {
        "standard_met": standard,
        "primary_policy_retained": primary,
        "conclusion": conclusion,
        "no_same_switch_regression": no_regression,
        "utility_improved": improved,
        "physical_ok": physical_ok,
        "reserve_ok": reserve_ok,
        "corrected_on_regresses_vs_legacy_off": loses_legacy_off,
        "corrected_on_regresses_vs_corrected_off": loses_off,
        "complete": complete,
        "default_mode": "legacy",
    }


def _fmt(value, digits: int = 9) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_report(summary: dict, path: Path) -> None:
    judgment = summary.get("judgment") or {}
    conclusion = judgment.get("conclusion", "本轮未形成性能判定")
    lines = [
        "# M1.5：无正收益业务分支的返航尾成本修正",
        "",
        f"性能判定：{conclusion}。",
        "",
        "默认终端模式仍是 legacy。本轮候选 `return_tail_when_no_business` 只在显式打开时生效，不替换旧默认，也不把校准次数或效用变化解释成主动信息收益。",
        "",
    ]
    if not summary.get("missions_ran"):
        lines.append(f"主试未运行。原因：{summary.get('stop_reason', '')}。")
        if summary.get("gate_errors"):
            lines.append("")
            lines.append("快照门禁：")
            for error in summary["gate_errors"]:
                lines.append(f"- {error}")
        lines.append("")
    else:
        lines.extend(
            [
                f"同开关比较没有交付量或净效用回退：{judgment.get('no_same_switch_regression')}。至少一处净效用改善：{judgment.get('utility_improved')}。物理完成未变差：{judgment.get('physical_ok')}。备用电量计数未变差：{judgment.get('reserve_ok')}。",
                f"corrected_ON 相对 legacy_OFF 是否回退：{judgment.get('corrected_on_regresses_vs_legacy_off')}。相对 corrected_OFF 是否回退：{judgment.get('corrected_on_regresses_vs_corrected_off')}。主策略保留资格：{judgment.get('primary_policy_retained')}。",
                "超过旧 ON 不能单独保留为主策略。corrected_ON 若低于 legacy_OFF 或 corrected_OFF，就不描述成优于当前最好执行。校准次数或效用变化不证明主动信息价值。",
                "below_reserve 单列，不称为物理返航失败。J 的比较容差是绝对差 1e-9；按时比特按整数比较，不允许减少。评分 tie 容差不作为性能回退容差。",
                "",
                "## 逐场景差值",
                "",
            ]
        )
        headers = ["场景", "比较", "bit差", "J差", "能耗差 J", "回退", "物理变差", "备用电量变差"]
        body = []
        for row in summary.get("comparisons", []):
            body.append(
                [
                    row["scenario_id"],
                    row["comparison"],
                    row["bits_delta"],
                    _fmt(row["utility_delta"]),
                    _fmt(row["energy_delta"], 3),
                    int(bool(row["regressed"])),
                    ",".join(row["physical_worse"]) if row["physical_worse"] else "",
                    int(bool(row["reserve_worse"])),
                ]
            )
        lines.append(_markdown_table(headers, body))
        lines.extend(["", "## 分档最差值", "", "下面先给每档最差差值，再给合计。最差值为负就是回退，不拿合计掩盖。", ""])
        tier_headers = ["比较", "档", "条数", "J最小差", "J合计", "bit最小差", "回退条数", "物理变差", "备用电量变差"]
        tier_body = []
        for row in summary.get("tiers", []):
            tier_body.append(
                [
                    row["comparison"],
                    row["condition"],
                    row["n"],
                    _fmt(row["utility_delta_min"]),
                    _fmt(row["utility_delta_sum"]),
                    row["bits_delta_min"],
                    row["regressions"],
                    row["physical_regressions"],
                    row["reserve_regressions"],
                ]
            )
        lines.append(_markdown_table(tier_headers, tier_body))
        lines.extend(["", "## 新航次原始结果", ""])
        mission_headers = [
            "场景",
            "方法",
            "bit",
            "J",
            "能耗 J",
            "请求校准",
            "执行校准",
            "修复改派校准",
            "首服务0",
            "时刻0 s",
            "首服务1",
            "时刻1 s",
            "未返航",
            "未落地",
            "耗尽",
            "返航失败",
            "低于备用",
            "none触发",
            "首次分歧时段",
        ]
        mission_body = []
        for row in summary.get("episodes", []):
            mission_body.append(
                [
                    row["scenario_id"],
                    row["method"],
                    _i(row, "on_time_bits"),
                    _fmt(row["utility"]),
                    _fmt(row["energy_j"], 3),
                    row.get("calibrate_requested", ""),
                    row.get("calibrate_executed", ""),
                    row.get("repair_calibrate_requests", ""),
                    row.get("first_service_0_target", row.get("first_service_0", "")),
                    row.get("first_service_0_s", ""),
                    row.get("first_service_1_target", row.get("first_service_1", "")),
                    row.get("first_service_1_s", ""),
                    row.get("not_returned", ""),
                    row.get("not_landed", ""),
                    row.get("energy_exhausted", ""),
                    row.get("return_failures", ""),
                    row.get("below_reserve", ""),
                    row.get("tail_trigger_decisions", ""),
                    row.get("first_divergence_epoch", ""),
                ]
            )
        lines.append(_markdown_table(mission_headers, mission_body))
    lines.extend(["", "## 成本修正证据", ""])
    gate = summary.get("snapshot") or {}
    stats = gate.get("stats") or {}
    lines.append(
        f"六个快照的 legacy 重评最大绝对差见下表。评分展开 {stats.get('rollouts', 0)} 次，缓存命中 {stats.get('cache_hits', 0)} 次，尾分段进入积分的片段计数 {stats.get('tail_piece_evals', 0)}。"
    )
    lines.append("")
    if gate.get("snapshots"):
        lines.append(
            _markdown_table(
                ["快照", "类别", "legacy Q绝对差", "正业务不一致", "排序变化", "新联合动作", "修复仍请求校准"],
                [
                    [
                        row["snapshot"],
                        row["registered_class"],
                        _fmt(row["legacy_q_abs"], 3),
                        row["positive_branch_mismatches"],
                        row["rank_changes"],
                        row["new_joint"],
                        row["repair_still_requests_calibrate"],
                    ]
                    for row in gate["snapshots"]
                ],
            )
        )
    lines.extend(["", "三条 single_score 快照的 C 与替代动作。Q 差减去终端差应接近 0，说明尾项进入的是正式评分而不是旁路显示。", ""])
    if gate.get("components"):
        lines.append(
            _markdown_table(
                ["快照", "动作", "R30", "旧终端", "新终端", "新Q", "后验分位", "落地时间 s", "后验推进 J", "地面辅助 J", "通信 J", "Q差减终端差"],
                [
                    [
                        row["snapshot"],
                        row["action"],
                        _fmt(row["r30"]),
                        _fmt(row["legacy_terminal"]),
                        _fmt(row["new_terminal"]),
                        _fmt(row["new_q"]),
                        _fmt(row["posterior_q95"], 6),
                        row["landing_s"],
                        _fmt(row["propulsion_at_posterior_j"], 3),
                        _fmt(row["ground_aux_j"], 3),
                        _fmt(row["radio_j"], 3),
                        _fmt(row["q_gap_minus_terminal_gap"]),
                    ]
                    for row in gate["components"]
                ],
            )
        )
    lines.extend(["", "新分支动作差的局部细网格使用已登记的 31 外层、15 内层。误差界是两边绝对误差相加。未分辨的差距保留原 tie 规则，不加大扫描。", ""])
    if gate.get("fine"):
        lines.append(
            _markdown_table(
                ["快照", "状态", "15点差距", "31点差距", "合并误差界"],
                [
                    [
                        row.get("snapshot", ""),
                        row.get("status", ""),
                        _fmt(row.get("gap15", "")),
                        _fmt(row.get("gap31", "")),
                        _fmt(row.get("combined_abs_error", "")),
                    ]
                    for row in gate["fine"]
                ],
            )
        )
    compute = summary.get("compute") or {}
    mission_stats = compute.get("missions") or {}
    tail = compute.get("tail") or {}
    lines.extend(
        [
            "",
            "## 计算代价",
            "",
            f"快照评分展开 {stats.get('rollouts', 0)} 次，上限 6000。主试评分展开 {mission_stats.get('rollouts', 0)} 次，上限 250000。",
            f"主试缓存命中 {mission_stats.get('cache_hits', 0)}，代数积分 {mission_stats.get('algebraic_integrals', 0)}，代数耗时 {_fmt(mission_stats.get('algebraic_seconds', 0), 3)} 秒，执行步数 {mission_stats.get('physical_steps', 0)}，规划步数 {mission_stats.get('planning_steps', 0)}。",
            f"名义返航报价 {tail.get('return_quotes', 0)} 次，报价缓存命中 {tail.get('return_cache_hits', 0)}，逐秒模拟 {tail.get('return_simulated_seconds', 0)} 秒，解析地面区间 {tail.get('analytic_ground_intervals', 0)}。解析区间的一到两个采样已经计入逐秒步数，定价时不另加一遍。",
            f"总耗时 {_fmt(summary.get('elapsed_s', 0), 1)} 秒。",
            "",
            "## 局部近似仍有的边界",
            "",
            "这只替换旧 none 分支的零值。被零下限舍弃的负收益业务只记录最大值，不在本轮重排。正收益报价里已经包含的回传、返航和落地不再加一份费用。",
            "尾过程按 RETURN 飞行并落地，不把缓存回传当成纯返航，也不给缓存虚构交付收益。另一架 UAV 在报价里等待；它的能耗不进这架 UAV 的终端。",
            "名义轨迹在充足电量下生成，再用当前电量把总能耗截断。截断后的代价是实际耗尽的能量，没有额外惩罚，也不假设已经成功返航。备用电量仍只是准入约束。",
            "落地之后的重复地面区间用同一采样相乘。任务时钟停在 1800 秒。这个修正不是完整的最优终端，也没有改变默认模式。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _base_manifest(cfg: dict, tests_status: str, git_info: dict) -> dict:
    return {
        "instruction": cfg["instruction"],
        "instruction_commit": git_info.get("head"),
        "implementation": "workspace diff at run time; the result commit is the implementation commit",
        "diff_stat": git_info.get("diff_stat", ""),
        "command": [sys.executable, "-m", "wt.active_model_identification.run_m15"],
        "checks": CHECKS,
        "why_these_tests": "只覆盖本轮四个改动面：尾成本对执行器、legacy 六快照、尾项进入积分、正业务与修复不漂移，然后是登记的 24 条航次。m13 与 m14 单测只因为 describe、terminal_value 和 split_action 被本轮碰到。没有跑全库、安全专项或训练。",
        "unit_tests_status": tests_status,
        "seeds_source": "M1.3 scenario manifest; no new seeds",
        "input_hashes": {
            "frozen_config": _sha256(M13_DIR / "frozen_config.json"),
            "scenario_manifest": _sha256(M13_DIR / "scenario_manifest.json"),
            "episodes": _sha256(M13_DIR / "episodes.csv"),
            "decisions": _sha256(M13_DIR / "decisions.csv"),
            "q_decomposition_rows": _sha256(M14_DIR / "q_decomposition_rows.csv"),
            "instruction": _sha256(ROOT / cfg["instruction"]),
        },
        "source_commits": git_info.get("path_commits", {}),
        "tolerances": cfg["tolerances"],
        "budget": cfg["budget"],
        "default_mode": "legacy",
    }


def run(out_dir: Path, force: bool) -> dict:
    _refuse(out_dir)
    if (out_dir / "summary.json").exists() and not force:
        raise SystemExit(f"existing summary at {out_dir}; pass --force to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for name in ("completed_missions.jsonl", "mission_stats.json", "summary.json", "snapshot_gate.json"):
            target = out_dir / name
            if target.exists():
                target.unlink()
    started = time.perf_counter()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if int(cfg["budget"]["snapshot_rollouts"]) > 6000 or int(cfg["budget"]["mission_rollouts"]) > 250000:
        raise SystemExit("config budget exceeds the registered cap")
    if float(cfg["tolerances"]["utility_abs"]) != UTILITY_TOL:
        raise SystemExit("utility tolerance is not the registered 1e-9")
    frozen = json.loads((M13_DIR / "frozen_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((M13_DIR / "scenario_manifest.json").read_text(encoding="utf-8"))
    if int(frozen["belief"]["quadrature_nodes"]) != int(cfg["scoring"]["theta_nodes"]):
        raise SystemExit("frozen theta nodes do not match the M1.5 registration")
    if int(frozen["integral"]["segment_inner_nodes"]) != int(cfg["scoring"]["segment_inner_nodes"]):
        raise SystemExit("frozen inner nodes do not match the M1.5 registration")
    if int(frozen["integral"]["rescore_theta_nodes"]) != int(cfg["scoring"]["fine_theta_nodes"]):
        raise SystemExit("fine theta nodes do not match the frozen rescore grid")
    if int(frozen["integral"]["rescore_inner_nodes"]) != int(cfg["scoring"]["fine_inner_nodes"]):
        raise SystemExit("fine inner nodes do not match the frozen rescore grid")
    if [row["id"] for row in manifest["scenarios"]] != cfg["scenarios"]:
        raise SystemExit("scenario manifest does not match the registered M1.5 list")
    legacy_episodes = [row for row in _read_csv(M13_DIR / "episodes.csv") if row.get("branch") == "core"]
    legacy_decisions = [row for row in _read_csv(M13_DIR / "decisions.csv") if row.get("branch") == "core"]
    if len(legacy_episodes) != 24:
        raise SystemExit(f"expected 24 legacy core episodes, found {len(legacy_episodes)}")
    legacy_spec = _with_mode(frozen, "legacy")
    tail_spec = _with_mode(frozen, TAIL_MODE)
    factor = float(manifest["budget"]["integration_factor"])
    (out_dir / "frozen_config.json").write_text(json.dumps(tail_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    reset_tail_stats()
    clear_return_tail_cache()
    clear_completion_cache()
    clear_rollout_cache()
    git_info = _git(
        [
            "results/active_model_identification_m13/m13_20261114/episodes.csv",
            "results/active_model_identification_m13/m13_20261114/frozen_config.json",
        ]
    )
    git_info["seeds"] = sorted({int(row["seed"]) for row in manifest["scenarios"]})
    print("tests", flush=True)
    test_code = _run_tests(out_dir / "necessary_tests.log")
    tests_status = "PASS" if test_code == 0 and _tests_ok(out_dir / "necessary_tests.log") else "FAIL"
    run_manifest = _base_manifest(cfg, tests_status, git_info)
    run_manifest["seeds"] = git_info["seeds"]
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "instruction_commit": git_info.get("head"),
        "unit_tests_status": tests_status,
        "default_mode": "legacy",
        "terminal_mode": TAIL_MODE,
        "missions_ran": False,
        "stop_reason": "",
        "gate_errors": [],
    }
    if tests_status != "PASS":
        summary["stop_reason"] = "necessary_tests_failed"
        summary["elapsed_s"] = time.perf_counter() - started
        _emit(out_dir, summary)
        return summary
    gate = _snapshots(out_dir, legacy_spec, tail_spec, factor, cfg)
    summary["snapshot"] = gate
    summary["gate_errors"] = gate["errors"]
    if not gate["passed"]:
        summary["stop_reason"] = "snapshot_gate_failed"
        summary["elapsed_s"] = time.perf_counter() - started
        summary["compute"] = {"snapshots": gate["stats"], "tail": gate["tail"]}
        _emit(out_dir, summary)
        return summary
    tail_after_snapshots = tail_stats()
    mission = _missions(out_dir, tail_spec, manifest, cfg, factor, legacy_decisions)
    summary["missions_ran"] = True
    summary["stop_reason"] = mission["reason"]
    summary["episodes"] = mission["episodes"]
    summary["mission_complete"] = mission["complete"]
    if mission["complete"]:
        comparisons = _comparisons(mission["episodes"], legacy_episodes, cfg)
        summary["comparisons"] = comparisons
        summary["tiers"] = _tier_rows(comparisons)
        summary["judgment"] = _judge(comparisons, True)
    else:
        summary["judgment"] = {
            "standard_met": False,
            "primary_policy_retained": False,
            "conclusion": "本轮未满足性能保留标准",
            "complete": False,
            "default_mode": "legacy",
        }
    tail_after = tail_stats()
    summary["compute"] = {
        "snapshots": gate["stats"],
        "missions": mission["stats"],
        "tail_after_snapshots": tail_after_snapshots,
        "tail": {key: int(tail_after[key]) - int(tail_after_snapshots.get(key, 0)) for key in tail_after},
        "tail_total": tail_after,
    }
    summary["elapsed_s"] = time.perf_counter() - started
    _write_csv(out_dir / "episodes.csv", mission["episodes"])
    _write_csv(out_dir / "decisions.csv", mission["decisions"])
    _write_csv(out_dir / "deliveries.csv", mission["deliveries"])
    if mission["complete"]:
        _write_csv(out_dir / "comparisons.csv", summary["comparisons"])
        _write_csv(out_dir / "tier_summary.csv", summary["tiers"])
    _emit(out_dir, summary)
    return summary


def _emit(out_dir: Path, summary: dict) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    compute = summary.get("compute")
    if compute is not None:
        (out_dir / "compute_stats.json").write_text(json.dumps(compute, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["stop_reason"] = summary.get("stop_reason", "")
        manifest["conclusion"] = (summary.get("judgment") or {}).get("conclusion", "")
        manifest["elapsed_s"] = summary.get("elapsed_s")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, ROOT / "docs" / "research" / "Active_Model_Identification_M15_Report_CN.md")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse_parser()
    args = parser.parse_args()
    summary = run(ROOT / args.out, args.force)
    judgment = summary.get("judgment") or {}
    print(
        json.dumps(
            {
                "tests": summary.get("unit_tests_status"),
                "stop": summary.get("stop_reason"),
                "conclusion": judgment.get("conclusion", ""),
                "standard_met": judgment.get("standard_met"),
                "primary": judgment.get("primary_policy_retained"),
                "elapsed_s": summary.get("elapsed_s"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def argparse_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/active_model_identification_m15/m15_20261114")
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    main()
