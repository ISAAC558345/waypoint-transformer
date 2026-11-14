"""Run the M1.6 initial-guard selection check.

The 180 s review is recorded before any new mission. A mid-interval substitution,
a failed reproduction, or a b3_loose primary-OFF regression stops the round.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import wt.active_model_identification.world as world_mod
from wt.active_model_identification.belief import prior_belief, quadrature
from wt.active_model_identification.completion import (
    TAIL_MODE,
    admits,
    clear_completion_cache,
    clear_return_tail_cache,
    completion_quote,
    quote_stats,
    reset_quote_stats,
    reset_tail_stats,
    tail_stats,
)
from wt.active_model_identification.policies import (
    GUARD_SELECTION,
    LEGACY_SELECTION,
    initial_effective_action,
    pick_action,
    select_lookahead,
    selectable_candidates,
    tie_abs,
)
from wt.active_model_identification.run_m12 import _sha256, _write_csv
from wt.active_model_identification.run_m13 import METHODS, _append_progress, _load_progress, _play, _stats
from wt.active_model_identification.scenarios import observation_noise
from wt.active_model_identification.segment import BudgetExceeded, clear_rollout_cache, roll_action
from wt.active_model_identification.spec import ROOT
from wt.active_model_identification.world import action_allowed, advance, make_world, on_time_bits, utility

CONFIG_PATH = ROOT / "configs" / "experiments" / "active_model_identification_m16.json"
M13_DIR = ROOT / "results" / "active_model_identification_m13" / "m13_20261114"
M15_DIR = ROOT / "results" / "active_model_identification_m15" / "m15_20261114"
UTILITY_TOL = 1e-9
PHYSICAL = ("not_returned", "not_landed", "energy_exhausted", "return_failures", "below_reserve")
OFF = "lookahead_calibration_off"
ON = "lookahead_calibration_on"


def _refuse(out_dir: Path) -> None:
    if "active_model_identification_m16" not in set(out_dir.parts):
        raise SystemExit("refusing to write outside results/active_model_identification_m16")
    if "active_model_identification_m13" in set(out_dir.parts) or "active_model_identification_m15" in set(out_dir.parts):
        raise SystemExit("refusing to overwrite an earlier result directory")


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _label(action) -> str:
    return f"{action[0]}:{action[1]}"


def _parse(label: str):
    kind, target = str(label).split(":")
    return (kind, None if target == "None" else int(target))


def _public(stats: dict) -> dict:
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
        "selection_candidates",
        "selection_kept",
        "selection_aliases",
        "budget_exhausted",
    )
    return {key: stats.get(key, 0) for key in keys}


def _with_modes(spec: dict, terminal: str, selection: str) -> dict:
    cloned = copy.deepcopy(spec)
    cloned["terminal"] = {"mode": terminal}
    cloned["selection"] = {"mode": selection}
    return cloned


def _index_decisions(rows: list[dict]) -> dict:
    found = {}
    for row in rows:
        if row.get("branch") not in (None, "", "core"):
            continue
        if row.get("scenario_id") != "b3_loose" and "scenario_id" in row:
            continue
        if row.get("scenario_id") not in (None, "b3_loose"):
            continue
        found.setdefault(row["method"], {})[(int(row["epoch"]), int(row["uav"]))] = row
    return found


def _episode_map(rows: list[dict]) -> dict:
    return {
        (row["scenario_id"], row["method"]): row
        for row in rows
        if row.get("branch", "core") == "core"
    }


def _num(row: dict, key: str) -> float:
    return float(row[key])


def _bits(row: dict) -> int:
    return int(round(float(row["on_time_bits"])))


def _run_tests(log_path: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_active_model_identification_m16",
        "tests.test_active_model_identification_m15",
        "tests.test_active_model_identification_m13",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return int(completed.returncode)


def _replay(spec: dict, scenario: dict, method: str, requests: dict, epochs: range, factor: float):
    world = make_world(spec, scenario["battery_j"], f"{scenario['id']}-{method}")
    beliefs = [prior_belief(spec), prior_belief(spec)]
    noise = observation_noise(int(scenario["seed"]), float(spec["belief"]["sigma_j"]))
    events = []
    previous = {packet.id: packet.holder for packet in world.packets if packet.buoy == 0}
    for epoch in epochs:
        actions = [_parse(requests[(epoch, uav)]["requested"]) for uav in (0, 1)]
        advance(
            world,
            actions,
            list(scenario["theta"]),
            beliefs,
            spec,
            factor,
            None,
            planning=False,
            interval_index=epoch,
            noise_j=noise,
        )
        for packet in world.packets:
            if packet.buoy != 0 or previous.get(packet.id) == packet.holder:
                continue
            events.append(
                {
                    "method": method,
                    "epoch": epoch,
                    "t_end_s": world.t,
                    "packet": packet.id,
                    "from": previous.get(packet.id),
                    "to": packet.holder,
                    "delivered_s": packet.delivered_s,
                    "remaining_bits": packet.remaining_bits,
                }
            )
            previous[packet.id] = packet.holder
    final = []
    for packet in world.packets:
        if packet.buoy != 0:
            continue
        final.append(
            {
                "method": method,
                "packet": packet.id,
                "holder": packet.holder,
                "remaining_bits": packet.remaining_bits,
                "delivered_s": packet.delivered_s,
                "bits": packet.bits,
            }
        )
    return world, beliefs, events, final


def _state_errors(world, beliefs, saved: dict, tolerances: dict) -> list[str]:
    errors = []
    for uav in world.uavs:
        row = saved[(5, uav.index)]
        mean, variance = beliefs[uav.index].moments()
        checks = (
            ("x", uav.x, tolerances["replay_position_abs"]),
            ("y", uav.y, tolerances["replay_position_abs"]),
            ("energy_j", uav.energy_j, tolerances["replay_energy_abs"]),
            ("mean_after", mean, tolerances["replay_belief_abs"]),
            ("variance_after", variance, tolerances["replay_belief_abs"]),
        )
        for key, value, tol in checks:
            saved_key = key if key in row else key
            if abs(float(value) - float(row[saved_key])) > tol:
                errors.append(f"uav{uav.index} {key} {value} != {row[saved_key]}")
        if str(uav.airborne) != row["airborne"]:
            errors.append(f"uav{uav.index} airborne {uav.airborne} != {row['airborne']}")
    return errors


def _guard_record(world, beliefs, spec: dict, factor: float, uav_index: int) -> dict:
    uav = world.uavs[uav_index]
    belief = beliefs[uav_index]
    theta = belief.quantile(float(spec["belief"]["guard_probability"]))
    quote = completion_quote(world, uav_index, 0, theta, spec)
    allowed = action_allowed(uav, ("service", 0), theta, spec, factor, world)
    effective = initial_effective_action(world, uav_index, ("service", 0), belief, spec, factor)
    return {
        "uav": uav_index,
        "theta_q95": theta,
        "energy_j": uav.energy_j,
        "reserve_j": float(spec["energy"]["reserve_j"]),
        "quote_energy_j": quote.energy_j,
        "quote_finished": quote.finished,
        "quote_landed": quote.landed,
        "quote_on_time": quote.on_time,
        "quote_bits": quote.on_time_bits,
        "admits": admits(uav, quote, spec),
        "action_allowed": allowed,
        "initial_effective": _label(effective),
        "cache": list(uav.cache_ids),
        "x": uav.x,
        "y": uav.y,
        "airborne": uav.airborne,
    }


def _trace_interval(world, beliefs, spec, factor, scenario, requested, saved_executed) -> dict:
    calls = []
    option = world_mod.apply_option_guard
    guard = world_mod._return_guard

    def traced_option(uav, action, belief, spec_arg, factor_arg, traced_world=None):
        updated, changed = option(uav, action, belief, spec_arg, factor_arg, traced_world)
        calls.append({"guard": "option", "uav": uav.index, "phase": 0, "incoming": _label(action), "outgoing": _label(updated), "changed": changed, "energy_j": uav.energy_j})
        return updated, changed

    def traced_return(uav, action, belief, spec_arg, factor_arg):
        updated, changed = guard(uav, action, belief, spec_arg, factor_arg)
        phase = sum(1 for item in calls if item["guard"] == "return" and item["uav"] == uav.index)
        calls.append({"guard": "return", "uav": uav.index, "phase": phase, "incoming": _label(action), "outgoing": _label(updated), "changed": changed, "energy_j": uav.energy_j})
        return updated, changed

    snap = world.snapshot()
    copied = [item.copy() for item in beliefs]
    world_mod.apply_option_guard = traced_option
    world_mod._return_guard = traced_return
    try:
        observations = advance(
            snap,
            requested,
            list(scenario["theta"]),
            copied,
            spec,
            factor,
            None,
            planning=False,
            interval_index=6,
            noise_j=observation_noise(int(scenario["seed"]), float(spec["belief"]["sigma_j"])),
        )
    finally:
        world_mod.apply_option_guard = option
        world_mod._return_guard = guard
    rows = []
    initial = True
    mid = False
    for uav in (0, 1):
        option_row = next(item for item in calls if item["guard"] == "option" and item["uav"] == uav and item["incoming"] == _label(requested[uav]))
        return_changes = [item for item in calls if item["guard"] == "return" and item["uav"] == uav and item["changed"]]
        executed = _label(tuple(observations[uav]["executed"]))
        phase = None if not return_changes else int(return_changes[0]["phase"])
        requested_label = _label(requested[uav])
        converted = requested_label.startswith("service:") and executed == "return:None"
        at_start = converted and option_row["changed"] and option_row["outgoing"] == "return:None" and phase is None
        at_middle = converted and (not option_row["changed"]) and phase is not None and phase > 0
        if converted and not at_start:
            initial = False
        if at_middle:
            mid = True
        rows.append(
            {
                "uav": uav,
                "requested": requested_label,
                "option_outgoing": option_row["outgoing"],
                "option_changed": option_row["changed"],
                "return_change_phase": phase,
                "executed": executed,
                "saved_executed": saved_executed[uav],
                "executed_matches_saved": executed == saved_executed[uav],
                "initial_substitution": at_start,
                "mid_interval_substitution": at_middle,
            }
        )
    return {"calls": calls, "uavs": rows, "initial_substitution": initial and any(row["initial_substitution"] for row in rows), "mid_interval_substitution": mid}


def _score_snapshot(method: str, world, beliefs, specs: dict, factor: float, stats: dict, saved: dict, q_tol: float) -> dict:
    tau = tie_abs(next(iter(specs.values())))
    modes = {}
    errors = []
    for name, spec in specs.items():
        if int(stats.get("rollouts", 0)) >= int(stats["rollout_cap"]):
            stats["budget_exhausted"] = True
            errors.append(f"{method} {name} diagnostic cap")
            break
        joint, tables = select_lookahead(method, world, beliefs, spec, factor, stats)
        mode_rows = []
        for uav, table in enumerate(tables):
            _kept, mapping = selectable_candidates(
                spec, world, beliefs, uav, include_calibrate=method == ON, factor=factor
            )
            mapped = {_label(row["requested"]): row for row in mapping}
            grid = quadrature(beliefs[uav], int(spec["belief"]["quadrature_nodes"]))
            for rank, (score, action) in enumerate(table):
                roll = roll_action(world, beliefs, action, uav, float(grid[0][0]), spec, factor, stats)
                label = _label(action)
                info = mapped[label]
                mode_rows.append(
                    {
                        "method": method,
                        "mode": name,
                        "uav": uav,
                        "rank": rank,
                        "action": label,
                        "score": score,
                        "valued_executed": _label(roll.executed),
                        "initial_effective": _label(info["effective"]),
                        "alias": info["alias"],
                    }
                )
        pick = [pick_action(table, tau) for table in tables]
        record = {
            "joint": [_label(action) for action in joint],
            "pick": [_label(item[1]) for item in pick],
            "pick_q": [item[0] for item in pick],
            "rows": mode_rows,
        }
        modes[name] = record
        if name == "tail_legacy_selection":
            for uav in (0, 1):
                row = saved["m15"][method][(6, uav)]
                if record["joint"][uav] != row["requested"]:
                    errors.append(f"{method} uav{uav} joint {record['joint'][uav]} != m15 {row['requested']}")
                if record["pick"][uav] != row["table_choice"]:
                    errors.append(f"{method} uav{uav} pick {record['pick'][uav]} != m15 {row['table_choice']}")
                if abs(float(record["pick_q"][uav]) - float(row["chosen_q"])) > q_tol:
                    errors.append(f"{method} uav{uav} q {record['pick_q'][uav]} != m15 {row['chosen_q']}")
        if name == "legacy_terminal":
            for uav in (0, 1):
                row = saved["m13"][method][(6, uav)]
                if record["joint"][uav] != row["requested"]:
                    errors.append(f"{method} legacy joint uav{uav} {record['joint'][uav]} != m13 {row['requested']}")
                if record["pick"][uav] != row["table_choice"]:
                    errors.append(f"{method} legacy pick uav{uav} {record['pick'][uav]} != m13 {row['table_choice']}")
        if name == "tail_guard_selection":
            for uav, action in enumerate(joint):
                effective = initial_effective_action(world, uav, action, beliefs[uav], spec, factor)
                if effective != action:
                    errors.append(f"{method} uav{uav} new request {_label(action)} executes as {_label(effective)}")
                if action == ("service", 0):
                    service_effective = initial_effective_action(world, uav, ("service", 0), beliefs[uav], spec, factor)
                    if service_effective != ("service", 0):
                        errors.append(f"{method} uav{uav} inadmissible service:0 remained selected")
    return {"modes": modes, "errors": errors}


def _pair(name: str, new_row: dict, old_row: dict) -> dict:
    bits_delta = _bits(new_row) - _bits(old_row)
    utility_delta = _num(new_row, "utility") - _num(old_row, "utility")
    physical = [key for key in PHYSICAL if int(round(_num(new_row, key))) > int(round(_num(old_row, key)))]
    reserve = int(round(_num(new_row, "below_reserve"))) > int(round(_num(old_row, "below_reserve")))
    return {
        "comparison": name,
        "scenario_id": new_row["scenario_id"],
        "base": new_row["base"],
        "condition": new_row["condition"],
        "bits_new": _bits(new_row),
        "bits_old": _bits(old_row),
        "bits_delta": bits_delta,
        "utility_new": _num(new_row, "utility"),
        "utility_old": _num(old_row, "utility"),
        "utility_delta": utility_delta,
        "energy_new": _num(new_row, "energy_j"),
        "energy_old": _num(old_row, "energy_j"),
        "energy_delta": _num(new_row, "energy_j") - _num(old_row, "energy_j"),
        "bits_regressed": bits_delta < 0,
        "utility_regressed": utility_delta < -UTILITY_TOL,
        "regressed": bits_delta < 0 or utility_delta < -UTILITY_TOL,
        "utility_improved": utility_delta > UTILITY_TOL,
        "physical_worse": physical,
        "reserve_worse": reserve or "below_reserve" in physical,
        "below_reserve_new": int(round(_num(new_row, "below_reserve"))),
        "below_reserve_old": int(round(_num(old_row, "below_reserve"))),
    }


def _play_guarded(method, scenario, spec, factor, stats, world, beliefs, start_epoch, legacy_requests):
    mapping_rows = []

    def capture(epoch, captured_world, captured_beliefs, requested, _method=method, _spec=spec, _factor=factor, _rows=mapping_rows):
        for uav in range(len(captured_world.uavs)):
            _kept, mapping = selectable_candidates(
                _spec,
                captured_world,
                captured_beliefs,
                uav,
                include_calibrate=_method == ON,
                factor=_factor,
            )
            for row in mapping:
                _rows.append(
                    {
                        "method": _method,
                        "scenario_id": scenario["id"],
                        "epoch": epoch,
                        "uav": uav,
                        "requested_label": _label(row["requested"]),
                        "effective": _label(row["effective"]),
                        "kept": row["kept"],
                        "alias": row["alias"],
                        "joint": _label(requested[uav]),
                    }
                )

    episode, decisions, deliveries, _snapshot = _play(
        method, scenario, spec, factor, stats, "core", start_epoch, world, beliefs, [], capture
    )
    episode["terminal_mode"] = TAIL_MODE
    episode["selection_mode"] = GUARD_SELECTION
    first = None
    for epoch in range(start_epoch, 60):
        current = {int(row["uav"]): row["requested"] for row in decisions if int(row["epoch"]) == epoch}
        if not current or (epoch, 0) not in legacy_requests or (epoch, 1) not in legacy_requests:
            continue
        previous = {uav: legacy_requests[(epoch, uav)]["requested"] for uav in (0, 1)}
        if current.get(0) != previous[0] or current.get(1) != previous[1]:
            first = {"epoch": epoch, "legacy": previous, "new": current}
            break
    episode["first_divergence_epoch"] = "" if first is None else first["epoch"]
    episode["first_divergence_legacy"] = "" if first is None else f"{first['legacy'][0]}|{first['legacy'][1]}"
    episode["first_divergence_new"] = "" if first is None else f"{first['new'].get(0)}|{first['new'].get(1)}"
    if start_epoch == 0 and legacy_requests:
        prefix_match = all(
            next(row["requested"] for row in decisions if int(row["epoch"]) == epoch and int(row["uav"]) == uav)
            == legacy_requests[(epoch, uav)]["requested"]
            for epoch in range(6)
            for uav in (0, 1)
        )
    else:
        prefix_match = ""
    episode["prefix_match_before_180s"] = prefix_match
    return episode, decisions, deliveries, mapping_rows


def _judge(comparisons: list[dict], scenarios: list[str], complete: bool) -> dict:
    off_names = ("new_OFF-m15_OFF", "new_OFF-m13_OFF")
    on_names = ("new_ON-m15_ON", "new_ON-m13_ON")

    def covered(names: tuple[str, ...]) -> bool:
        return all(any(row["comparison"] == name and row["scenario_id"] == scenario for row in comparisons) for name in names for scenario in scenarios)

    def clean(names: tuple[str, ...]) -> bool:
        rows = [row for row in comparisons if row["comparison"] in names]
        return bool(rows) and all(not row["regressed"] and not row["physical_worse"] and not row["reserve_worse"] for row in rows)

    off_improved = any(row["comparison"] in off_names and row["utility_improved"] for row in comparisons)
    support = complete and covered(off_names) and clean(off_names) and off_improved
    on_pass = complete and covered(on_names) and clean(on_names)
    if support:
        conclusion = "被动候选在本组登记场景得到性能支持"
    elif not complete:
        conclusion = "本轮在进入全部 24 条之前结束"
    else:
        conclusion = "本轮未满足被动候选的逐场景性能标准"
    return {
        "passive_supported": support,
        "on_no_regression": on_pass,
        "off_utility_improved": off_improved,
        "conclusion": conclusion,
        "default_selection": LEGACY_SELECTION,
        "default_terminal": "legacy",
    }


def _fmt(value, digits: int = 9) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def write_report(path: Path, summary: dict) -> None:
    review = summary.get("review", {})
    judgment = summary.get("judgment", {})
    lines = [
        "# M1.6：可执行候选与浮标分配一致性",
        "",
        f"性能判定：{judgment.get('conclusion', summary.get('stop_reason', '未完成'))}。",
        "",
        "默认选择模式仍是 legacy，默认终端模式仍是 legacy。本轮候选同时使用 M1.5 的 return_tail_when_no_business 和 initial_guard_consistent，只在显式打开时生效。主要候选是保留普通业务遥测的 OFF，ON 是对照。校准开关的差值只说明行为，不解释成信息价值。本轮没有验证风浪感知或共享环境收益。",
        "",
        "## 期初替代",
        "",
    ]
    if review:
        lines.append(
            f"b3_loose 在 180 秒前的请求与 M1.3 一致：{review.get('prefix_matches_m13')}。重放状态与 M1.5 日志一致：{review.get('state_matches')}。"
        )
        lines.append(
            f"期初替代成立：{review.get('initial_substitution')}。区间中途替代：{review.get('mid_interval_substitution')}。"
        )
        for row in review.get("guards", []):
            lines.append(
                f"{row['method']} UAV{row['uav']}：电量 {_fmt(row['energy_j'], 3)} J，service:0 报价 {_fmt(row['quote_energy_j'], 3)} J，准入 {row['admits']}，期初有效动作 {row['initial_effective']}。"
            )
        for row in review.get("traces", []):
            for item in row["uavs"]:
                if not str(item["requested"]).startswith("service:"):
                    continue
                lines.append(
                    f"{row['method']} UAV{item['uav']} 请求 {item['requested']}，期初守卫输出 {item['option_outgoing']}，返航守卫改变相位 {item['return_change_phase']}，区间末执行 {item['executed']}。"
                )
    lines.extend(["", "## 候选次序与 20000 bit", ""])
    for row in review.get("score_summary", []):
        lines.append(
            f"{row['method']}：legacy 联合请求 {row['legacy_joint']}，M1.5 尾模式联合请求 {row['tail_joint']}，新选择联合请求 {row['guard_joint']}。尾模式首选 {row['tail_pick']}，legacy 首选 {row['legacy_pick']}。"
        )
    lines.append("M1.5 在 180 秒把两机首选都放到 service:0 后，冲突修复让绝对 Q 较低的一架改为 RETURN；另一架分到的 service:0 在期初就被原准入器改成 RETURN。缺掉的是浮标 0 的两个 10000 bit 包。")
    for row in review.get("buoy0_final", []):
        lines.append(f"{row['source']} {row['method']} {row['packet']} 最终持有者 {row['holder']}，交付时刻 {row['delivered_s']}。")
    lines.extend(["", "## 新航次", ""])
    if not summary.get("missions_ran"):
        lines.append("诊断或实现验证未通过，没有运行新航次。")
    else:
        lines.append(f"b3_loose 先行 OFF 是否保持 M1.3 与 M1.5 的交付量和净效用：{summary.get('pilot_off_keeps')}。其余场景是否运行：{summary.get('rest_ran')}。")
        if summary.get("pilot_note"):
            lines.append(str(summary["pilot_note"]))
        lines.extend(
            [
                "",
                "| 场景 | 比较 | bit差 | J差 | 能耗差 J | 回退 | 物理变差 | 备用电量变差 |",
                "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in summary.get("comparisons", []):
            lines.append(
                f"| {row['scenario_id']} | {row['comparison']} | {row['bits_delta']} | {_fmt(row['utility_delta'])} | {_fmt(row['energy_delta'], 3)} | {int(row['regressed'])} | {','.join(row['physical_worse'])} | {int(row['reserve_worse'])} |"
            )
        lines.extend(["", "| 场景 | 方法 | bit | J | 能耗 J | 180秒前一致 | 首次分歧 |", "| --- | --- | ---: | ---: | ---: | --- | --- |"])
        for row in summary.get("episodes", []):
            lines.append(
                f"| {row['scenario_id']} | {row['method']} | {_bits(row)} | {_fmt(row['utility'])} | {_fmt(row['energy_j'], 3)} | {row.get('prefix_match_before_180s', '')} | {row.get('first_divergence_epoch', '')} |"
            )
    lines.extend(
        [
            "",
            "## 计算",
            "",
            f"诊断评分展开 {summary.get('diagnostic_rollouts')}，上限 6000。主试评分展开 {summary.get('mission_rollouts')}，上限 250000。同状态续跑展开 {summary.get('continuation_rollouts')}，上限 24000。",
            f"选择别名 {summary.get('selection_aliases')}，保留候选 {summary.get('selection_kept')}。准入报价新算 {summary.get('admission_quotes')}，命中 {summary.get('admission_hits')}。尾报价逐秒 {summary.get('tail_simulated_seconds')}。总耗时 {_fmt(summary.get('elapsed_s'), 1)} 秒。",
            "去重少展开的是被期初守卫改写的标签。报价缓存只复用同一物理状态，不改变动作价值。这里的耗时不与旧航次的墙钟时间比作加速。",
            "",
            "## 仍保留的边界",
            "",
            "这只规范期初会被原守卫改写的标签。当前可准入、但本区间后段才触发返航的动作仍按原模型预测。冲突规则、tie 容差和终端公式没有改。另一架 UAV 在单机评分里等待。负结果和未跑的场景都不另做规则搜索。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _git() -> dict:
    try:
        head = subprocess.check_output(["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        diff = subprocess.check_output(["git", "-c", "core.longpaths=true", "diff", "--stat"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError as exc:
        return {"head": "unknown", "diff_stat": "", "error": str(exc)}
    return {"head": head, "diff_stat": diff}


def run(out_dir: Path) -> dict:
    _refuse(out_dir)
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if float(cfg["tolerances"]["utility_abs"]) != UTILITY_TOL:
        raise SystemExit("utility tolerance is not the registered 1e-9")
    if int(cfg["budget"]["diagnostic_rollouts"]) != 6000 or int(cfg["budget"]["mission_rollouts"]) != 250000:
        raise SystemExit("budgets do not match the registration")
    frozen = json.loads((M13_DIR / "frozen_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((M13_DIR / "scenario_manifest.json").read_text(encoding="utf-8"))
    if [row["id"] for row in manifest["scenarios"]] != cfg["scenarios"]:
        raise SystemExit("scenario order does not match the registration")
    if int(frozen["belief"]["quadrature_nodes"]) != int(cfg["scoring"]["theta_nodes"]):
        raise SystemExit("frozen theta nodes differ")
    if int(frozen["integral"]["segment_inner_nodes"]) != int(cfg["scoring"]["segment_inner_nodes"]):
        raise SystemExit("frozen inner nodes differ")
    factor = float(manifest["budget"]["integration_factor"])
    scenario = next(row for row in manifest["scenarios"] if row["id"] == cfg["pilot_scenario"])
    m13_decisions = _index_decisions([row for row in _read_csv(M13_DIR / "decisions.csv") if row.get("branch") == "core" and row["scenario_id"] == "b3_loose"])
    m15_decisions = _index_decisions([row for row in _read_csv(M15_DIR / "decisions.csv") if row.get("branch") == "core" and row["scenario_id"] == "b3_loose"])
    m13_episodes = _episode_map(_read_csv(M13_DIR / "episodes.csv"))
    m15_episodes = _episode_map(_read_csv(M15_DIR / "episodes.csv"))
    candidate = _with_modes(frozen, TAIL_MODE, GUARD_SELECTION)
    specs = {
        "legacy_terminal": _with_modes(frozen, "legacy", LEGACY_SELECTION),
        "tail_legacy_selection": _with_modes(frozen, TAIL_MODE, LEGACY_SELECTION),
        "tail_guard_selection": candidate,
    }
    (out_dir / "frozen_config.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    clear_rollout_cache()
    clear_completion_cache()
    clear_return_tail_cache()
    reset_tail_stats()
    reset_quote_stats()
    manifest_doc = {
        "instruction": cfg["instruction"],
        "instruction_commit": cfg["instruction_commit"],
        "implementation": _git(),
        "checks": [
            {"change": "期初守卫把 service 改成 RETURN 的时刻", "impact": "若是区间中途替代，选择映射会修错位置", "check": "重放 b3_loose 前六个区间，记录期初守卫和逐秒替代"},
            {"change": "新增 initial_guard_consistent", "impact": "不可执行的 service 标签可能继续占住浮标，或把可执行冲突修坏", "check": "一架不准入时标签退出；两架可执行时原修复结果不变；别名只评分一次"},
            {"change": "正式入口使用新选择和 M1.5 尾终端", "impact": "180 秒的旧表可能对不上，或新请求仍不可执行", "check": "两份冻结快照重现 M1.5 联合请求；新模式请求与期初执行器一致"},
            {"change": "b3_loose 先行后再决定是否补 22 条", "impact": "主要 OFF 可能继续少交付", "check": "新 OFF 对 M1.3 OFF 和 M1.5 OFF 的 bit 与 J 都不下降才继续"},
        ],
        "why_these_tests": "只覆盖期初替代时刻、新选择语义、冻结快照重现和先行性能门槛。m13 与 m15 单测只因为 score_actions 的缺省路径经过同一入口。没有跑全库或安全专项。",
        "input_hashes": {
            "m13_frozen_config": _sha256(M13_DIR / "frozen_config.json"),
            "m13_manifest": _sha256(M13_DIR / "scenario_manifest.json"),
            "m13_episodes": _sha256(M13_DIR / "episodes.csv"),
            "m13_decisions": _sha256(M13_DIR / "decisions.csv"),
            "m15_episodes": _sha256(M15_DIR / "episodes.csv"),
            "m15_decisions": _sha256(M15_DIR / "decisions.csv"),
            "instruction": _sha256(ROOT / cfg["instruction"]),
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("tests", flush=True)
    test_code = _run_tests(out_dir / "necessary_tests.log")
    tests_ok = test_code == 0 and "\nOK" in (out_dir / "necessary_tests.log").read_text(encoding="utf-8", errors="replace")
    summary = {
        "tests": "PASS" if tests_ok else "FAIL",
        "missions_ran": False,
        "rest_ran": False,
        "pilot_off_keeps": False,
        "stop_reason": "" if tests_ok else "unit_tests_failed",
        "review": {},
        "comparisons": [],
        "episodes": [],
    }
    if not tests_ok:
        summary["elapsed_s"] = time.perf_counter() - started
        _emit(out_dir, summary)
        return summary

    print("review", flush=True)
    review_errors = []
    guards = []
    traces = []
    buoy_final = []
    buoy_events = []
    prefix_matches = True
    for method in METHODS:
        m15_rows = m15_decisions[method]
        m13_rows = m13_decisions[method]
        for epoch in range(6):
            for uav in (0, 1):
                if m15_rows[(epoch, uav)]["requested"] != m13_rows[(epoch, uav)]["requested"]:
                    prefix_matches = False
                    review_errors.append(f"{method} epoch {epoch} uav {uav} prefix request differs")
                if m15_rows[(epoch, uav)]["executed"] != m13_rows[(epoch, uav)]["executed"]:
                    prefix_matches = False
                    review_errors.append(f"{method} epoch {epoch} uav {uav} prefix executed differs")
        world, beliefs, _events, _final = _replay(frozen, scenario, method, m15_rows, range(6), factor)
        state_errors = _state_errors(world, beliefs, m15_rows, cfg["tolerances"])
        review_errors.extend(f"{method} {item}" for item in state_errors)
        for uav in (0, 1):
            record = _guard_record(world, beliefs, frozen, factor, uav)
            record["method"] = method
            guards.append(record)
        requested = [_parse(m15_rows[(6, uav)]["requested"]) for uav in (0, 1)]
        traced = _trace_interval(
            world,
            beliefs,
            frozen,
            factor,
            scenario,
            requested,
            [m15_rows[(6, uav)]["executed"] for uav in (0, 1)],
        )
        traced["method"] = method
        traces.append(traced)
        full, _beliefs, events, final = _replay(frozen, scenario, method, m15_rows, range(60), factor)
        for row in final:
            buoy_final.append({"source": "m15_replay", **row})
        buoy_events.extend({"source": "m15_replay", **row} for row in events)
        episode = m15_episodes[(scenario["id"], method)]
        if _bits(episode) != int(round(on_time_bits(full))):
            review_errors.append(f"{method} full replay bits {on_time_bits(full)} != m15 {episode['on_time_bits']}")
    for traced in traces:
        for item in traced["uavs"]:
            if not item["executed_matches_saved"]:
                review_errors.append(f"{traced['method']} uav{item['uav']} executed {item['executed']} != saved {item['saved_executed']}")
    initial = all(row["initial_substitution"] for row in traces) and not review_errors
    mid = any(row["mid_interval_substitution"] for row in traces)
    review = {
        "prefix_matches_m13": prefix_matches and not any("prefix" in item for item in review_errors),
        "state_matches": not any("uav" in item and "prefix" not in item for item in review_errors),
        "initial_substitution": initial and not mid,
        "mid_interval_substitution": mid,
        "errors": review_errors,
        "guards": guards,
        "traces": traces,
        "buoy0_final": buoy_final,
        "score_summary": [],
    }
    summary["review"] = review
    _write_csv(out_dir / "guard_records.csv", guards)
    _write_csv(out_dir / "substitution_trace.csv", [item for row in traces for item in row["uavs"]])
    _write_csv(out_dir / "buoy0_events.csv", buoy_events)
    _write_csv(out_dir / "buoy0_final.csv", buoy_final)
    if mid or not review["initial_substitution"]:
        summary["stop_reason"] = "mid_interval_substitution" if mid else "initial_substitution_not_confirmed"
        summary["elapsed_s"] = time.perf_counter() - started
        (out_dir / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
        _emit(out_dir, summary)
        return summary

    print("snapshot scores", flush=True)
    stats = _stats()
    stats["rollout_cap"] = int(cfg["budget"]["diagnostic_rollouts"])
    score_rows = []
    score_errors = list(review_errors)
    saved = {"m13": m13_decisions, "m15": m15_decisions}
    for method in METHODS:
        world, beliefs, _events, _final = _replay(frozen, scenario, method, m15_decisions[method], range(6), factor)
        try:
            scored = _score_snapshot(method, world, beliefs, specs, factor, stats, saved, float(cfg["tolerances"]["rescore_q_abs"]))
        except BudgetExceeded:
            stats["budget_exhausted"] = True
            score_errors.append(f"{method} diagnostic cap")
            break
        score_errors.extend(scored["errors"])
        for mode, record in scored["modes"].items():
            score_rows.extend(record["rows"])
            review["score_summary"].append(
                {
                    "method": method,
                    "mode": mode,
                    "legacy_joint": "",
                    "tail_joint": "",
                    "guard_joint": "",
                    "legacy_pick": "",
                    "tail_pick": "",
                }
            )
        modes = scored["modes"]
        if {"legacy_terminal", "tail_legacy_selection", "tail_guard_selection"} <= set(modes):
            review["score_summary"] = [row for row in review["score_summary"] if row["method"] != method]
            review["score_summary"].append(
                {
                    "method": method,
                    "legacy_joint": "|".join(modes["legacy_terminal"]["joint"]),
                    "tail_joint": "|".join(modes["tail_legacy_selection"]["joint"]),
                    "guard_joint": "|".join(modes["tail_guard_selection"]["joint"]),
                    "legacy_pick": "|".join(modes["legacy_terminal"]["pick"]),
                    "tail_pick": "|".join(modes["tail_legacy_selection"]["pick"]),
                }
            )
    review["score_errors"] = score_errors
    review["initial_substitution"] = True
    summary["review"] = review
    summary["diagnostic_rollouts"] = int(stats.get("rollouts", 0))
    _write_csv(out_dir / "snapshot_scores.csv", score_rows)
    (out_dir / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    if score_errors or stats.get("budget_exhausted"):
        summary["stop_reason"] = "snapshot_reproduction_failed"
        summary["elapsed_s"] = time.perf_counter() - started
        _emit(out_dir, summary)
        return summary

    print("pilot", flush=True)
    mission_stats = _stats()
    mission_stats["rollout_cap"] = int(cfg["budget"]["mission_rollouts"])
    episodes = []
    decisions = []
    deliveries = []
    mappings = []
    progress = out_dir / "completed_missions.jsonl"
    done, episodes, decisions, deliveries = _load_progress(progress)
    by_id = {row["id"]: row for row in manifest["scenarios"]}

    def _one(method: str, scenario_row: dict, stats_dict: dict) -> bool:
        key = f"core|{scenario_row['id']}|{method}"
        if key in done:
            print(f"resume skip {key}", flush=True)
            return True
        if int(stats_dict.get("rollouts", 0)) >= int(stats_dict["rollout_cap"]) or stats_dict.get("budget_exhausted"):
            stats_dict["budget_exhausted"] = True
            return False
        print(f"run {scenario_row['id']} {method} rollouts={stats_dict.get('rollouts', 0)}", flush=True)
        world = make_world(candidate, scenario_row["battery_j"], f"{scenario_row['id']}-{method}")
        beliefs = [prior_belief(candidate), prior_belief(candidate)]
        try:
            episode, decision_rows, delivery_rows, mapping_rows = _play_guarded(
                method,
                scenario_row,
                candidate,
                factor,
                stats_dict,
                world,
                beliefs,
                0,
                m15_decisions[method] if scenario_row["id"] == "b3_loose" else {},
            )
        except BudgetExceeded:
            stats_dict["budget_exhausted"] = True
            summary["stop_reason"] = f"mission_cap_during_{scenario_row['id']}_{method}"
            return False
        if scenario_row["id"] != "b3_loose":
            episode["prefix_match_before_180s"] = ""
        episodes.append(episode)
        decisions.extend(decision_rows)
        deliveries.extend(delivery_rows)
        mappings.extend(mapping_rows)
        done.add(key)
        _append_progress(progress, key, episode, decision_rows, delivery_rows)
        (out_dir / "mission_stats.json").write_text(json.dumps(mission_stats), encoding="utf-8")
        return True

    # Pilot divergence uses M1.5 b3_loose only. Other scenarios get an empty legacy map and no prefix claim.
    for method in METHODS:
        if not _one(method, scenario, mission_stats):
            break
    summary["missions_ran"] = any(row["scenario_id"] == "b3_loose" for row in episodes)
    pilot_off = next((row for row in episodes if row["scenario_id"] == "b3_loose" and row["method"] == OFF), None)
    pilot_on = next((row for row in episodes if row["scenario_id"] == "b3_loose" and row["method"] == ON), None)
    keeps = False
    if pilot_off is not None:
        keeps = (not _pair("new_OFF-m13_OFF", pilot_off, m13_episodes[("b3_loose", OFF)])["regressed"]) and (
            not _pair("new_OFF-m15_OFF", pilot_off, m15_episodes[("b3_loose", OFF)])["regressed"]
        )
    summary["pilot_off_keeps"] = keeps
    prefix_ok = bool(pilot_off and pilot_off.get("prefix_match_before_180s") and pilot_on and pilot_on.get("prefix_match_before_180s"))
    summary["pilot_note"] = (
        "两条先行航次在 180 秒前与冻结前缀一致，它们同时作为同状态反事实续跑。"
        if prefix_ok
        else "先行航次在 180 秒前已经分歧。完整航次差值不能单独归因于 180 秒那一次分配。"
    )
    continuation_stats = _stats()
    continuation_stats["rollout_cap"] = int(cfg["budget"]["continuation_rollouts"])
    if summary["missions_ran"] and not prefix_ok and not mission_stats.get("budget_exhausted"):
        print("continuation", flush=True)
        for method in METHODS:
            if int(continuation_stats.get("rollouts", 0)) >= int(continuation_stats["rollout_cap"]):
                continuation_stats["budget_exhausted"] = True
                summary["stop_reason"] = "continuation_cap"
                break
            world, beliefs, _events, _final = _replay(candidate, scenario, method, m15_decisions[method], range(6), factor)
            try:
                episode, decision_rows, delivery_rows, mapping_rows = _play_guarded(
                    method, scenario, candidate, factor, continuation_stats, world, beliefs, 6, m15_decisions[method]
                )
            except BudgetExceeded:
                continuation_stats["budget_exhausted"] = True
                summary["stop_reason"] = "continuation_cap"
                break
            episode["branch"] = "counterfactual"
            episode["prefix_match_before_180s"] = True
            episodes.append(episode)
            decisions.extend(decision_rows)
            deliveries.extend(delivery_rows)
            mappings.extend(mapping_rows)
    if keeps and not mission_stats.get("budget_exhausted"):
        print("remaining", flush=True)
        for scenario_id in cfg["scenarios"]:
            if scenario_id == "b3_loose":
                continue
            for method in METHODS:
                if not _one(method, by_id[scenario_id], mission_stats):
                    break
            if mission_stats.get("budget_exhausted") or summary.get("stop_reason", "").startswith("mission_cap"):
                break
        summary["rest_ran"] = all(f"core|{scenario_id}|{method}" in done for scenario_id in cfg["scenarios"] if scenario_id != "b3_loose" for method in METHODS)
    elif pilot_off is not None and pilot_on is not None and not keeps and not summary.get("stop_reason"):
        summary["stop_reason"] = "pilot_off_regressed"
    core_episodes = [row for row in episodes if row.get("branch", "core") == "core"]
    comparisons = []
    for row in core_episodes:
        scenario_id = row["scenario_id"]
        method = row["method"]
        if method == OFF:
            comparisons.append(_pair("new_OFF-m15_OFF", row, m15_episodes[(scenario_id, OFF)]))
            comparisons.append(_pair("new_OFF-m13_OFF", row, m13_episodes[(scenario_id, OFF)]))
        else:
            comparisons.append(_pair("new_ON-m15_ON", row, m15_episodes[(scenario_id, ON)]))
            comparisons.append(_pair("new_ON-m13_ON", row, m13_episodes[(scenario_id, ON)]))
    by_scenario = {}
    for row in core_episodes:
        by_scenario.setdefault(row["scenario_id"], {})[row["method"]] = row
    for scenario_id, pair in by_scenario.items():
        if OFF in pair and ON in pair:
            comparisons.append(_pair("new_ON-new_OFF", pair[ON], pair[OFF]))
    complete = all(f"core|{scenario_id}|{method}" in done for scenario_id in cfg["scenarios"] for method in METHODS)
    summary["episodes"] = episodes
    summary["comparisons"] = comparisons
    summary["complete"] = complete
    if complete:
        summary["judgment"] = _judge(comparisons, cfg["scenarios"], True)
        summary["stop_reason"] = ""
    elif summary["missions_ran"]:
        summary["judgment"] = {
            "passive_supported": False,
            "on_no_regression": False,
            "off_utility_improved": False,
            "conclusion": "b3_loose 的主要 OFF 未保持交付量或净效用，本轮结束" if summary.get("stop_reason") == "pilot_off_regressed" else "本轮在进入全部 24 条之前结束",
            "default_selection": LEGACY_SELECTION,
            "default_terminal": "legacy",
        }
    else:
        summary["judgment"] = {
            "passive_supported": False,
            "conclusion": summary.get("stop_reason") or "未运行主试",
            "default_selection": LEGACY_SELECTION,
            "default_terminal": "legacy",
        }
    _write_csv(out_dir / "episodes.csv", core_episodes)
    _write_csv(out_dir / "counterfactual_episodes.csv", [row for row in episodes if row.get("branch") == "counterfactual"])
    _write_csv(out_dir / "decisions.csv", decisions)
    _write_csv(out_dir / "deliveries.csv", deliveries)
    _write_csv(out_dir / "candidate_mapping.csv", mappings)
    _write_csv(out_dir / "comparisons.csv", comparisons)
    summary["mission_rollouts"] = int(mission_stats.get("rollouts", 0))
    summary["continuation_rollouts"] = int(continuation_stats.get("rollouts", 0))
    summary["selection_aliases"] = int(mission_stats.get("selection_aliases", 0)) + int(continuation_stats.get("selection_aliases", 0)) + int(stats.get("selection_aliases", 0))
    summary["selection_kept"] = int(mission_stats.get("selection_kept", 0)) + int(stats.get("selection_kept", 0))
    summary["admission_quotes"] = quote_stats().get("admission_quotes")
    summary["admission_hits"] = quote_stats().get("admission_cache_hits")
    summary["tail_simulated_seconds"] = tail_stats().get("return_simulated_seconds")
    summary["diagnostic_stats"] = _public(stats)
    summary["mission_stats"] = _public(mission_stats)
    summary["continuation_stats"] = _public(continuation_stats)
    summary["elapsed_s"] = time.perf_counter() - started
    _emit(out_dir, summary)
    return summary


def _emit(out_dir: Path, summary: dict) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ROOT / "docs" / "research" / "Active_Model_Identification_M16_Report_CN.md"
    write_report(report, summary)
    manifest_path = out_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["tests"] = summary.get("tests")
    manifest["stop_reason"] = summary.get("stop_reason")
    manifest["conclusion"] = summary.get("judgment", {}).get("conclusion")
    manifest["elapsed_s"] = summary.get("elapsed_s")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/active_model_identification_m16/m16_20261114")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_dir = ROOT / args.out
    if (out_dir / "summary.json").exists() and not args.force:
        raise SystemExit(f"{out_dir} already has summary.json")
    summary = run(out_dir)
    print(
        json.dumps(
            {
                "tests": summary.get("tests"),
                "stop": summary.get("stop_reason"),
                "conclusion": summary.get("judgment", {}).get("conclusion"),
                "pilot_off_keeps": summary.get("pilot_off_keeps"),
                "complete": summary.get("complete"),
                "elapsed_s": summary.get("elapsed_s"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
