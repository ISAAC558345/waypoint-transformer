"""Attribute the 240 s backhaul that spent more energy than returning.

The two archived message arms are replayed first. Quotes at that instant toggle
only whether UAV 1 may use the wind message. The two first actions that the
archive did not fly are the only new closed-loop runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

from wt.active_model_identification.policies import select_lookahead, tie_abs
from wt.active_model_identification.run_w1 import ENERGY_TOL, _feasible, _label
from wt.active_model_identification.run_w2 import _read_csv
from wt.active_model_identification.run_w4 import (
    _choose,
    _condition,
    _key,
    assess,
    baseline,
    build_link,
    load_w4,
    play_message,
    quote_candidate,
)
from wt.active_model_identification.spec import ROOT
from wt.active_model_identification.world import flight_state

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w41.json"
W4_DIR = ROOT / "results" / "wind_observation_w4" / "w4_20261115"
RUN_ID = "w41_20261115"
DECISION_S = 240.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> str:
    try:
        return subprocess.check_output(["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def load_w41() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if int(cfg["decision_s"]) != 240 or int(cfg["focus_uav"]) != 1:
        raise SystemExit("W4.1 registration moved the decision or the focus UAV")
    return cfg


def _uav_state(uav) -> dict:
    return {
        "x_m": float(uav.x),
        "y_m": float(uav.y),
        "vx_m_s": float(uav.vx),
        "vy_m_s": float(uav.vy),
        "speed_m_s": float(math.hypot(uav.vx, uav.vy)),
        "energy_j": float(uav.energy_j),
        "airborne": bool(uav.airborne),
        "flight_state": flight_state(uav),
        "land_timer_s": float(uav.land_timer_s),
    }


def _packets(world) -> list[dict]:
    rows = []
    for packet in world.packets:
        buoy, number = _key(packet.id)
        rows.append(
            {
                "packet": f"b{buoy}-p{number}",
                "buoy": buoy,
                "holder": packet.holder,
                "delivered_s": None if packet.delivered_s is None else float(packet.delivered_s),
                "remaining_bits": float(packet.remaining_bits),
            }
        )
    return rows


def _beliefs(beliefs, spec: dict) -> list[dict]:
    probability = float(spec["belief"]["guard_probability"])
    return [{"mean": float(item.mean()), "q95": float(item.quantile(probability))} for item in beliefs]


def _at_decision(trace: list[dict]) -> dict:
    return next(item for item in trace if abs(float(item["t_s"]) - DECISION_S) < 1e-9)


def _match_archive(rows: list[dict], archived: list[dict], mode: str) -> None:
    saved = [row for row in archived if row["mode"] == mode]
    if len(saved) != len(rows):
        raise SystemExit(f"{mode} replay has {len(rows)} rows, archive has {len(saved)}")
    for found, expected in zip(rows, saved):
        for key in ("epoch", "uav", "requested", "executed", "reason"):
            if str(found[key]) != str(expected[key]):
                raise SystemExit(f"{mode} e{found['epoch']} u{found['uav']} {key} {found[key]} != {expected[key]}")
        if abs(float(found["energy_j"]) - float(expected["energy_j"])) > 1e-6:
            raise SystemExit(f"{mode} e{found['epoch']} u{found['uav']} energy drifted")


def _score_table(case: dict, world, beliefs, uav: int, message, scheduler_action) -> dict:
    spec = case["spec"]
    reserve = float(spec["energy"]["reserve_j"])
    block = spec["utility"]
    tau = tie_abs(spec)
    scored = []
    raw = []
    for action in _feasible(spec, world, uav):
        risk = quote_candidate(case, world, beliefs, uav, action, message, "q95")
        mean = quote_candidate(case, world, beliefs, uav, action, message, "mean")
        needs_bits = action[0] in ("service", "backhaul")
        admitted = bool(
            risk["landed"]
            and risk["on_time"]
            and risk["terminal_j"] + 1e-6 >= reserve
            and (not needs_bits or risk["bits"] > 0.0)
        )
        utility = -1.0e6
        if mean["landed"]:
            utility = mean["bits"] / float(block["b_ref_bits"]) - float(block["lambda_e"]) * mean["energy_j"] / float(block["e_ref_j"])
        target = [item["delivered_s"] for item in mean["deliveries"] if int(item["buoy"]) == int(load_w41()["target_buoy"])]
        scored.append(
            {
                "action": _label(action),
                "admitted": admitted,
                "utility": float(utility),
                "bits": float(mean["bits"]),
                "quote_energy_j": float(mean["energy_j"]),
                "terminal_j": float(risk["terminal_j"]),
                "holds_reserve": bool(risk["terminal_j"] + 1e-6 >= reserve),
                "predicted_arrival_s": target,
                "arrival_used_in_score": False,
                "bits_used_in_score": True,
                "energy_used_in_score": True,
                "terminal_used_in_admission": True,
            }
        )
        raw.append({"action": action, "admitted": admitted, "utility": utility})
    chosen, reason = _choose(raw, scheduler_action, tau)
    best = max((row["utility"] for row in scored if row["admitted"]), default=None)
    tied = []
    if best is not None:
        tied = [row["action"] for row in scored if row["admitted"] and best - row["utility"] <= tau + 1e-15]
    return {"rows": scored, "chosen": _label(chosen), "reason": reason, "tied": tied, "scheduler": _label(scheduler_action)}


def _intervals(trace: list[dict], rows: list[dict]) -> list[dict]:
    found = []
    previous_end = None
    for state in trace:
        ends = {int(row["uav"]): row for row in rows if int(row["epoch"]) == int(state["epoch"])}
        for uav, row in ends.items():
            start = float(state["energy_j"][uav])
            end = float(row["energy_j"])
            gap = 0.0 if previous_end is None else float(previous_end[uav]) - start
            found.append(
                {
                    "epoch": int(state["epoch"]),
                    "t_s": float(state["t_s"]),
                    "uav": uav,
                    "executed": row["executed"],
                    "start_j": start,
                    "end_j": end,
                    "gap_before_j": gap,
                    "spent_j": start - end,
                }
            )
        previous_end = [float(ends[uav]["energy_j"]) for uav in range(len(state["energy_j"]))]
    return found


def _deliveries(world, buoy: int) -> list[dict]:
    rows = []
    for item in world.deliveries:
        if int(item["buoy"]) != buoy:
            continue
        rows.append({"id": item["id"], "uav": int(item["uav"]), "delivered_s": float(item["delivered_s"]), "bits": float(item["bits"])})
    rows.sort(key=lambda item: item["delivered_s"])
    return rows


def _cell(case: dict, condition: dict, link: dict, snapshot: dict, mode: str, action) -> dict:
    origin = (
        snapshot["world"].snapshot(),
        [item.copy() for item in snapshot["beliefs"]],
        int(snapshot["epoch"]),
        {"sender_j_drawn": 0.0, "receiver_j_drawn": 0.0},
    )
    preset = (int(snapshot["epoch"]), 1, action)
    result = play_message(case, condition, mode, link, origin=origin, preset=preset)
    outcome = assess(result["world"], case["spec"], set(), 3, 0.0)
    outcome.update(
        {
            "mode": mode,
            "preset": _label(action),
            "executed": next(row["executed"] for row in result["rows"] if int(row["uav"]) == 1),
            "partner_executed": next(row["executed"] for row in result["rows"] if int(row["uav"]) == 0),
            "deliveries": _deliveries(result["world"], 3),
            "message_j_in_snapshot": True,
        }
    )
    return outcome


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        flat = []
        for row in rows:
            copied = {}
            for key, value in row.items():
                copied[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            flat.append(copied)
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    out = out_dir or (ROOT / "results" / "wind_observation_w41" / RUN_ID)
    if "wind_observation_w41" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w41")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_w41()
    w4, case = load_w4()
    condition = _condition(case, "headwind_return")
    played = baseline(case, condition)
    link = build_link(played, case, condition, w4)
    archived_rows = _read_csv(W4_DIR / "online.csv")
    archived_arms = _read_csv(W4_DIR / "arms.csv")
    traces = {}
    replays = {}
    for mode in ("no_send", "timely"):
        print("replay", mode, flush=True)
        trace: list[dict] = []
        replays[mode] = play_message(case, condition, mode, link, trace=trace)
        traces[mode] = trace
        _match_archive(replays[mode]["rows"], archived_rows, mode)
        saved = next(row for row in archived_arms if row["mode"] == mode)
        outcome = assess(replays[mode]["world"], case["spec"], set(), 3, 0.0)
        if abs(outcome["on_time_bits"] - float(saved["on_time_bits"])) > 1e-6 or abs(outcome["energy_j"] - float(saved["energy_j"])) > 1e-4:
            raise SystemExit(f"{mode} endpoint does not match the W4 arm")
    focus = int(cfg["focus_uav"])
    states = {mode: _at_decision(traces[mode]) for mode in ("no_send", "timely")}
    state_rows = []
    for mode, state in states.items():
        previous = next(row for row in replays[mode]["rows"] if abs(float(row["t_s"]) - 210.0) < 1e-9 and int(row["uav"]) == focus)
        decision = next(row for row in replays[mode]["rows"] if abs(float(row["t_s"]) - DECISION_S) < 1e-9 and int(row["uav"]) == focus)
        partner = next(row for row in replays[mode]["rows"] if abs(float(row["t_s"]) - DECISION_S) < 1e-9 and int(row["uav"]) == 0)
        state_rows.append(
            {
                "mode": mode,
                "uavs": [_uav_state(uav) for uav in state["world"].uavs],
                "packets": _packets(state["world"]),
                "beliefs": _beliefs(state["beliefs"], case["spec"]),
                "used_message": state["info"] is not None,
                "previous_executed": previous["executed"],
                "requested": decision["requested"],
                "executed": decision["executed"],
                "reason": decision["reason"],
                "partner_requested": partner["requested"],
                "guard_rewrote": decision["requested"] != decision["executed"],
            }
        )
    tables = {}
    for mode, state in states.items():
        scheduler, _tables = select_lookahead(case["cfg"]["method"], state["world"], state["beliefs"], case["spec"], case["factor"], None)
        natural = state["info"]
        table = _score_table(case, state["world"], state["beliefs"], focus, natural, scheduler[focus])
        saved = next(row for row in state_rows if row["mode"] == mode)
        if table["chosen"] != saved["requested"] or table["reason"] != saved["reason"]:
            raise SystemExit(f"{mode} quote chooses {table['chosen']} ({table['reason']}), archive requested {saved['requested']} ({saved['reason']})")
        tables[f"{mode}_natural"] = table
    timely_state = states["timely"]
    scheduler, _tables = select_lookahead(
        case["cfg"]["method"], timely_state["world"], timely_state["beliefs"], case["spec"], case["factor"], None
    )
    toggled = {
        "message": _score_table(case, timely_state["world"], timely_state["beliefs"], focus, link["message"], scheduler[focus]),
        "none": _score_table(case, timely_state["world"], timely_state["beliefs"], focus, None, scheduler[focus]),
    }
    print("cross", flush=True)
    cells = []
    presets = {"return": ("return", None), "backhaul": ("backhaul", None)}
    for mode, name in (("timely", "backhaul"), ("timely", "return"), ("no_send", "backhaul"), ("no_send", "return")):
        cell = _cell(case, condition, link, timely_state, mode, presets[name])
        cell["information"] = "message" if mode == "timely" else "none"
        cells.append(cell)
        print(cell["information"], cell["preset"], cell["executed"], cell["on_time_bits"], cell["energy_j"], flush=True)
    reproduced = next(cell for cell in cells if cell["information"] == "message" and cell["preset"] == "backhaul:None")
    timely_saved = next(row for row in archived_arms if row["mode"] == "timely")
    if abs(reproduced["on_time_bits"] - float(timely_saved["on_time_bits"])) > 1e-6 or abs(reproduced["energy_j"] - float(timely_saved["energy_j"])) > 1e-3:
        raise SystemExit("forcing the archived backhaul does not reproduce the timely endpoint")
    intervals = {mode: _intervals(traces[mode], replays[mode]["rows"]) for mode in ("no_send", "timely")}
    paired = []
    nosend = {(row["epoch"], row["uav"]): row for row in intervals["no_send"]}
    for row in intervals["timely"]:
        other = nosend[(row["epoch"], row["uav"])]
        paired.append(
            {
                "epoch": row["epoch"],
                "t_s": row["t_s"],
                "uav": row["uav"],
                "timely_executed": row["executed"],
                "no_send_executed": other["executed"],
                "spent_delta_j": row["spent_j"] - other["spent_j"],
                "gap_delta_j": row["gap_before_j"] - other["gap_before_j"],
            }
        )
    explained = sum(row["spent_delta_j"] + row["gap_delta_j"] for row in paired)
    archived_gap = float(next(row["energy_j"] for row in archived_arms if row["mode"] == "timely")) - float(
        next(row["energy_j"] for row in archived_arms if row["mode"] == "no_send")
    )
    message_gap = sum(row["gap_delta_j"] for row in paired)
    quote_order = sorted(
        (row for row in toggled["message"]["rows"] if row["action"] in ("return:None", "backhaul:None") and row["admitted"]),
        key=lambda row: row["utility"],
        reverse=True,
    )
    realized = { (cell["information"], cell["preset"]): cell for cell in cells }
    return_cell = realized[("message", "return:None")]
    backhaul_cell = realized[("message", "backhaul:None")]
    ranking_opposite = bool(quote_order and quote_order[0]["action"] == "backhaul:None" and return_cell["energy_j"] + 1e-6 < backhaul_cell["energy_j"] and abs(return_cell["on_time_bits"] - backhaul_cell["on_time_bits"]) <= 1e-6 and return_cell["reserve_met"] and backhaul_cell["reserve_met"])
    if ranking_opposite:
        proposal = "返航脚本把仍留在机上的缓存记成 0 bit，回传脚本把这批 bit 计入分数。同一信息下两种首动作事后都把 3 号浮标送到岸上，排序因此和实际能耗相反。后续只应让返航报价计入闭环还会交付的缓存，不改备用阈值、风权或报文时限。"
    else:
        proposal = ""
    summary = {
        "schema_version": 1,
        "parent_commit": _git(),
        "config_sha256": _sha256(CONFIG_PATH),
        "elapsed_s": time.perf_counter() - started,
        "archive_match": True,
        "states": state_rows,
        "tables": tables,
        "toggled": toggled,
        "cells": [
            {key: cell[key] for key in ("information", "preset", "executed", "partner_executed", "on_time_bits", "energy_j", "utility", "uav_energy_j", "returned", "below_reserve", "not_landed", "deliveries", "reserve_met")}
            for cell in cells
        ],
        "ledger_gap_j": explained,
        "archived_gap_j": archived_gap,
        "unexplained_j": archived_gap - explained,
        "message_gap_j": message_gap,
        "ranking_opposite": ranking_opposite,
        "proposal": proposal,
        "default_changed": False,
        "scheduler_reference_j": float(json.loads((W4_DIR / "summary.json").read_text(encoding="utf-8"))["control"]["energy_j"]),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out / "quotes.csv", [{**row, "information": name} for name, table in toggled.items() for row in table["rows"]])
    _write_csv(out / "cells.csv", summary["cells"])
    _write_csv(out / "ledger.csv", paired)
    print(f"gap {archived_gap:.6f} explained {explained:.6f} opposite {ranking_opposite}", flush=True)
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
