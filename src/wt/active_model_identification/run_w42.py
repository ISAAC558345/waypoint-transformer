"""Separate a shared shore link from later reselection at the 240 s backhaul.

archive actions: a request, an execution, or an energy in the timely replay
differs from the W4 online log.

archive arrival: buoy 3's two packets do not arrive at the W4.1 times.

solo quote: the partner-waits script no longer reproduces the archived
253.645 s and 261.645 s arrivals.

The partner's request for this one interval is already computed before the
step. Later requests stay in the archived log. Arrival time does not enter
the score.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

from wt.active_model_identification import world as world_mod
from wt.active_model_identification.policies import select_lookahead, tie_abs
from wt.active_model_identification.run_w1 import _feasible, _label, _parse
from wt.active_model_identification.run_w2 import _read_csv
from wt.active_model_identification.run_w4 import (
    _choose,
    _condition,
    _key,
    assess,
    build_link,
    load_w4,
    play_message,
    quote_candidate,
)
from wt.active_model_identification.scenarios import observation_noise
from wt.active_model_identification.spec import ROOT
from wt.active_model_identification.wind import _keep, mean_wind_fn, script_action
from wt.active_model_identification.world import advance, pad_distance_m, pad_radius_m, tx_seconds

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w42.json"
W4_DIR = ROOT / "results" / "wind_observation_w4" / "w4_20261115"
RUN_ID = "w42_20261115"
DECISION_S = 240.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> str:
    try:
        return subprocess.check_output(["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def load_w42() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if int(cfg["decision_s"]) != 240 or int(cfg["focus_uav"]) != 1 or int(cfg["partner_uav"]) != 0:
        raise SystemExit("W4.2 registration moved the decision or the UAV roles")
    return cfg


def _listen(record: list[dict]):
    original_lora = world_mod._try_lora
    original_backhaul = world_mod._backhaul
    state = {"kind": "control", "uav": None}

    def lora(world, step_end, payload_bytes):
        airtime = original_lora(world, step_end, payload_bytes)
        if airtime is not None:
            record.append({"t_s": float(world.t), "kind": state["kind"], "uav": state["uav"], "airtime_s": float(airtime)})
        return airtime

    def backhaul(world, uav, spec, step_end):
        previous = (state["kind"], state["uav"])
        state["kind"] = "backhaul"
        state["uav"] = int(uav.index)
        try:
            return original_backhaul(world, uav, spec, step_end)
        finally:
            state["kind"], state["uav"] = previous

    world_mod._try_lora = lora
    world_mod._backhaul = backhaul

    def restore() -> None:
        world_mod._try_lora = original_lora
        world_mod._backhaul = original_backhaul

    return restore


def _physical_grants(case: dict, condition: dict, state: dict, rows: list[dict]) -> list[dict]:
    """Replay archived requests only. Quotes inside the full mission are not link time."""
    from wt.active_model_identification.run_w4 import _truth

    spec = case["spec"]
    world = state["world"].snapshot()
    beliefs = [item.copy() for item in state["beliefs"]]
    noise = observation_noise(int(case["scenario"]["seed"]), float(spec["belief"]["sigma_j"]))
    wind_at = _truth(condition, case["cfg"]["field"])
    grants: list[dict] = []
    restore = _listen(grants)
    try:
        epoch0 = int(state["epoch"])
        for epoch in range(epoch0, epoch0 + 4):
            actions = []
            for uav in range(2):
                label = next(row["requested"] for row in rows if int(row["epoch"]) == epoch and int(row["uav"]) == uav)
                actions.append(_parse(label))
            observations = advance(
                world,
                actions,
                list(case["scenario"]["theta"]),
                beliefs,
                spec,
                case["factor"],
                None,
                planning=False,
                interval_index=epoch,
                noise_j=noise,
                wind_at=wind_at,
            )
            for obs in observations:
                saved = next(row for row in rows if int(row["epoch"]) == epoch and int(row["uav"]) == int(obs["uav"]))
                if _label(tuple(obs["executed"])) != saved["executed"]:
                    raise SystemExit("airtime replay executed a different action")
                if abs(float(obs["energy_j"]) - float(saved["energy_j"])) > 1e-6:
                    raise SystemExit("airtime replay drifted from the archive")
    finally:
        restore()
    return grants


def _packet_name(packet_id: str) -> str:
    buoy, number = _key(packet_id)
    return f"b{buoy}-p{number}"


def _queue(world) -> list[dict]:
    rows = []
    for packet in world.packets:
        rows.append(
            {
                "packet": _packet_name(packet.id),
                "buoy": int(packet.buoy),
                "holder": packet.holder,
                "remaining_bits": float(packet.remaining_bits),
                "delivered_s": None if packet.delivered_s is None else float(packet.delivered_s),
            }
        )
    return rows


def _queue_air_s(world, uav: int, spec: dict) -> float:
    holder = f"uav:{uav}"
    return float(sum(tx_seconds(packet.remaining_bits, spec) for packet in world.packets if packet.holder == holder and packet.remaining_bits > 0.0))


def _sent_bits(before, after) -> dict[int, float]:
    later = {packet.id: packet for packet in after.packets}
    sent = {0: 0.0, 1: 0.0}
    for packet in before.packets:
        if not str(packet.holder).startswith("uav:"):
            continue
        uav = int(str(packet.holder).split(":")[1])
        sent[uav] += float(packet.remaining_bits) - float(later[packet.id].remaining_bits)
    return sent


def _match_archive(rows: list[dict], archived: list[dict]) -> None:
    saved = [row for row in archived if row["mode"] == "timely"]
    if len(saved) != len(rows):
        raise SystemExit(f"timely replay has {len(rows)} rows, archive has {len(saved)}")
    for found, expected in zip(rows, saved):
        for key in ("epoch", "uav", "requested", "executed"):
            if str(found[key]) != str(expected[key]):
                raise SystemExit(f"e{found['epoch']} u{found['uav']} {key} {found[key]} != {expected[key]}")
        if abs(float(found["energy_j"]) - float(expected["energy_j"])) > 1e-6:
            raise SystemExit(f"e{found['epoch']} u{found['uav']} energy drifted")


def _target_times(world, buoy: int, start_s: float) -> list[dict]:
    rows = []
    for item in world.deliveries:
        if int(item["buoy"]) != buoy or float(item["delivered_s"]) <= start_s + 1e-9:
            continue
        rows.append(
            {
                "packet": _packet_name(item["id"]),
                "uav": int(item["uav"]),
                "delivered_s": float(item["delivered_s"]),
                "bits": float(item["bits"]),
                "on_time": bool(item["on_time"]),
            }
        )
    rows.sort(key=lambda item: (item["delivered_s"], item["packet"]))
    return rows


def _roll(case: dict, world, beliefs, message, focus: int, partner_mode: str, action, theta_mode: str, *, one_interval: bool) -> dict:
    spec = case["spec"]
    thetas = [item.mean() for item in beliefs]
    if theta_mode == "q95":
        thetas[focus] = beliefs[focus].quantile(float(spec["belief"]["guard_probability"]))
    elif theta_mode != "mean":
        raise ValueError(theta_mode)
    wind_at = mean_wind_fn(message, case["cfg"]["field"], float(case["cfg"]["prior_std_m_s"]))
    sim = world.snapshot()
    local = [item.copy() for item in beliefs]
    consumed_before = [uav.consumed_j for uav in sim.uavs]
    start_t = float(sim.t)
    decision = float(spec["time"]["decision_s"])
    mission = float(spec["time"]["mission_s"])
    interval0 = int(round(start_t / decision))
    partner = 1 - focus
    step = 0
    limit = int(mission // decision) + 4
    while step < limit and sim.t < mission - 1e-9:
        if one_interval and step > 0:
            break
        if one_interval:
            focus_action = action
            partner_action = ("wait", None) if partner_mode == "wait" else ("backhaul", None)
        else:
            focus_action = script_action(sim, focus, action, spec, step)
            if focus_action is None:
                break
            if partner_mode == "wait":
                partner_action = ("wait", None)
            else:
                partner_action = script_action(sim, partner, ("backhaul", None), spec, step) or ("wait", None)
        actions = [("wait", None), ("wait", None)]
        actions[focus] = focus_action
        actions[partner] = partner_action
        advance(
            sim,
            actions,
            list(thetas),
            local,
            spec,
            case["factor"],
            None,
            planning=False,
            interval_index=interval0 + step,
            noise_j=lambda _uav, _interval: 0.0,
            wind_at=wind_at,
            admit=_keep,
            return_guard=_keep,
        )
        step += 1
    landed = (not sim.uavs[focus].airborne) and pad_distance_m(sim.uavs[focus]) <= pad_radius_m(spec)
    failed = bool(sim.uavs[focus].failed)
    idle_s = max(0.0, mission - float(sim.t)) if landed and not one_interval else 0.0
    idle_j = float(spec["energy"]["ground_idle_w"]) * idle_s
    deliveries = []
    for item in sim.deliveries:
        if float(item["delivered_s"]) <= start_t + 1e-9:
            continue
        deliveries.append(
            {
                "packet": _packet_name(item["id"]),
                "buoy": int(item["buoy"]),
                "uav": int(item["uav"]),
                "delivered_s": float(item["delivered_s"]),
                "bits": float(item["bits"]),
                "on_time": bool(item["on_time"]),
            }
        )
    deliveries.sort(key=lambda item: (item["delivered_s"], item["packet"]))
    focus_bits = sum(item["bits"] for item in deliveries if item["uav"] == focus and item["on_time"])
    return {
        "bits": float(focus_bits),
        "energy_j": float(sim.uavs[focus].consumed_j - consumed_before[focus]) + idle_j,
        "spent_j": [float(sim.uavs[index].consumed_j - consumed_before[index]) for index in range(2)],
        "terminal_j": float(sim.uavs[focus].energy_j) - idle_j,
        "landed": bool(landed) and not failed,
        "on_time": sim.t <= mission + 1e-9,
        "deliveries": deliveries,
        "queue": _queue(sim),
        "t_end_s": float(sim.t),
    }


def _score_pair(spec: dict, action, mean: dict, risk: dict) -> dict:
    reserve = float(spec["energy"]["reserve_j"])
    block = spec["utility"]
    needs_bits = action[0] in ("service", "backhaul")
    admitted = bool(risk["landed"] and risk["on_time"] and risk["terminal_j"] + 1e-6 >= reserve and (not needs_bits or mean["bits"] > 0.0))
    utility = -1.0e6
    if mean["landed"]:
        utility = mean["bits"] / float(block["b_ref_bits"]) - float(block["lambda_e"]) * mean["energy_j"] / float(block["e_ref_j"])
    arrivals = [item["delivered_s"] for item in mean["deliveries"] if int(item["buoy"]) == 3 and item["uav"] == 1]
    return {
        "action": _label(action),
        "admitted": admitted,
        "utility": float(utility),
        "bits": float(mean["bits"]),
        "quote_energy_j": float(mean["energy_j"]),
        "terminal_j": float(risk["terminal_j"]),
        "predicted_arrival_s": arrivals,
        "arrival_used_in_score": False,
    }


def _from_quote(spec: dict, quoted_mean: dict, quoted_risk: dict, action) -> tuple[dict, dict]:
    mean = {
        "bits": float(quoted_mean["bits"]),
        "energy_j": float(quoted_mean["energy_j"]),
        "terminal_j": float(quoted_mean["terminal_j"]),
        "landed": bool(quoted_mean["landed"]),
        "on_time": bool(quoted_mean["on_time"]),
        "deliveries": [
            {"packet": item["packet"], "buoy": int(item["buoy"]), "uav": 1, "delivered_s": float(item["delivered_s"]), "bits": None, "on_time": True}
            for item in quoted_mean["deliveries"]
        ],
    }
    risk = {
        "bits": float(quoted_risk["bits"]),
        "energy_j": float(quoted_risk["energy_j"]),
        "terminal_j": float(quoted_risk["terminal_j"]),
        "landed": bool(quoted_risk["landed"]),
        "on_time": bool(quoted_risk["on_time"]),
        "deliveries": [],
    }
    return mean, _score_pair(spec, action, mean, risk)


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


def _times(rows: list[dict], packet: str) -> float | None:
    found = [item["delivered_s"] for item in rows if item["packet"] == packet]
    return None if not found else float(found[0])


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    out = out_dir or (ROOT / "results" / "wind_observation_w42" / RUN_ID)
    if "wind_observation_w42" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w42")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_w42()
    w4, case = load_w4()
    condition = _condition(case, "headwind_return")
    parent = json.loads((ROOT / cfg["parent_w41_result"] / "summary.json").read_text(encoding="utf-8"))
    parent_quote = next(row["predicted_arrival_s"] for row in parent["toggled"]["message"]["rows"] if row["action"] == "backhaul:None")
    parent_actual = next(cell["deliveries"] for cell in parent["cells"] if cell["information"] == "message" and cell["preset"] == "backhaul:None")
    if [float(item) for item in parent_quote] != [float(item) for item in cfg["quote_arrival_s"]]:
        raise SystemExit("registered quote arrivals do not match W4.1")
    if [float(item["delivered_s"]) for item in parent_actual] != [float(item) for item in cfg["actual_arrival_s"]]:
        raise SystemExit("registered actual arrivals do not match W4.1")
    focus = int(cfg["focus_uav"])
    spec = case["spec"]
    from wt.active_model_identification.run_w4 import baseline

    played = baseline(case, condition)
    baseline_link = build_link(played, case, condition, w4)
    trace: list[dict] = []
    replay = play_message(case, condition, "timely", baseline_link, trace=trace)
    archived_rows = _read_csv(W4_DIR / "online.csv")
    _match_archive(replay["rows"], archived_rows)
    saved_arm = next(row for row in _read_csv(W4_DIR / "arms.csv") if row["mode"] == "timely")
    outcome = assess(replay["world"], spec, set(), 3, 0.0)
    if abs(outcome["on_time_bits"] - float(saved_arm["on_time_bits"])) > 1e-6 or abs(outcome["energy_j"] - float(saved_arm["energy_j"])) > 1e-4:
        raise SystemExit("timely endpoint does not match the W4 arm")
    actual_targets = _target_times(replay["world"], 3, DECISION_S)
    if [item["delivered_s"] for item in actual_targets] != [float(item) for item in cfg["actual_arrival_s"]]:
        raise SystemExit("buoy 3 arrivals moved off the W4.1 archive")
    by_epoch = {int(item["epoch"]): item for item in trace}
    state = by_epoch[int(DECISION_S // float(spec["time"]["decision_s"]))]
    if state["info"] is None:
        raise SystemExit("the timely snapshot does not yet hold the wind message")
    message = state["info"]
    scheduler, _tables = select_lookahead(case["cfg"]["method"], state["world"], state["beliefs"], spec, case["factor"], None)
    scheduler_action = scheduler[focus]
    solo_rows = []
    solo_choice_rows = []
    for action in _feasible(spec, state["world"], focus):
        quoted_mean = quote_candidate(case, state["world"], state["beliefs"], focus, action, message, "mean")
        quoted_risk = quote_candidate(case, state["world"], state["beliefs"], focus, action, message, "q95")
        mean, scored = _from_quote(spec, quoted_mean, quoted_risk, action)
        if action == ("backhaul", None):
            check = _roll(case, state["world"], state["beliefs"], message, focus, "wait", action, "mean", one_interval=False)
            if [item["delivered_s"] for item in check["deliveries"] if item["buoy"] == 3] != [float(item) for item in cfg["quote_arrival_s"]]:
                raise SystemExit("partner-waits script missed the archived quote arrivals")
            if abs(check["terminal_j"] - quoted_mean["terminal_j"]) > 1e-4 or abs(check["energy_j"] - quoted_mean["energy_j"]) > 1e-4 or abs(check["bits"] - quoted_mean["bits"]) > 1e-6:
                raise SystemExit("partner-waits roller drifted from quote_candidate")
        solo_rows.append(scored)
        solo_choice_rows.append({"action": action, "admitted": scored["admitted"], "utility": scored["utility"]})
    solo_chosen, solo_reason = _choose(solo_choice_rows, scheduler_action, tie_abs(spec))
    archived_choice = next(row for row in replay["rows"] if abs(float(row["t_s"]) - DECISION_S) < 1e-9 and int(row["uav"]) == focus)
    if _label(solo_chosen) != archived_choice["requested"] or solo_reason != archived_choice["reason"]:
        raise SystemExit("solo quote no longer reproduces the archived 240 s choice")
    joint = {}
    window = {}
    for action in _feasible(spec, state["world"], focus):
        joint[_label(action)] = {
            "mean": _roll(case, state["world"], state["beliefs"], message, focus, "backhaul", action, "mean", one_interval=False),
            "q95": _roll(case, state["world"], state["beliefs"], message, focus, "backhaul", action, "q95", one_interval=False),
        }
    for partner_mode in ("wait", "backhaul"):
        window[partner_mode] = _roll(
            case, state["world"], state["beliefs"], message, focus, partner_mode, ("backhaul", None), "mean", one_interval=True
        )
    joint_choice_rows = []
    joint_public = []
    for action in _feasible(spec, state["world"], focus):
        label = _label(action)
        scored = _score_pair(spec, action, joint[label]["mean"], joint[label]["q95"])
        joint_public.append(scored)
        joint_choice_rows.append({"action": action, "admitted": scored["admitted"], "utility": scored["utility"]})
    joint_chosen, joint_reason = _choose(joint_choice_rows, scheduler_action, tie_abs(spec))
    choice_changed = _label(joint_chosen) != _label(solo_chosen)
    closed = None
    if choice_changed:
        origin = (state["world"].snapshot(), [item.copy() for item in state["beliefs"]], int(state["epoch"]), {"sender_j_drawn": 0.0, "receiver_j_drawn": 0.0})
        forced = play_message(case, condition, "timely", baseline_link, origin=origin, preset=(int(state["epoch"]), focus, joint_chosen))
        closed = assess(forced["world"], spec, set(), 3, 0.0)
        closed["deliveries"] = _target_times(forced["world"], 3, DECISION_S)
        closed["requested"] = _label(joint_chosen)
    intervals = []
    decision = float(spec["time"]["decision_s"])
    grants = _physical_grants(case, condition, state, replay["rows"])
    epoch0 = int(DECISION_S // decision)
    for epoch in range(epoch0, epoch0 + 4):
        current = by_epoch[epoch]
        nxt = by_epoch[epoch + 1]
        actions = []
        executed = []
        for uav in range(2):
            row = next(item for item in replay["rows"] if int(item["epoch"]) == epoch and int(item["uav"]) == uav)
            actions.append(row["requested"])
            executed.append(row["executed"])
        t0 = float(current["t_s"])
        t1 = float(nxt["t_s"])
        sent = _sent_bits(current["world"], nxt["world"])
        requested_s = []
        granted_s = []
        for uav in range(2):
            requested_s.append(_queue_air_s(current["world"], uav, spec) if executed[uav] == "backhaul:None" else 0.0)
            granted_s.append(
                sum(item["airtime_s"] for item in grants if item["kind"] == "backhaul" and item["uav"] == uav and t0 - 1e-9 <= item["t_s"] < t1 - 1e-9)
            )
        control_s = sum(item["airtime_s"] for item in grants if item["kind"] == "control" and t0 - 1e-9 <= item["t_s"] < t1 - 1e-9)
        if sum(granted_s) + control_s > decision + 1e-6:
            raise SystemExit(f"radio log at {t0} exceeds the decision interval")
        for uav, label in enumerate(executed):
            if label != "backhaul:None" and granted_s[uav] > 1e-9:
                raise SystemExit(f"non-backhaul interval at {t0} recorded payload airtime")
            remaining = sum(item["remaining_bits"] for item in _queue(nxt["world"]) if item["holder"] == f"uav:{uav}")
            if label == "backhaul:None" and remaining <= 1e-6 and abs(granted_s[uav] - requested_s[uav]) > 1e-6:
                raise SystemExit(f"finished cache at {t0} does not match the granted airtime")
        start_j = [float(uav.energy_j) for uav in current["world"].uavs]
        end_j = [float(next(item["energy_j"] for item in replay["rows"] if int(item["epoch"]) == epoch and int(item["uav"]) == uav)) for uav in range(2)]
        intervals.append(
            {
                "t_s": t0,
                "requested": actions,
                "executed": executed,
                "requested_air_s": requested_s,
                "granted_air_s": granted_s,
                "control_air_s": control_s,
                "bits_sent": [sent[0], sent[1]],
                "spent_j": [start_j[uav] - end_j[uav] for uav in range(2)],
                "queue_after": _queue(nxt["world"]),
            }
        )
    queue_residual_bits = 0.0
    for interval in intervals:
        if interval["executed"][focus] == "backhaul:None":
            continue
        epoch = int(round(float(interval["t_s"]) / decision))
        before = {item["packet"]: item["remaining_bits"] for item in _queue(by_epoch[epoch]["world"]) if item["buoy"] == 3}
        after = {item["packet"]: item["remaining_bits"] for item in interval["queue_after"] if item["buoy"] == 3}
        for packet, bits in before.items():
            queue_residual_bits += abs(after[packet] - bits)
    log_bits = intervals[0]["bits_sent"]
    for uav in range(2):
        predicted = 0.0
        for packet in state["world"].packets:
            if packet.holder != f"uav:{uav}":
                continue
            later = next(item for item in window["backhaul"]["queue"] if item["packet"] == _packet_name(packet.id))
            predicted += float(packet.remaining_bits) - float(later["remaining_bits"])
        if abs(predicted - log_bits[uav]) > 1e-6:
            raise SystemExit(f"joint 30 s bits for UAV {uav} do not match the execution log")
    quote_rows = next(row for row in solo_rows if row["action"] == "backhaul:None")
    splits = []
    for packet, quote_s, actual_s in zip(("b3-p0", "b3-p1"), cfg["quote_arrival_s"], cfg["actual_arrival_s"]):
        joint_s = _times(joint["backhaul:None"]["mean"]["deliveries"], packet)
        if joint_s is None:
            raise SystemExit(f"joint script did not deliver {packet}")
        shared = float(joint_s) - float(quote_s)
        reselection = float(actual_s) - float(joint_s)
        gap = float(actual_s) - float(quote_s)
        splits.append(
            {
                "packet": packet,
                "quote_s": float(quote_s),
                "joint_s": float(joint_s),
                "actual_s": float(actual_s),
                "shared_link_s": shared,
                "reselection_s": reselection,
                "residual_s": gap - shared - reselection,
            }
        )
    summary = {
        "schema_version": 1,
        "parent_commit": _git(),
        "config_sha256": _sha256(CONFIG_PATH),
        "elapsed_s": time.perf_counter() - started,
        "archive_match": True,
        "scheduler_action": _label(scheduler_action),
        "solo_choice": {"action": _label(solo_chosen), "reason": solo_reason},
        "joint_choice": {"action": _label(joint_chosen), "reason": joint_reason},
        "choice_changed": choice_changed,
        "closed_loop": closed,
        "default_changed": False,
        "arrival_used_in_score": False,
        "joint_candidate_computed_before_step": True,
        "later_requests_known_at_decision": False,
        "queue_residual_bits": queue_residual_bits,
        "splits": splits,
        "intervals": intervals,
        "window": {
            mode: {
                "spent_j": item["spent_j"],
                "bits": item["bits"],
                "queue": item["queue"],
                "deliveries": item["deliveries"],
            }
            for mode, item in window.items()
        },
        "log_bits_sent": log_bits,
        "solo_scores": solo_rows,
        "joint_scores": joint_public,
        "quote_backhaul": quote_rows,
        "baseline_energy_j": float(saved_arm["energy_j"]),
        "baseline_utility": float(saved_arm["utility"]),
        "baseline_bits": float(saved_arm["on_time_bits"]),
        "energy_delta_j": None if closed is None else float(closed["energy_j"]) - float(saved_arm["energy_j"]),
        "utility_delta": None if closed is None else float(closed["utility"]) - float(saved_arm["utility"]),
        "recommend_more": bool(
            closed is not None
            and abs(float(closed["on_time_bits"]) - float(saved_arm["on_time_bits"])) <= 1e-6
            and closed["reserve_met"]
            and float(closed["energy_j"]) + 1.0 < float(saved_arm["energy_j"])
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out / "link.csv", intervals)
    packet_rows = []
    for source, rows in (
        ("quote_wait", quote_rows["predicted_arrival_s"] and [{"packet": name, "delivered_s": stamp, "uav": 1} for name, stamp in zip(("b3-p0", "b3-p1"), quote_rows["predicted_arrival_s"])]),
        ("joint_script", [item for item in joint["backhaul:None"]["mean"]["deliveries"]]),
        ("actual", [item for item in replay["world"].deliveries if float(item["delivered_s"]) > DECISION_S + 1e-9]),
    ):
        for item in rows:
            packet = item["packet"] if "packet" in item else _packet_name(item["id"])
            packet_rows.append(
                {
                    "source": source,
                    "packet": packet,
                    "uav": item.get("uav"),
                    "delivered_s": item["delivered_s"],
                    "on_time": item.get("on_time"),
                }
            )
    _write_csv(out / "packets.csv", packet_rows)
    _write_csv(out / "scores.csv", [{**row, "partner": "wait"} for row in solo_rows] + [{**row, "partner": "backhaul"} for row in joint_public])
    print(
        f"choice {summary['solo_choice']['action']} -> {summary['joint_choice']['action']} changed {choice_changed} "
        f"shared {splits[0]['shared_link_s']:.3f} reselection {splits[0]['reselection_s']:.3f}",
        flush=True,
    )
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
