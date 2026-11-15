"""Explain the headwind return quote, then price a controller rollout.

The analytic quote stays in the guard. The bypass estimate integrates the
existing controller one second at a time, then idles at 1 W until the deadline.
Truth wind is used only for the offline ledger and for counterfactual labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

from wt.active_model_identification.run_w1 import _feasible, _label, _parse, replay_case
from wt.active_model_identification.run_w2 import ENERGY_TOL, _read_csv, load_w2, visible
from wt.active_model_identification.run_w21 import W2_DIR
from wt.active_model_identification.scenarios import observation_noise
from wt.active_model_identification.spec import ROOT
from wt.active_model_identification.wind import (
    _keep,
    charge_message,
    mean_wind_fn,
    observe,
    predict,
    return_energy_parts,
    script_action,
    true_wind,
)
from wt.active_model_identification.world import (
    _charge_radio,
    _move,
    _try_lora,
    advance,
    energy_used_j,
    mission_terminal,
    on_time_bits,
    pad_distance_m,
    pad_radius_m,
    pad_speed_m_s,
)
from wt.ocean_observation_m0.radio import rx_power_w, tx_power_w

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w3.json"
RUN_ID = "w3_20261115"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> str:
    try:
        return subprocess.check_output(["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def load_w3() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if abs(float(cfg["reserve_j"]) - 4000.0) > 1e-9 or abs(float(cfg["ground_idle_w"]) - 1.0) > 1e-12:
        raise SystemExit("W3 registration changed the reserve or the idle wattage")
    return cfg


def _message(replay: dict, condition: dict, w2: dict) -> dict:
    rng = random.Random(f"w1-{condition['id']}-{w2['seed']}")
    sample = replay["sample"]
    return observe(condition, sample["x_m"], sample["y_m"], sample["t_s"], w2["field"], float(w2["observation_sigma_m_s"]), rng)


def iter_states(replay: dict, condition: dict, mode: str, message: dict, rows: list[dict], w2: dict):
    """Yield each archived decision, then step that interval's executed action."""
    case = replay["case"]
    spec = case["spec"]
    world = replay["prefix"].snapshot()
    beliefs = [item.copy() for item in replay["prefix_beliefs"]]
    decider = int(w2["decider_uav"])
    observer = int(w2["observer_uav"])
    charged = None
    wind_at = lambda x, y, t, condition=condition: true_wind(condition, x, y, t, w2["field"])
    noise = observation_noise(int(case["scenario"]["seed"]), float(spec["belief"]["sigma_j"]))
    for row in rows:
        if abs(world.t - float(row["t_s"])) > 1e-6:
            raise SystemExit(f"archived time {row['t_s']} is not {world.t}")
        if visible(mode, world.t, w2) and charged is None:
            charged = charge_message(world, observer, decider, int(w2["payload_bytes"]))
        info = message if visible(mode, world.t, w2) else None
        yield row, world.snapshot(), [item.copy() for item in beliefs], info
        epoch = int(row["epoch"])
        actions = [("wait", None) for _ in world.uavs]
        actions[decider] = _parse(row["executed"])
        actions[observer] = _parse(case["requests"][(epoch, observer)]["executed"])
        advance(
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
            admit=_keep,
            return_guard=_keep,
        )


def _kind(uav, action, spec: dict) -> str:
    if not uav.airborne:
        return "ground"
    landing = action[0] == "return" and pad_distance_m(uav) <= pad_radius_m(spec) and math.hypot(uav.vx, uav.vy) <= pad_speed_m_s(spec)
    return "landing" if landing else "flight"


def _nominal_radio(world, uav, spec: dict, payload: int, step_end: float, receive: bool) -> float:
    before = uav.consumed_j
    airtime = _try_lora(world, step_end, payload)
    if airtime is None:
        return 0.0
    _charge_radio(world, uav, (rx_power_w() if receive else tx_power_w()) * airtime)
    return uav.consumed_j - before


def step_return(world, replay: dict, condition: dict, w2: dict, *, seconds: int, wind_fn, theta: float, decider_only: bool) -> dict:
    """Integrate one return with the existing controller.

    ``wind_fn`` and ``theta`` are supplied by the caller. The offline ledger
    passes the executor's truth; the bypass estimate passes the information set.
    """
    spec = replay["case"]["spec"]
    sim = world.snapshot()
    decider = int(w2["decider_uav"])
    buckets = {"flight_j": 0.0, "landing_j": 0.0, "ground_j": 0.0, "radio_j": 0.0, "flight_s": 0, "landing_s": 0, "ground_s": 0}
    touchdown_s = None
    limit = float(spec["time"]["mission_s"])
    for _ in range(seconds):
        if sim.t >= limit - 1e-9:
            break
        focus = sim.uavs[decider]
        clock = int(round(sim.t))
        if clock % int(spec["time"]["decision_s"]) == 0:
            buckets["radio_j"] += _nominal_radio(sim, focus, spec, int(spec["radio"]["command_bytes"]), sim.t + 1.0, True)
        kind = _kind(focus, ("return", None), spec)
        before = focus.consumed_j
        targets = (decider,) if decider_only else range(len(sim.uavs))
        for index in targets:
            item = sim.uavs[index]
            action = ("return", None) if index == decider else ("wait", None)
            item_theta = theta if index == decider else float(replay["case"]["scenario"]["theta"][index])
            _move(item, action, spec, item_theta, clock % int(spec["time"]["decision_s"]), wind_fn(item.x, item.y, sim.t))
        delta = focus.consumed_j - before
        buckets[f"{kind}_j"] += delta
        buckets[f"{kind}_s"] += 1
        if kind == "landing" and (not focus.airborne) and touchdown_s is None:
            touchdown_s = float(sim.t + 1.0)
        if clock % int(spec["time"]["decision_s"]) == int(spec["time"]["decision_s"]) - 1:
            buckets["radio_j"] += _nominal_radio(sim, focus, spec, int(spec["radio"]["telemetry_bytes"]), sim.t + 1.0, False)
        sim.t += 1.0
    return {
        "buckets": buckets,
        "touchdown_s": touchdown_s,
        "terminal_energy_j": float(sim.uavs[decider].energy_j),
        "landed": (not sim.uavs[decider].airborne) and pad_distance_m(sim.uavs[decider]) <= pad_radius_m(spec),
    }


def quote_return(world, decider: int, theta: float, wind_xy, spec: dict, factor: float) -> dict:
    uav = world.uavs[decider]
    parts = return_energy_parts(uav, theta, spec, factor, wind_xy)
    remaining = max(0.0, float(spec["time"]["mission_s"]) - float(world.t))
    idle_s = max(0.0, remaining - parts["cruise_s"] - parts["landing_s"])
    idle_j = float(spec["energy"]["ground_idle_w"]) * idle_s
    return {
        "cruise_s": float(parts["cruise_s"]),
        "cruise_j": float(parts["cruise_j"]),
        "landing_s": float(parts["landing_s"]),
        "landing_j": float(parts["landing_j"]),
        "flight_total_j": float(parts["total_j"]),
        "idle_s": idle_s,
        "idle_j": idle_j,
        "terminal_energy_j": float(uav.energy_j) - float(parts["total_j"]) - idle_j,
    }


def reconcile_interval(world, replay: dict, condition: dict, beliefs, info, w2: dict) -> dict:
    """Price the archived 30 s return and the analytic quote on the same state."""
    spec = replay["case"]["spec"]
    factor = replay["case"]["factor"]
    decider = int(w2["decider_uav"])
    uav = world.uavs[decider]
    predicted = predict(info, uav.x, uav.y, world.t, w2["field"], float(w2["prior_std_m_s"]))["wind_xy"]
    truth = true_wind(condition, uav.x, uav.y, world.t, w2["field"])
    theta_q = beliefs[decider].quantile(float(spec["belief"]["guard_probability"]))
    theta_mean = beliefs[decider].mean()
    theta_true = float(replay["case"]["scenario"]["theta"][decider])
    executed = step_return(
        world,
        replay,
        condition,
        w2,
        seconds=int(spec["time"]["decision_s"]),
        wind_fn=lambda x, y, t: true_wind(condition, x, y, t, w2["field"]),
        theta=theta_true,
        decider_only=True,
    )
    quotes = {
        "q95_predicted": quote_return(world, decider, theta_q, predicted, spec, factor),
        "mean_predicted": quote_return(world, decider, theta_mean, predicted, spec, factor),
        "q95_truth_wind": quote_return(world, decider, theta_q, truth, spec, factor),
        "truth_theta_truth_wind": quote_return(world, decider, theta_true, truth, spec, factor),
    }
    analytic = quotes["q95_predicted"]
    buckets = executed["buckets"]
    raw_gap = (buckets["flight_j"] + buckets["landing_j"] + buckets["ground_j"] + buckets["radio_j"]) - analytic["flight_total_j"]
    return {
        "t_s": float(world.t),
        "distance_m": float(pad_distance_m(uav)),
        "speed_m_s": float(math.hypot(uav.vx, uav.vy)),
        "energy_j": float(uav.energy_j),
        "theta_mean": float(theta_mean),
        "theta_q95": float(theta_q),
        "theta_truth": theta_true,
        "predicted_wind_m_s": [float(predicted[0]), float(predicted[1])],
        "truth_wind_m_s": [float(truth[0]), float(truth[1])],
        "executed": {key: (int(value) if key.endswith("_s") else float(value)) for key, value in buckets.items()},
        "interval_end_energy_j": float(executed["terminal_energy_j"]),
        "touchdown_s": executed["touchdown_s"],
        "quotes": quotes,
        "raw_gap_j": float(raw_gap),
        "flight_gap_j": float(buckets["flight_j"] - analytic["cruise_j"]),
        "landing_gap_j": float(buckets["landing_j"] - analytic["landing_j"]),
        "omitted_ground_and_radio_j": float(buckets["ground_j"] + buckets["radio_j"]),
        "truth_wind_formula_shift_j": float(quotes["q95_truth_wind"]["flight_total_j"] - analytic["flight_total_j"]),
        "duration_gap_s": float(buckets["flight_s"] - analytic["cruise_s"]),
    }


def project_terminal(world, replay: dict, condition: dict, beliefs, info, w2: dict, *, theta_mode: str, wind_mode: str) -> dict:
    spec = replay["case"]["spec"]
    decider = int(w2["decider_uav"])
    if theta_mode == "mean":
        theta = beliefs[decider].mean()
    elif theta_mode == "q95":
        theta = beliefs[decider].quantile(float(spec["belief"]["guard_probability"]))
    elif theta_mode == "truth":
        theta = float(replay["case"]["scenario"]["theta"][decider])
    else:
        raise ValueError(theta_mode)
    if wind_mode == "predicted":
        wind_fn = mean_wind_fn(info, w2["field"], float(w2["prior_std_m_s"]))
    elif wind_mode == "truth":
        wind_fn = lambda x, y, t: true_wind(condition, x, y, t, w2["field"])
    else:
        raise ValueError(wind_mode)
    remaining = int(round(float(spec["time"]["mission_s"]) - float(world.t)))
    rolled = step_return(world, replay, condition, w2, seconds=remaining, wind_fn=wind_fn, theta=float(theta), decider_only=True)
    rolled["theta"] = float(theta)
    rolled["theta_mode"] = theta_mode
    rolled["wind_mode"] = wind_mode
    rolled["holds_reserve"] = rolled["terminal_energy_j"] + 1e-6 >= float(spec["energy"]["reserve_j"])
    return rolled


def old_terminal(world, beliefs, info, replay: dict, w2: dict) -> dict:
    spec = replay["case"]["spec"]
    decider = int(w2["decider_uav"])
    uav = world.uavs[decider]
    theta = beliefs[decider].quantile(float(spec["belief"]["guard_probability"]))
    wind = predict(info, uav.x, uav.y, world.t, w2["field"], float(w2["prior_std_m_s"]))["wind_xy"]
    quote = quote_return(world, decider, theta, wind, spec, replay["case"]["factor"])
    quote["holds_reserve"] = quote["terminal_energy_j"] + 1e-6 >= float(spec["energy"]["reserve_j"])
    return quote


def pure_return_idle(rows: list[dict], t_s: float) -> bool:
    later = [row for row in rows if float(row["t_s"]) + 1e-9 >= t_s]
    return bool(later) and all(row["executed"].split(":")[0] in ("return", "wait") for row in later)


def fork_action(world, replay: dict, condition: dict, beliefs, action, w2: dict) -> dict:
    """Replay one existing candidate under truth wind. The return guard stays off."""
    case = replay["case"]
    spec = case["spec"]
    sim = world.snapshot()
    local = [item.copy() for item in beliefs]
    decider = int(w2["decider_uav"])
    observer = int(w2["observer_uav"])
    watched = [packet.id for packet in sim.packets if packet.holder == f"uav:{decider}"]
    wind_at = lambda x, y, t: true_wind(condition, x, y, t, w2["field"])
    noise = observation_noise(int(case["scenario"]["seed"]), float(spec["belief"]["sigma_j"]))
    interval = 0
    mission = float(spec["time"]["mission_s"])
    while sim.t < mission - ENERGY_TOL:
        epoch = int(round(sim.t / float(spec["time"]["decision_s"])))
        chosen = script_action(sim, decider, action, spec, interval)
        if chosen is None:
            chosen = ("wait", None)
        actions = [("wait", None) for _ in sim.uavs]
        actions[decider] = chosen
        if (epoch, observer) in case["requests"]:
            actions[observer] = _parse(case["requests"][(epoch, observer)]["executed"])
        advance(
            sim,
            actions,
            list(case["scenario"]["theta"]),
            local,
            spec,
            case["factor"],
            None,
            planning=False,
            interval_index=epoch,
            noise_j=noise,
            wind_at=wind_at,
            admit=_keep,
            return_guard=_keep,
        )
        interval += 1
    terminals = mission_terminal(sim, spec)
    packets = []
    by_id = {packet.id: packet for packet in sim.packets}
    for packet_id in watched:
        packet = by_id[packet_id]
        packets.append({"id": packet_id, "holder": packet.holder, "delivered_s": packet.delivered_s, "on_time": packet.delivered_s is not None and packet.delivered_s <= mission + 1e-9 and packet.holder == "shore"})
    return {
        "action": _label(action),
        "on_time_bits": float(on_time_bits(sim)),
        "energy_j": float(math.fsum(energy_used_j(sim))),
        "uav_energy_j": [float(item["energy_j"]) for item in terminals],
        "returned": int(sum(bool(item["returned"]) for item in terminals)),
        "below_reserve": int(sum(bool(item["below_reserve"]) for item in terminals)),
        "not_landed": int(sum(bool(item["not_landed"]) for item in terminals)),
        "packets": packets,
        "cache_on_time": all(item["on_time"] for item in packets) and bool(packets),
        "reserve_met": all(item["energy_j"] + 1e-6 >= float(spec["energy"]["reserve_j"]) and not item["not_landed"] for item in terminals),
    }


def _mae(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return float(sum(abs(row[key]) for row in rows) / len(rows))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    out = out_dir or (ROOT / "results" / "wind_observation_w3" / RUN_ID)
    if "wind_observation_w3" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w3")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_w3()
    w2 = load_w2()
    replay = replay_case()
    if not replay["ok"]:
        raise SystemExit("W1 prefix replay no longer matches M1.6")
    decisions = _read_csv(W2_DIR / "decisions.csv")
    scenarios = _read_csv(W2_DIR / "scenarios.csv")
    headwind = next(row for row in w2["conditions"] if row["id"] == "headwind_return")
    message = _message(replay, headwind, w2)
    timely_rows = [row for row in decisions if row["condition"] == "headwind_return" and row["mode"] == "timely"]
    state_270 = None
    beliefs_270 = None
    for row, world, beliefs, info in iter_states(replay, headwind, "timely", message, timely_rows, w2):
        if abs(float(row["t_s"]) - float(cfg["audit_time_s"])) <= ENERGY_TOL:
            state_270 = world
            beliefs_270 = beliefs
            info_270 = info
            break
    if state_270 is None:
        raise SystemExit("270 s state was not on the archived trajectory")
    ledger = reconcile_interval(state_270, replay, headwind, beliefs_270, info_270, w2)
    truth_roll = project_terminal(state_270, replay, headwind, beliefs_270, info_270, w2, theta_mode="truth", wind_mode="truth")
    errors = []
    highlights = []
    forks = []
    for condition in w2["conditions"]:
        arm_message = _message(replay, condition, w2)
        for mode in w2["modes"]:
            rows = [row for row in decisions if row["condition"] == condition["id"] and row["mode"] == mode]
            reference = next(row for row in scenarios if row["condition"] == condition["id"] and row["mode"] == mode)
            archived_terminal = json.loads(reference["uav_energy_j"])[int(w2["decider_uav"])]
            for row, world, beliefs, info in iter_states(replay, condition, mode, arm_message, rows, w2):
                if pure_return_idle(rows, float(row["t_s"])):
                    label = float(archived_terminal)
                    label_source = "archived_terminal"
                else:
                    rolled = fork_action(world, replay, condition, beliefs, ("return", None), w2)
                    label = float(rolled["uav_energy_j"][int(w2["decider_uav"])])
                    label_source = "truth_return_fork"
                old = old_terminal(world, beliefs, info, replay, w2)
                new_mean = project_terminal(world, replay, condition, beliefs, info, w2, theta_mode="mean", wind_mode="predicted")
                new_q = project_terminal(world, replay, condition, beliefs, info, w2, theta_mode="q95", wind_mode="predicted")
                actual_holds = label + 1e-6 >= float(cfg["reserve_j"])
                item = {
                    "condition": condition["id"],
                    "mode": mode,
                    "t_s": float(row["t_s"]),
                    "used_message": bool(info),
                    "label_j": label,
                    "label_source": label_source,
                    "old_q95_j": old["terminal_energy_j"],
                    "new_mean_j": new_mean["terminal_energy_j"],
                    "new_q95_j": new_q["terminal_energy_j"],
                    "old_error_j": old["terminal_energy_j"] - label,
                    "new_mean_error_j": new_mean["terminal_energy_j"] - label,
                    "new_q95_error_j": new_q["terminal_energy_j"] - label,
                    "actual_holds": actual_holds,
                    "old_false_clear": bool(old["holds_reserve"] and not actual_holds),
                    "new_q95_false_clear": bool(new_q["holds_reserve"] and not actual_holds),
                    "old_false_alarm": bool((not old["holds_reserve"]) and actual_holds),
                    "new_q95_false_alarm": bool((not new_q["holds_reserve"]) and actual_holds),
                }
                errors.append(item)
                if condition["id"] == "headwind_return" and mode == "timely" and float(row["t_s"]) in set(cfg["highlight_times_s"]):
                    highlights.append(item)
                    for action in _feasible(replay["case"]["spec"], world, int(w2["decider_uav"])):
                        outcome = fork_action(world, replay, condition, beliefs, action, w2)
                        outcome.update({"condition": condition["id"], "mode": mode, "t_s": float(row["t_s"])})
                        forks.append(outcome)
    headwind_errors = [row for row in errors if row["condition"] == "headwind_return" and row["mode"] == "timely"]
    first_warning = next((row["t_s"] for row in headwind_errors if row["new_q95_j"] + 1e-6 < float(cfg["reserve_j"])), None)
    feasible = [row for row in forks if row["cache_on_time"] and row["reserve_met"]]
    last_feasible = max((row["t_s"] for row in feasible), default=None)
    summary_rows = []
    for condition in w2["conditions"]:
        for mode in w2["modes"]:
            group = [row for row in errors if row["condition"] == condition["id"] and row["mode"] == mode]
            summary_rows.append(
                {
                    "condition": condition["id"],
                    "mode": mode,
                    "old_mae_j": _mae(group, "old_error_j"),
                    "new_mean_mae_j": _mae(group, "new_mean_error_j"),
                    "new_q95_mae_j": _mae(group, "new_q95_error_j"),
                    "old_false_clear": sum(bool(row["old_false_clear"]) for row in group),
                    "new_q95_false_clear": sum(bool(row["new_q95_false_clear"]) for row in group),
                    "old_false_alarm": sum(bool(row["old_false_alarm"]) for row in group),
                    "new_q95_false_alarm": sum(bool(row["new_q95_false_alarm"]) for row in group),
                }
            )
    adopt = False
    if feasible:
        conclusion = "已检验动作里有真风回放同时保住交付和备用。本轮只记录，不改默认策略。"
    else:
        conclusion = "控制器积分补上了返航刹车的时长。180、210 和 270 秒的已有动作都不能同时保住交付和备用。默认策略不变。"
    summary = {
        "schema_version": 1,
        "parent_commit": _git(),
        "config_sha256": _sha256(CONFIG_PATH),
        "result_dir": str(out.relative_to(ROOT)).replace("\\", "/"),
        "elapsed_s": time.perf_counter() - started,
        "ledger": ledger,
        "truth_roll_terminal_j": truth_roll["terminal_energy_j"],
        "highlights": highlights,
        "error_summary": summary_rows,
        "first_q95_warning_s": first_warning,
        "feasible_actions": [{key: row[key] for key in ("t_s", "action", "on_time_bits", "uav_energy_j", "cache_on_time", "reserve_met")} for row in feasible],
        "last_feasible_t_s": last_feasible,
        "paired_scheduler_run": False,
        "conclusion": conclusion,
        "adopt": adopt,
        "default_changed": False,
    }
    public_forks = []
    for row in forks:
        copied = {key: row[key] for key in ("condition", "mode", "t_s", "action", "on_time_bits", "energy_j", "returned", "below_reserve", "not_landed", "cache_on_time", "reserve_met")}
        copied["uav_energy_j"] = json.dumps(row["uav_energy_j"])
        copied["packets"] = json.dumps(row["packets"], ensure_ascii=False)
        public_forks.append(copied)
    (out / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out / "errors.csv", errors)
    _write_csv(out / "forks.csv", public_forks)
    print(conclusion, flush=True)
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
