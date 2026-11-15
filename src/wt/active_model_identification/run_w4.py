"""Ask whether an earlier wind message can change delivery and reserve.

The executor uses the registered wind from takeoff. The no-message arm keeps the
existing scheduler. Forks replace one legal action for one interval and then
return to that scheduler. The message arms are separate: they rank existing
candidates by the W3 controller rollout and may use a sample only after the
link airtime has elapsed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

from wt.active_model_identification.belief import prior_belief
from wt.active_model_identification.policies import action_rank, select_lookahead, tie_abs
from wt.active_model_identification.run_w1 import ENERGY_TOL, _feasible, _label, _parse, load_case
from wt.active_model_identification.scenarios import observation_noise
from wt.active_model_identification.spec import ROOT, decision_count
from wt.active_model_identification.wind import (
    _keep,
    charge_message,
    fly_script,
    mean_wind_fn,
    message_cost_j,
    observe,
    true_wind,
)
from wt.active_model_identification.world import (
    advance,
    energy_used_j,
    make_world,
    mission_terminal,
    on_time_bits,
    utility,
)

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w4.json"
RUN_ID = "w4_20261115"
TAG = "w4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def load_w4() -> tuple[dict, dict]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    case = load_case()
    reserve = float(case["spec"]["energy"]["reserve_j"])
    idle = float(case["spec"]["energy"]["ground_idle_w"])
    if abs(float(cfg["reserve_j"]) - reserve) > 1e-9 or abs(float(cfg["ground_idle_w"]) - idle) > 1e-12:
        raise SystemExit("W4 registration changed the reserve or the idle wattage")
    if cfg["scenario_id"] != case["cfg"]["scenario_id"] or int(cfg["seed"]) != int(case["cfg"]["seed"]):
        raise SystemExit("W4 scenario does not match the frozen b3_loose case")
    return cfg, case


def _condition(case: dict, name: str) -> dict:
    return next(row for row in case["cfg"]["conditions"] if row["id"] == name)


def _truth(condition: dict, field: dict):
    return lambda x, y, t, condition=condition, field=field: true_wind(condition, x, y, t, field)


def _fresh(case: dict):
    spec = case["spec"]
    scenario = case["scenario"]
    world = make_world(spec, scenario["battery_j"], TAG)
    beliefs = [prior_belief(spec), prior_belief(spec)]
    return world, beliefs


def _key(packet_id: str) -> tuple[int, int]:
    buoy = int(packet_id.split("-b")[1].split("-p")[0])
    number = int(packet_id.split("-p")[1])
    return buoy, number


def assess(world, spec: dict, control_keys: set[tuple[int, int]], target_buoy: int, control_bits: float) -> dict:
    mission = float(spec["time"]["mission_s"])
    on_time: set[tuple[int, int]] = set()
    target = []
    for packet in world.packets:
        key = _key(packet.id)
        delivered = packet.delivered_s is not None and packet.holder == "shore" and float(packet.delivered_s) <= mission + 1e-9
        if delivered:
            on_time.add(key)
        if key[0] == target_buoy:
            target.append(
                {
                    "packet": f"b{key[0]}-p{key[1]}",
                    "holder": packet.holder,
                    "delivered_s": packet.delivered_s,
                    "on_time": delivered,
                }
            )
    terminals = mission_terminal(world, spec)
    missing = sorted(f"b{buoy}-p{number}" for buoy, number in control_keys - on_time)
    target_on_time = all(item["on_time"] for item in target) and bool(target)
    reserve_met = all(bool(item["returned"]) for item in terminals)
    bits = float(on_time_bits(world))
    return {
        "on_time_bits": bits,
        "energy_j": float(sum(energy_used_j(world))),
        "utility": float(utility(world, spec)),
        "uav_energy_j": [float(item["energy_j"]) for item in terminals],
        "returned": int(sum(bool(item["returned"]) for item in terminals)),
        "below_reserve": int(sum(bool(item["below_reserve"]) for item in terminals)),
        "not_landed": int(sum(bool(item["not_landed"]) for item in terminals)),
        "target": target,
        "missing_control_packets": missing,
        "target_on_time": target_on_time,
        "business_kept": not missing,
        "reserve_met": reserve_met,
        "full_goal": (not missing) and target_on_time and reserve_met,
        "delivery_held": bits + ENERGY_TOL >= control_bits and reserve_met,
        "energy_shortfall_j": [max(0.0, float(spec["energy"]["reserve_j"]) - float(item["energy_j"])) for item in terminals],
    }


def play(case: dict, condition: dict, world, beliefs, start_epoch: int, preset: list | None = None) -> dict:
    """Run the existing scheduler. ``preset`` replaces only the first interval."""
    spec = case["spec"]
    scenario = case["scenario"]
    method = case["cfg"]["method"]
    factor = case["factor"]
    noise = observation_noise(int(scenario["seed"]), float(spec["belief"]["sigma_j"]))
    wind_at = _truth(condition, case["cfg"]["field"])
    rows = []
    states = []
    for epoch in range(start_epoch, decision_count(spec)):
        states.append(
            {
                "epoch": epoch,
                "t_s": float(world.t),
                "world": world.snapshot(),
                "beliefs": [item.copy() for item in beliefs],
            }
        )
        if preset is not None and epoch == start_epoch:
            requested = list(preset)
        else:
            requested, _tables = select_lookahead(method, world, beliefs, spec, factor, None)
        observations = advance(
            world,
            requested,
            list(scenario["theta"]),
            beliefs,
            spec,
            factor,
            None,
            planning=False,
            interval_index=epoch,
            noise_j=noise,
            wind_at=wind_at,
        )
        for obs in observations:
            rows.append(
                {
                    "epoch": int(obs["interval"]),
                    "t_s": float(obs["interval"]) * float(spec["time"]["decision_s"]),
                    "uav": int(obs["uav"]),
                    "requested": _label(tuple(obs["requested"])),
                    "executed": _label(tuple(obs["executed"])),
                    "energy_j": float(obs["energy_j"]),
                    "x_m": float(obs["x"]),
                    "y_m": float(obs["y"]),
                    "airborne": bool(obs["airborne"]),
                    "intervened": bool(obs["intervened"]),
                }
            )
    return {"rows": rows, "states": states, "world": world, "beliefs": beliefs}


def baseline(case: dict, condition: dict) -> dict:
    world, beliefs = _fresh(case)
    played = play(case, condition, world, beliefs, 0, None)
    played["condition"] = condition["id"]
    return played


def build_link(played: dict, case: dict, condition: dict, cfg: dict) -> dict:
    """Earliest decision-time sample after takeoff, usable only after the airtime."""
    observer = int(cfg["observer_uav"])
    sample_state = next(state for state in played["states"] if state["t_s"] > 0.0 and state["world"].uavs[observer].airborne)
    uav = sample_state["world"].uavs[observer]
    rng = random.Random(f"w1-{condition['id']}-{case['cfg']['seed']}")
    message = observe(
        condition,
        uav.x,
        uav.y,
        float(sample_state["t_s"]),
        case["cfg"]["field"],
        float(case["cfg"]["observation_sigma_m_s"]),
        rng,
    )
    cost = message_cost_j(int(case["cfg"]["payload_bytes"]))
    ready = float(sample_state["t_s"]) + float(cost["airtime_s"])
    visible = next(state["t_s"] for state in played["states"] if state["t_s"] > float(sample_state["t_s"]) + 1e-9 and state["t_s"] + 1e-9 >= ready)
    step = float(case["spec"]["time"]["decision_s"])
    commits = [
        row
        for row in played["rows"]
        if row["requested"].startswith("service:") and int(row["requested"].split(":")[1]) == int(cfg["target_buoy"])
    ]
    team = min(commits, key=lambda row: (row["t_s"], row["uav"]))
    receiver_rows = [row for row in commits if row["uav"] == int(cfg["receiver_uav"])]
    receiver = min(receiver_rows, key=lambda row: row["t_s"])
    return {
        "sample_t_s": float(sample_state["t_s"]),
        "sample_x_m": float(uav.x),
        "sample_y_m": float(uav.y),
        "airtime_s": float(cost["airtime_s"]),
        "message_j": float(cost["sender_j"] + cost["receiver_j"]),
        "timely_t_s": float(visible),
        "late_t_s": float(visible) + step,
        "team_commit_t_s": float(team["t_s"]),
        "team_commit_uav": int(team["uav"]),
        "receiver_commit_t_s": float(receiver["t_s"]),
        "team_shortfall_s": max(0.0, float(visible) - float(team["t_s"])),
        "receiver_shortfall_s": max(0.0, float(visible) - float(receiver["t_s"])),
        "message": message,
        "cost": cost,
    }


def _visible(mode: str, t_s: float, link: dict) -> bool:
    if mode == "no_send":
        return False
    ready = float(link["timely_t_s"] if mode == "timely" else link["late_t_s"])
    return t_s + 1e-9 >= ready


def quote_candidate(case: dict, world, beliefs, uav_index: int, action, message: dict | None, theta_mode: str) -> dict:
    spec = case["spec"]
    thetas = [item.mean() for item in beliefs]
    if theta_mode == "q95":
        thetas[uav_index] = beliefs[uav_index].quantile(float(spec["belief"]["guard_probability"]))
    elif theta_mode != "mean":
        raise ValueError(theta_mode)
    wind_at = mean_wind_fn(message, case["cfg"]["field"], float(case["cfg"]["prior_std_m_s"]))
    result = fly_script(
        world,
        beliefs,
        spec,
        case["factor"],
        uav_index,
        action,
        thetas,
        wind_at,
        lambda _uav, _interval: 0.0,
        admit=_keep,
        return_guard=_keep,
    )
    landed = bool(result["landed"]) and not bool(result["failed"])
    idle_s = max(0.0, float(spec["time"]["mission_s"]) - float(result["t_end_s"])) if landed else 0.0
    idle_j = float(spec["energy"]["ground_idle_w"]) * idle_s
    terminal = float(result["sim"].uavs[uav_index].energy_j) - idle_j
    return {
        "terminal_j": terminal,
        "bits": float(result["on_time_bits"]),
        "energy_j": float(result["focus_energy_j"]) + idle_j,
        "landed": landed,
        "on_time": bool(result["on_time"]),
    }


def _choose(scored: list[dict], scheduler_action, tau: float) -> tuple[tuple, str]:
    admitted = [row for row in scored if row["admitted"]]
    if not admitted:
        return scheduler_action, "fallback_scheduler"
    best = max(row["utility"] for row in admitted)
    pool = [row for row in admitted if best - row["utility"] <= tau + 1e-15]
    for row in pool:
        if row["action"] == scheduler_action:
            return scheduler_action, "tie_scheduler"
    pool.sort(key=lambda row: (action_rank(row["action"]), -row["utility"]))
    return pool[0]["action"], "max_utility"


def _repair(actions: list, tables: list[list[dict]]) -> list:
    if not (actions[0] == actions[1] and actions[0][0] == "service"):
        return actions
    utilities = []
    for uav, action in enumerate(actions):
        row = next(item for item in tables[uav] if item["action"] == action)
        utilities.append(row["utility"] if row["admitted"] else -1.0e9)
    weaker = 0 if utilities[0] <= utilities[1] else 1
    other = actions[1 - weaker]
    ranked = sorted(tables[weaker], key=lambda row: (-row["utility"], action_rank(row["action"])))
    for row in ranked:
        if row["admitted"] and row["action"] != other:
            actions[weaker] = row["action"]
            return actions
    actions[weaker] = ("return", None)
    return actions


def play_message(case: dict, condition: dict, mode: str, link: dict) -> dict:
    spec = case["spec"]
    world, beliefs = _fresh(case)
    scenario = case["scenario"]
    noise = observation_noise(int(scenario["seed"]), float(spec["belief"]["sigma_j"]))
    wind_at = _truth(condition, case["cfg"]["field"])
    tau = tie_abs(spec)
    reserve = float(spec["energy"]["reserve_j"])
    block = spec["utility"]
    rows = []
    charged = None
    for epoch in range(decision_count(spec)):
        if _visible(mode, world.t, link) and charged is None:
            charged = charge_message(
                world,
                int(case["cfg"]["observer_uav"]),
                int(case["cfg"]["decider_uav"]),
                int(case["cfg"]["payload_bytes"]),
            )
        info = link["message"] if _visible(mode, world.t, link) else None
        scheduler, _tables = select_lookahead(case["cfg"]["method"], world, beliefs, spec, case["factor"], None)
        tables = []
        chosen = []
        reasons = []
        for uav, scheduler_action in enumerate(scheduler):
            scored = []
            for action in _feasible(spec, world, uav):
                risk = quote_candidate(case, world, beliefs, uav, action, info, "q95")
                mean = quote_candidate(case, world, beliefs, uav, action, info, "mean")
                needs_bits = action[0] in ("service", "backhaul")
                admitted = bool(risk["landed"] and risk["on_time"] and risk["terminal_j"] + 1e-6 >= reserve and (not needs_bits or risk["bits"] > 0.0))
                score = -1.0e6
                if mean["landed"]:
                    score = mean["bits"] / float(block["b_ref_bits"]) - float(block["lambda_e"]) * mean["energy_j"] / float(block["e_ref_j"])
                scored.append({"action": action, "admitted": admitted, "utility": score, "terminal_j": risk["terminal_j"], "bits": mean["bits"]})
            action, reason = _choose(scored, scheduler_action, tau)
            tables.append(scored)
            chosen.append(action)
            reasons.append(reason)
        chosen = _repair(chosen, tables)
        observations = advance(
            world,
            chosen,
            list(scenario["theta"]),
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
            rows.append(
                {
                    "condition": condition["id"],
                    "mode": mode,
                    "epoch": int(obs["interval"]),
                    "t_s": float(obs["interval"]) * float(spec["time"]["decision_s"]),
                    "uav": int(obs["uav"]),
                    "requested": _label(tuple(obs["requested"])),
                    "executed": _label(tuple(obs["executed"])),
                    "reason": reasons[int(obs["uav"])],
                    "used_message": bool(info),
                    "energy_j": float(obs["energy_j"]),
                }
            )
        if epoch <= 6 or epoch % 15 == 0:
            labels = " ".join(_label(action) for action in chosen)
            print(f"{condition['id']} {mode} e{epoch:02d} {labels}", flush=True)
    message_j = 0.0 if charged is None else float(charged["sender_j_drawn"] + charged["receiver_j_drawn"])
    return {"rows": rows, "world": world, "message_j": message_j}


def _mission_changed(left: dict, right: dict) -> bool:
    return (
        abs(left["on_time_bits"] - right["on_time_bits"]) > ENERGY_TOL
        or left["below_reserve"] != right["below_reserve"]
        or left["not_landed"] != right["not_landed"]
    )


def _actions_changed(left: list[dict], right: list[dict]) -> tuple[bool, float | None]:
    paired = {(row["epoch"], row["uav"]): row for row in right}
    for row in left:
        other = paired[(row["epoch"], row["uav"])]
        if row["requested"] != other["requested"] or row["executed"] != other["executed"]:
            return True, float(row["t_s"])
    return False, None


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _public_outcome(row: dict) -> dict:
    copied = {key: row[key] for key in row if key not in ("target",)}
    copied["target"] = json.dumps(row["target"], ensure_ascii=False)
    copied["uav_energy_j"] = json.dumps(row["uav_energy_j"])
    copied["missing_control_packets"] = json.dumps(row["missing_control_packets"], ensure_ascii=False)
    copied["energy_shortfall_j"] = json.dumps(row["energy_shortfall_j"])
    return copied


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    out = out_dir or (ROOT / "results" / "wind_observation_w4" / RUN_ID)
    if "wind_observation_w4" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w4")
    out.mkdir(parents=True, exist_ok=True)
    cfg, case = load_w4()
    headwind = _condition(case, "headwind_return")
    print("baseline", flush=True)
    played = baseline(case, headwind)
    control = assess(played["world"], case["spec"], set(), int(cfg["target_buoy"]), 0.0)
    control_keys = set()
    for packet in played["world"].packets:
        key = _key(packet.id)
        delivered = packet.delivered_s is not None and packet.holder == "shore" and float(packet.delivered_s) <= float(case["spec"]["time"]["mission_s"]) + 1e-9
        if delivered:
            control_keys.add(key)
    control = assess(played["world"], case["spec"], control_keys, int(cfg["target_buoy"]), control["on_time_bits"])
    link = build_link(played, case, headwind, cfg)
    print(
        f"sample {link['sample_t_s']} timely {link['timely_t_s']} team {link['team_commit_t_s']} short {link['team_shortfall_s']}",
        flush=True,
    )
    confirmation = play(
        case,
        headwind,
        played["states"][0]["world"].snapshot(),
        [item.copy() for item in played["states"][0]["beliefs"]],
        0,
        None,
    )
    confirmed = assess(confirmation["world"], case["spec"], control_keys, int(cfg["target_buoy"]), control["on_time_bits"])
    if abs(confirmed["on_time_bits"] - control["on_time_bits"]) > 1e-3 or max(abs(a - b) for a, b in zip(confirmed["uav_energy_j"], control["uav_energy_j"])) > 1e-3:
        raise SystemExit("scheduler continuation from the takeoff snapshot diverges")
    forks = []
    inspected = []
    for state in played["states"]:
        if state["t_s"] >= 180.0 - 1e-9:
            continue
        requests = [row for row in played["rows"] if row["epoch"] == state["epoch"]]
        for request in requests:
            kind = request["requested"].split(":")[0]
            if kind not in ("service", "return"):
                continue
            inspected.append({"t_s": state["t_s"], "uav": request["uav"], "requested": request["requested"]})
            uav = int(request["uav"])
            current = _parse(request["requested"])
            partner = {row["uav"]: _parse(row["requested"]) for row in requests}
            for action in _feasible(case["spec"], state["world"], uav):
                if action == current:
                    continue
                actions = [partner[0], partner[1]]
                actions[uav] = action
                print(f"fork t={state['t_s']} u{uav} { _label(action) }", flush=True)
                forked = play(
                    case,
                    headwind,
                    state["world"].snapshot(),
                    [item.copy() for item in state["beliefs"]],
                    int(state["epoch"]),
                    actions,
                )
                outcome = assess(forked["world"], case["spec"], control_keys, int(cfg["target_buoy"]), control["on_time_bits"])
                outcome.update(
                    {
                        "t_s": state["t_s"],
                        "uav": uav,
                        "action": _label(action),
                        "source": "offline_fork",
                        "executed": forked["rows"][uav]["executed"],
                    }
                )
                forks.append(outcome)
    baseline_choice = {
        "t_s": float(link["team_commit_t_s"]),
        "uav": int(link["team_commit_uav"]),
        "action": next(row["requested"] for row in played["rows"] if row["t_s"] == link["team_commit_t_s"] and row["uav"] == link["team_commit_uav"]),
        "source": "scheduler_baseline",
        "executed": next(row["executed"] for row in played["rows"] if row["t_s"] == link["team_commit_t_s"] and row["uav"] == link["team_commit_uav"]),
        **control,
    }
    feasible = [row for row in forks if row["full_goal"]]
    latest = max([row["t_s"] for row in feasible] + ([baseline_choice["t_s"]] if control["full_goal"] else []), default=None)
    online_rows = []
    arms = []
    tailwind_arms = []
    ran_online = bool(control["full_goal"] or feasible)
    if ran_online:
        head_arms = []
        for mode in ("no_send", "timely", "late"):
            print("online", mode, flush=True)
            result = play_message(case, headwind, mode, link)
            outcome = assess(result["world"], case["spec"], control_keys, int(cfg["target_buoy"]), control["on_time_bits"])
            outcome.update({"condition": headwind["id"], "mode": mode, "message_j": result["message_j"], "source": "w3_quote"})
            head_arms.append(outcome)
            online_rows.extend(result["rows"])
            arms.append(outcome)
        timely_rows = [row for row in online_rows if row["mode"] == "timely" and row["condition"] == "headwind_return"]
        nosend_rows = [row for row in online_rows if row["mode"] == "no_send" and row["condition"] == "headwind_return"]
        action_change, first_change = _actions_changed(timely_rows, nosend_rows)
        by_mode = {row["mode"]: row for row in head_arms}
        mission_change = _mission_changed(by_mode["timely"], by_mode["no_send"])
        gain = bool(by_mode["timely"]["full_goal"] and not by_mode["no_send"]["full_goal"])
        if (action_change and mission_change) or gain:
            tailwind = _condition(case, "tailwind_return")
            tail_played = baseline(case, tailwind)
            tail_link = build_link(tail_played, case, tailwind, cfg)
            for mode in ("no_send", "timely", "late"):
                print("tailwind", mode, flush=True)
                result = play_message(case, tailwind, mode, tail_link)
                outcome = assess(result["world"], case["spec"], control_keys, int(cfg["target_buoy"]), control["on_time_bits"])
                outcome.update({"condition": tailwind["id"], "mode": mode, "message_j": result["message_j"], "source": "w3_quote"})
                tailwind_arms.append(outcome)
                online_rows.extend(result["rows"])
                arms.append(outcome)
    else:
        action_change, first_change, mission_change = False, None, False
    recommend = False
    if ran_online:
        head = {row["mode"]: row for row in arms if row["condition"] == "headwind_return"}
        gain = bool(head["timely"]["full_goal"] and not head["no_send"]["full_goal"])
        tail_ok = True
        if tailwind_arms:
            tail = {row["mode"]: row for row in tailwind_arms}
            tail_ok = tail["timely"]["on_time_bits"] + ENERGY_TOL >= tail["no_send"]["on_time_bits"] and tail["timely"]["below_reserve"] <= tail["no_send"]["below_reserve"]
        recommend = bool(gain and tail_ok and (not (action_change and mission_change) or tailwind_arms))
    if not ran_online:
        conclusion = "已检验的起飞后决策里，没有一条已有动作能同时保住对照组业务、3 号浮标两包和两机备用。"
    elif control["full_goal"] and not action_change:
        conclusion = "真风从起飞就作用时，原调度已经同时保住按时交付和备用。三条报文没有改已执行动作。"
    elif control["full_goal"] and action_change and not mission_change:
        conclusion = "报文改变了动作。按时交付、落地和备用没有分开。"
    elif recommend:
        conclusion = "及时报文让交付和备用达到标准，不发送没有达到，顺风对照没有少交付或失去备用。"
    else:
        conclusion = "早到的风观测没有形成可推荐的交付和备用收益。默认调度保持原样。"
    public_link = {key: value for key, value in link.items() if key not in ("message", "cost")}
    public_link["message"] = {key: link["message"][key] for key in ("x_m", "y_m", "sample_time_s", "wind_x_m_s", "wind_y_m_s", "sigma_m_s")}
    summary = {
        "schema_version": 1,
        "parent_commit": _git(),
        "config_sha256": _sha256(CONFIG_PATH),
        "result_dir": str(out.relative_to(ROOT)).replace("\\", "/"),
        "elapsed_s": time.perf_counter() - started,
        "control": control,
        "link": public_link,
        "inspected": inspected,
        "fork_full_goal": len(feasible),
        "latest_feasible_t_s": latest,
        "online": ran_online,
        "action_change": action_change,
        "first_action_change_s": first_change,
        "mission_change": mission_change,
        "recommend_expanded_validation": recommend,
        "default_changed": False,
        "conclusion": conclusion,
    }
    _write_csv(out / "timeline.csv", played["rows"])
    _write_csv(out / "forks.csv", [_public_outcome(row) for row in [baseline_choice, *forks]])
    _write_csv(out / "online.csv", online_rows)
    _write_csv(out / "arms.csv", [_public_outcome(row) for row in arms])
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(conclusion, flush=True)
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
