"""Rolling re-decision of the W1 wind message.

The quote is still the W1 full script. Only the current 30 s is flown,
then B chooses again. A keeps the frozen executed sequence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

from wt.active_model_identification.run_w1 import (
    ENERGY_TOL,
    _choose,
    _feasible,
    _incremental,
    _label,
    _parse,
    load_case,
    replay_case,
)
from wt.active_model_identification.scenarios import observation_noise
from wt.active_model_identification.spec import ROOT, decision_count
from wt.active_model_identification.wind import (
    _keep,
    adverse_wind_fn,
    charge_message,
    fly_script,
    make_return_guard,
    mean_wind_fn,
    message_cost_j,
    observe,
    script_action,
    true_wind,
)
from wt.active_model_identification.world import advance, energy_used_j, mission_terminal, on_time_bits, pad_distance_m, pad_radius_m, utility

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w2.json"
W1_DIR = ROOT / "results" / "wind_observation_w1" / "w1_20261114"
RUN_ID = "w2_20261115"
LOCKED = (
    "scenario_id",
    "method",
    "seed",
    "observer_uav",
    "decider_uav",
    "sample_epoch",
    "sample_time_s",
    "decision_epoch",
    "decision_time_s",
    "late_time_s",
    "payload_bytes",
    "observation_sigma_m_s",
    "prior_std_m_s",
    "reserve_j",
    "field",
    "conditions",
    "modes",
)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def load_w2() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = load_case()["cfg"]
    for key in LOCKED:
        if cfg[key] != parent[key]:
            raise SystemExit(f"W2 registration changed locked field {key}")
    return cfg


def load_w1_archive() -> dict:
    candidates: dict[tuple[str, str], list[dict]] = {}
    for row in _read_csv(W1_DIR / "candidates.csv"):
        candidates.setdefault((row["condition"], row["mode"]), []).append(row)
    scenarios = {(row["condition"], row["mode"]): row for row in _read_csv(W1_DIR / "scenarios.csv")}
    return {"candidates": candidates, "scenarios": scenarios}


def visible(mode: str, t: float, cfg: dict) -> bool:
    if mode == "no_send":
        return False
    if mode == "timely":
        return t + ENERGY_TOL >= float(cfg["decision_time_s"])
    if mode == "late":
        return t + ENERGY_TOL >= float(cfg["late_time_s"])
    raise KeyError(mode)


def _same_number(actual, expected, tol: float) -> bool:
    return abs(float(actual) - float(expected)) <= tol


def _flag(value) -> bool:
    return value is True or str(value) == "True"


def compare_opening(rows: list[dict], archive_rows: list[dict]) -> list[str]:
    found = []
    by_action = {row["action"]: row for row in archive_rows}
    actual = {row["action"] for row in rows}
    if actual != set(by_action):
        found.append(f"actions {sorted(actual)} != {sorted(by_action)}")
    for row in rows:
        ref = by_action.get(row["action"])
        if ref is None:
            continue
        prefix = row["action"]
        if _flag(row["admitted"]) != _flag(ref["admitted"]):
            found.append(f"{prefix} admitted {row['admitted']} != {ref['admitted']}")
        if not _same_number(row["utility"], ref["utility"], 1e-9):
            found.append(f"{prefix} utility {row['utility']} != {ref['utility']}")
        if not _same_number(row["score_bits"], ref["score_bits"], 1e-6):
            found.append(f"{prefix} bits {row['score_bits']} != {ref['score_bits']}")
        if not _same_number(row["score_energy_j"], ref["score_energy_j"], 1e-6):
            found.append(f"{prefix} score_energy {row['score_energy_j']} != {ref['score_energy_j']}")
        if not _same_number(row["admit_energy_j"], ref["admit_energy_j"], 1e-6):
            found.append(f"{prefix} admit_energy {row['admit_energy_j']} != {ref['admit_energy_j']}")
        if bool(row["chosen"]) != _flag(ref["chosen"]):
            found.append(f"{prefix} chosen {row['chosen']} != {ref['chosen']}")
    return found


def score_state(world, beliefs, info, spec, factor, scenario, decider: int, field: dict, prior: float, condition: dict, frozen: str) -> tuple[dict, list[dict]]:
    """Quote every current candidate with the W1 full script, then fly nothing."""
    actions = _feasible(spec, world, decider)
    if ("return", None) not in actions:
        raise SystemExit("return left the candidate set")
    origin = (world.uavs[decider].x, world.uavs[decider].y)
    home = (world.uavs[decider].home_x, world.uavs[decider].home_y)
    mean_fn = mean_wind_fn(info, field, prior)
    harsh_fn, _std = adverse_wind_fn(info, field, prior, origin, home)
    true_thetas = list(scenario["theta"])
    theta_mean = beliefs[decider].mean()
    theta_q = beliefs[decider].quantile(float(spec["belief"]["guard_probability"]))
    battery = world.uavs[decider].energy_j
    reserve = float(spec["energy"]["reserve_j"])
    noise = observation_noise(int(scenario["seed"]), float(spec["belief"]["sigma_j"]))
    scored = []
    rows = []
    sealed = [uav.consumed_j for uav in world.uavs]
    for action in actions:
        thetas = list(true_thetas)
        thetas[decider] = theta_mean
        quote = fly_script(world, beliefs, spec, factor, decider, action, thetas, mean_fn, noise, sentinel_battery=True)
        thetas_q = list(true_thetas)
        thetas_q[decider] = theta_q
        admission = fly_script(world, beliefs, spec, factor, decider, action, thetas_q, harsh_fn, noise, sentinel_battery=True)
        admitted = action[0] == "return" or (
            admission["landed"]
            and admission["on_time"]
            and not admission["failed"]
            and battery + ENERGY_TOL >= admission["focus_energy_j"] + reserve
            and (action[0] != "service" or admission["on_time_bits"] > 0.0)
        )
        label = _label(action)
        row = {
            "action": label,
            "admitted": admitted,
            "utility": _incremental(quote["on_time_bits"], quote["focus_energy_j"], spec),
            "score_bits": quote["on_time_bits"],
            "score_energy_j": quote["focus_energy_j"],
            "admit_energy_j": admission["focus_energy_j"],
            "admit_landed": admission["landed"],
            "admit_bits": admission["on_time_bits"],
            "chosen": False,
        }
        scored.append({"action": label, "admitted": admitted, "utility": row["utility"], "row": row})
        rows.append(row)
    if [uav.consumed_j for uav in world.uavs] != sealed:
        raise SystemExit("quote replay changed the live ledger")
    chosen = _choose(scored, frozen, float(spec["integral"]["tie_abs"]))
    for row in rows:
        row["chosen"] = row["action"] == chosen["action"]
    return chosen, rows


def _cache_ids(world, decider: int) -> list[str]:
    return [packet.id for packet in world.packets if packet.holder == f"uav:{decider}"]


def _landed(uav, spec: dict) -> bool:
    return (not uav.airborne) and pad_distance_m(uav) <= pad_radius_m(spec)


def _step(world, beliefs, spec, factor, scenario, decider: int, observer: int, action, epoch: int, observer_action, wind_at, info, field, prior: float):
    immediate = script_action(world, decider, action, spec, 0)
    requested = immediate if immediate is not None else ("wait", None)
    actions = [("wait", None) for _ in world.uavs]
    actions[decider] = requested
    actions[observer] = observer_action
    noise = observation_noise(int(scenario["seed"]), float(spec["belief"]["sigma_j"]))
    observations = advance(
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
        wind_at=wind_at,
        admit=_keep,
        return_guard=make_return_guard(info, field, prior, spec, factor, decider),
    )
    executed = tuple(observations[decider]["executed"])
    return requested, executed


def _packet_rows(world, watched: list[str], deadline: float) -> list[dict]:
    by_id = {packet.id: packet for packet in world.packets}
    rows = []
    for packet_id in watched:
        packet = by_id[packet_id]
        delivered = packet.delivered_s
        on_time = delivered is not None and delivered <= deadline + ENERGY_TOL and packet.holder == "shore"
        if on_time:
            reason = "on_time"
        elif packet.holder == "shore":
            reason = "late_delivery"
        elif packet.holder.startswith("uav:"):
            reason = "still_cached"
        else:
            reason = f"held_by_{packet.holder}"
        rows.append(
            {
                "id": packet.id,
                "bits": packet.bits,
                "holder": packet.holder,
                "delivered_s": delivered,
                "on_time": on_time,
                "reason": reason,
            }
        )
    return rows


def _block_reason(cache_ids: list[str], admitted: list[str], chosen: str, executed: str) -> str:
    if not cache_ids:
        return ""
    if "backhaul:None" not in admitted:
        return "admission"
    if chosen != "backhaul:None":
        return "utility"
    if executed != "backhaul:None":
        return "return_guard"
    return ""


def roll_arm(replay: dict, condition: dict, mode: str, message: dict, cfg: dict, *, stop_after_s: float | None = None) -> dict:
    case = replay["case"]
    spec = case["spec"]
    scenario = case["scenario"]
    factor = case["factor"]
    decider = int(cfg["decider_uav"])
    observer = int(cfg["observer_uav"])
    field = cfg["field"]
    prior = float(cfg["prior_std_m_s"])
    frozen = case["requests"][(int(cfg["decision_epoch"]), decider)]["requested"]
    world = replay["prefix"].snapshot()
    beliefs = [item.copy() for item in replay["prefix_beliefs"]]
    watched = _cache_ids(world, decider)
    if len(watched) != 2:
        raise SystemExit(f"expected the two cached packets, found {watched}")
    start_consumed = [uav.consumed_j for uav in world.uavs]
    charged = None
    decisions = []
    energy_steps = []
    mission = float(spec["time"]["mission_s"])
    decision_s = float(spec["time"]["decision_s"])
    wind_at = lambda x, y, t, condition=condition: true_wind(condition, x, y, t, field)
    while world.t < mission - ENERGY_TOL:
        if stop_after_s is not None and world.t > stop_after_s + ENERGY_TOL:
            break
        epoch = int(round(world.t / decision_s))
        if epoch >= decision_count(spec):
            break
        before = sum(uav.consumed_j for uav in world.uavs)
        if visible(mode, world.t, cfg) and charged is None:
            charged = charge_message(world, observer, decider, int(cfg["payload_bytes"]))
        info = message if visible(mode, world.t, cfg) else None
        cache_ids = _cache_ids(world, decider)
        chosen, rows = score_state(world, beliefs, info, spec, factor, scenario, decider, field, prior, condition, frozen)
        uav = world.uavs[decider]
        decision_energy = uav.energy_j
        decision_airborne = uav.airborne
        decision_landed = _landed(uav, spec)
        decision_below = decision_energy + ENERGY_TOL < float(spec["energy"]["reserve_j"])
        observer_action = _parse(case["requests"][(epoch, observer)]["executed"])
        requested, executed = _step(
            world, beliefs, spec, factor, scenario, decider, observer, _parse(chosen["action"]), epoch, observer_action, wind_at, info, field, prior
        )
        admitted = [row["action"] for row in rows if row["admitted"]]
        executed_label = _label(executed)
        decisions.append(
            {
                "condition": condition["id"],
                "mode": mode,
                "t_s": epoch * decision_s,
                "epoch": epoch,
                "used_message": info is not None,
                "candidates": [row["action"] for row in rows],
                "admitted": admitted,
                "chosen": chosen["action"],
                "selected_by": chosen["selected_by"],
                "requested": _label(requested),
                "executed": executed_label,
                "energy_j": decision_energy,
                "airborne": decision_airborne,
                "landed": decision_landed,
                "cache_ids": cache_ids,
                "below_reserve": decision_below,
                "block": _block_reason(cache_ids, admitted, chosen["action"], executed_label),
                "rows": rows,
            }
        )
        energy_steps.append(sum(item.consumed_j for item in world.uavs) - before)
    terminals = mission_terminal(world, spec)
    packets = _packet_rows(world, watched, mission)
    delivered_ids = [packet.id for packet in world.packets if packet.holder == "shore"]
    if len(delivered_ids) != len(set(delivered_ids)):
        raise SystemExit("a packet was delivered twice")
    consumed_delta = sum(uav.consumed_j - start_consumed[uav.index] for uav in world.uavs)
    if abs(consumed_delta - sum(energy_steps)) > 1e-6:
        raise SystemExit("interval energy does not add up to the arm ledger")
    nominal = message_cost_j(int(cfg["payload_bytes"]))
    if mode == "no_send" or charged is None:
        message_j = 0.0
        nominal_j = 0.0
    else:
        message_j = float(charged["sender_j_drawn"] + charged["receiver_j_drawn"])
        nominal_j = float(nominal["sender_j"] + nominal["receiver_j"])
    focus = terminals[decider]
    return {
        "condition": condition["id"],
        "mode": mode,
        "ax_m_s": condition["ax"],
        "chosen": decisions[0]["chosen"] if decisions else "",
        "selected_by": decisions[0]["selected_by"] if decisions else "",
        "on_time_bits": on_time_bits(world),
        "energy_j": sum(energy_used_j(world)),
        "utility": utility(world, spec),
        "message_j": message_j,
        "message_j_nominal": nominal_j,
        "charge_count": 0 if charged is None else 1,
        "returned": sum(int(item["returned"]) for item in terminals),
        "not_returned": sum(int(item["not_returned"]) for item in terminals),
        "not_landed": sum(int(item["not_landed"]) for item in terminals),
        "below_reserve": sum(int(item["below_reserve"]) for item in terminals),
        "energy_exhausted": sum(int(item["energy_exhausted"]) for item in terminals),
        "uav_energy_j": [item["energy_j"] for item in terminals],
        "decider_landed": (not focus["not_landed"]) and (not focus["not_returned"]),
        "decider_below_reserve": bool(focus["below_reserve"]),
        "decider_energy_j": focus["energy_j"],
        "decider_energy_exhausted": bool(focus["energy_exhausted"]),
        "decisions": decisions,
        "packets": packets,
        "first_below_reserve_s": next((row["t_s"] for row in decisions if row["below_reserve"]), mission if focus["below_reserve"] else None),
        "first_landed_s": next((row["t_s"] for row in decisions if row["landed"]), None),
        "first_block": next(({"t_s": row["t_s"], "reason": row["block"], "energy_j": row["energy_j"], "landed": row["landed"]} for row in decisions if row["block"]), None),
    }


def _archive_metrics(row: dict) -> dict:
    keys = ("on_time_bits", "energy_j", "utility", "returned", "not_returned", "not_landed", "below_reserve", "energy_exhausted")
    found = {key: float(row[key]) for key in keys}
    found["chosen"] = row["chosen"]
    return found


def _delta(left: dict, right: dict) -> dict:
    keys = ("on_time_bits", "energy_j", "utility", "returned", "not_returned", "not_landed", "below_reserve", "energy_exhausted")
    return {key: left[key] - right[key] for key in keys}


def _regressed(left: dict, right: dict) -> bool:
    """Mission regression is fewer on-time bits, a missed landing, or a worse reserve.

    The registered radio charge changes utility by about 3e-8. That charge stays
    in the utility column and is not labeled a mission regression by itself.
    """
    return (
        left["on_time_bits"] + 1e-6 < right["on_time_bits"]
        or left["not_landed"] > right["not_landed"]
        or left["below_reserve"] > right["below_reserve"]
        or left["energy_exhausted"] > right["energy_exhausted"]
    )


def _packet_lead(timely: list[dict], other: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in other}
    leads = []
    for row in timely:
        match = by_id[row["id"]]
        if row["delivered_s"] is None or match["delivered_s"] is None:
            leads.append({"id": row["id"], "lead_s": None, "timely_s": row["delivered_s"], "other_s": match["delivered_s"]})
        else:
            leads.append(
                {
                    "id": row["id"],
                    "lead_s": float(match["delivered_s"]) - float(row["delivered_s"]),
                    "timely_s": row["delivered_s"],
                    "other_s": match["delivered_s"],
                }
            )
    return leads


def _at_time(arm: dict, t: float) -> dict:
    return next(row for row in arm["decisions"] if abs(row["t_s"] - t) <= ENERGY_TOL)


def _utility_held(review: dict) -> bool:
    """Adoption still requires the original utility not to fall."""
    late = review["timely_minus_late"]["utility"]
    silent = review["timely_minus_no_send"]["utility"]
    archived = review["w2_minus_w1"]["timely"]["utility"]
    return late >= -1e-12 and silent >= -1e-12 and archived >= -1e-12


def judge(arms: list[dict], archive: dict, cfg: dict) -> tuple[list[dict], str, bool]:
    reviews = []
    for condition in cfg["conditions"]:
        group = {row["mode"]: row for row in arms if row["condition"] == condition["id"]}
        timely = group["timely"]
        late = group["late"]
        silent = group["no_send"]
        w1 = {mode: _archive_metrics(archive["scenarios"][(condition["id"], mode)]) for mode in cfg["modes"]}
        same_bits = abs(timely["on_time_bits"] - late["on_time_bits"]) <= 1e-6
        leads = _packet_lead(timely["packets"], late["packets"])
        earlier = same_bits and any(item["lead_s"] is not None and item["lead_s"] > 1e-6 for item in leads)
        bit_gain = timely["on_time_bits"] > late["on_time_bits"] + 0.5
        if _regressed(timely, late) or _regressed(timely, silent):
            level = "regression"
            reason = "及时组相对迟到组或不发送组减少了按时交付，或更差的落地、备用"
        elif bit_gain:
            level = "persists"
            reason = "提前一个周期的报文把多出的按时交付保持到了任务结束"
        elif earlier:
            level = "earlier_only"
            reason = "截止 bit 相同，及时组只是更早到岸；提前的秒数不另计效用"
        else:
            level = "no_change"
            reason = "截止 bit、落地和备用与迟到组相同"
        late_at_210 = _at_time(late, float(cfg["late_time_s"]))
        timely_at_210 = _at_time(timely, float(cfg["late_time_s"]))
        silent_at_210 = _at_time(silent, float(cfg["late_time_s"]))
        reviews.append(
            {
                "condition": condition["id"],
                "level": level,
                "reason": reason,
                "opening": {mode: group[mode]["chosen"] for mode in cfg["modes"]},
                "at_210": {
                    mode: {"chosen": group[mode] and _at_time(group[mode], float(cfg["late_time_s"]))["chosen"], "admitted": _at_time(group[mode], float(cfg["late_time_s"]))["admitted"], "used_message": _at_time(group[mode], float(cfg["late_time_s"]))["used_message"]}
                    for mode in cfg["modes"]
                },
                "first_divergence_s": {
                    "timely_minus_late": next((row["t_s"] for row, other in zip(timely["decisions"], late["decisions"]) if row["chosen"] != other["chosen"]), None),
                    "timely_minus_no_send": next((row["t_s"] for row, other in zip(timely["decisions"], silent["decisions"]) if row["chosen"] != other["chosen"]), None),
                },
                "timely_minus_late": _delta(timely, late),
                "timely_minus_no_send": _delta(timely, silent),
                "w2_minus_w1": {mode: _delta(group[mode], w1[mode]) for mode in cfg["modes"]},
                "packet_lead_vs_late_s": leads,
                "packets": {mode: group[mode]["packets"] for mode in cfg["modes"]},
                "decider_below_reserve": {mode: group[mode]["decider_below_reserve"] for mode in cfg["modes"]},
                "decider_landed": {mode: group[mode]["decider_landed"] for mode in cfg["modes"]},
                "decider_energy_j": {mode: group[mode]["decider_energy_j"] for mode in cfg["modes"]},
                "first_below_reserve_s": {mode: group[mode]["first_below_reserve_s"] for mode in cfg["modes"]},
                "first_block": {mode: group[mode]["first_block"] for mode in cfg["modes"]},
                "first_return_guard_s": {
                    mode: next((row["t_s"] for row in group[mode]["decisions"] if row["block"] == "return_guard"), None) for mode in cfg["modes"]
                },
                "reserve_met": not any(group[mode]["decider_below_reserve"] for mode in cfg["modes"] if mode == "timely"),
                "w1_chosen": {mode: w1[mode]["chosen"] for mode in cfg["modes"]},
                "at_210_detail": {"timely": timely_at_210["chosen"], "late": late_at_210["chosen"], "no_send": silent_at_210["chosen"]},
            }
        )
    adopt = all(
        row["level"] == "persists"
        and row["reserve_met"]
        and _utility_held(row)
        and not _regressed(
            next(arm for arm in arms if arm["condition"] == row["condition"] and arm["mode"] == "timely"),
            _archive_metrics(archive["scenarios"][(row["condition"], "timely")]),
        )
        for row in reviews
    )
    if adopt:
        conclusion = "两个风况里，提前一个周期的报文都把收益保持到任务结束，并且备用电量达标。该结论仍只属于这条显式实验。"
    elif any(row["level"] == "persists" for row in reviews):
        conclusion = "提前量只在部分风况把多出的按时交付保持到任务结束。另一风况或备用电量没有同时达到采用标准，默认调度不变。"
    elif any(row["level"] == "earlier_only" for row in reviews):
        conclusion = "滚动重选后，提前一个周期没有增加截止时的按时交付，只让相同的 bit 更早到岸。默认调度不变。"
    elif any(row["level"] == "regression" for row in reviews):
        conclusion = "滚动重选后，及时报文相对配对对照出现回退。默认调度不变。"
    else:
        conclusion = "滚动重选后，提前一个周期没有留下截止交付、落地或备用差别。默认调度不变。"
    return reviews, conclusion, adopt


def _public_decision(row: dict) -> dict:
    copied = dict(row)
    copied.pop("rows", None)
    for key in ("candidates", "admitted", "cache_ids"):
        copied[key] = "|".join(copied[key])
    return copied


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _public_arm(row: dict) -> dict:
    keys = (
        "condition",
        "mode",
        "ax_m_s",
        "chosen",
        "selected_by",
        "on_time_bits",
        "energy_j",
        "utility",
        "message_j",
        "message_j_nominal",
        "charge_count",
        "returned",
        "not_returned",
        "not_landed",
        "below_reserve",
        "energy_exhausted",
        "decider_landed",
        "decider_below_reserve",
        "decider_energy_j",
        "decider_energy_exhausted",
        "first_below_reserve_s",
    )
    copied = {key: row[key] for key in keys}
    copied["uav_energy_j"] = json.dumps(row["uav_energy_j"])
    return copied


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    out = out_dir or (ROOT / "results" / "wind_observation_w2" / RUN_ID)
    if "wind_observation_w2" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w2")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_w2()
    replay = replay_case()
    if not replay["ok"]:
        raise SystemExit("W1 prefix replay no longer matches M1.6")
    archive = load_w1_archive()
    print("prefix replay matches M1.6", flush=True)
    arms = []
    opening_errors = []
    for condition in cfg["conditions"]:
        rng = random.Random(f"w1-{condition['id']}-{cfg['seed']}")
        message = observe(
            condition,
            replay["sample"]["x_m"],
            replay["sample"]["y_m"],
            replay["sample"]["t_s"],
            cfg["field"],
            float(cfg["observation_sigma_m_s"]),
            rng,
        )
        for mode in cfg["modes"]:
            print(f"arm {condition['id']} {mode}", flush=True)
            arm = roll_arm(replay, condition, mode, message, cfg)
            opening = arm["decisions"][0]
            opening_errors.extend(
                f"{condition['id']} {mode} {item}" for item in compare_opening(opening["rows"], archive["candidates"][(condition["id"], mode)])
            )
            if opening_errors:
                (out / "opening_mismatch.json").write_text(json.dumps(opening_errors, ensure_ascii=False, indent=2), encoding="utf-8")
                raise SystemExit("180 s choice does not reproduce W1")
            if mode == "late" and opening["used_message"]:
                raise SystemExit("late message entered the 180 s decision")
            if mode == "no_send" and any(row["used_message"] for row in arm["decisions"]):
                raise SystemExit("no-send read a wind message")
            if mode == "late":
                at_210 = _at_time(arm, float(cfg["late_time_s"]))
                if not at_210["used_message"]:
                    raise SystemExit("late message was not read at 210 s")
            if arm["charge_count"] != (0 if mode == "no_send" else 1):
                raise SystemExit("message was charged more than once")
            arms.append(arm)
    reviews, conclusion, adopt = judge(arms, archive, cfg)
    summary = {
        "schema_version": 1,
        "parent_commit": _git(),
        "config_sha256": _sha256(CONFIG_PATH),
        "w1_candidates_sha256": _sha256(W1_DIR / "candidates.csv"),
        "result_dir": str(out.relative_to(ROOT)).replace("\\", "/"),
        "elapsed_s": time.perf_counter() - started,
        "opening_matches_w1": not opening_errors,
        "reviews": reviews,
        "conclusion": conclusion,
        "adopt": adopt,
        "default_changed": False,
    }
    _write_csv(out / "scenarios.csv", [_public_arm(row) for row in arms])
    _write_csv(out / "decisions.csv", [_public_decision(row) for arm in arms for row in arm["decisions"]])
    packet_rows = []
    for arm in arms:
        for packet in arm["packets"]:
            packet_rows.append({"condition": arm["condition"], "mode": arm["mode"], **packet})
    _write_csv(out / "packets.csv", packet_rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(conclusion, flush=True)
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
