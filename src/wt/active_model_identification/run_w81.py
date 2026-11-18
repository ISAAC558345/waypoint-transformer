"""Trace the packets lost by the W8 waypoint and replay the other candidate.

baseline reuse: the original action leaves the frozen W8 delivery or energy.

chosen reuse: service:1 leaves the frozen W8 delivery, energy, or wind error.

budget timeline: a candidate sends a different message, or a score uses the remote sample before 90 seconds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

from wt.active_model_identification.run_w1 import _label, _parse
from wt.active_model_identification.run_w4 import _condition, assess, load_w4
from wt.active_model_identification.run_w5 import _draw_deltas, _ripple
from wt.active_model_identification.run_w7 import _case_for, _fly
from wt.active_model_identification.run_w8 import (
    Field,
    _choose,
    _continue,
    _endpoint,
    _forecast,
    _kernel,
    _key,
    _merge_poses,
    _noise,
    _observe,
    _plan,
    _requested,
    _state,
    _truth,
)
from wt.active_model_identification.spec import ROOT
from wt.active_model_identification.wind import mean_wind_fn, message_cost_j
from wt.active_model_identification.world import mission_terminal
from wt.active_model_identification import run_w4 as run_w4_mod
from wt.active_model_identification import world as world_mod
from wt.active_model_identification.policies import selectable_candidates

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w81.json"
RUN_ID = "w81_20261118"


def _parent() -> str:
    try:
        return subprocess.check_output(["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _load() -> tuple[dict, str]:
    raw = CONFIG_PATH.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("status") != "registered_before_run":
        raise SystemExit("registration moved")
    return protocol, hashlib.sha256(raw).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _on_time(packet, mission: float) -> bool:
    return packet.holder == "shore" and packet.delivered_s is not None and float(packet.delivered_s) <= float(packet.deadline_s) + 1e-9 and float(packet.delivered_s) <= mission + 1e-9


class _Trace:
    def __init__(self) -> None:
        self.record = False
        self.arm = ""
        self.action: dict[int, str] = {}
        self.events: list[dict] = []
        self.actions: dict[tuple[str, int, int], str] = {}
        self.world = None
        self.real_move = world_mod._move
        self.real_collect = world_mod._collect_bulk
        self.real_backhaul = world_mod._backhaul
        self.real_advance = run_w4_mod.advance

    def install(self) -> None:
        holder = self

        def advance(world, *args, **kwargs):
            holder.world = world
            return holder.real_advance(world, *args, **kwargs)

        def move(uav, action, spec, theta, phase_s, wind_xy=(0.0, 0.0)):
            holder.real_move(uav, action, spec, theta, phase_s, wind_xy)
            if not holder.record or holder.world is None:
                return
            second = int(round(float(holder.world.t))) + 1
            label = _label(action)
            holder.action[int(uav.index)] = label
            holder.actions[(holder.arm, second, int(uav.index))] = label

        def collect(world, uav, buoy_id, spec):
            before = {packet.id: packet.holder for packet in world.packets}
            holder.real_collect(world, uav, buoy_id, spec)
            if not holder.record:
                return
            second = int(round(float(world.t))) + 1
            for packet in world.packets:
                if before[packet.id] == "buoy" and packet.holder == f"uav:{int(uav.index)}":
                    holder.events.append(
                        {
                            "arm": holder.arm,
                            "kind": "collect",
                            "t_s": second,
                            "id": packet.id,
                            "buoy": int(packet.buoy),
                            "bits": float(packet.bits),
                            "uav": int(uav.index),
                            "action": holder.action.get(int(uav.index), ""),
                            "delivered_s": "",
                        }
                    )

        def backhaul(world, uav, spec, step_end):
            before = {packet.id: (packet.holder, float(packet.remaining_bits), packet.delivered_s) for packet in world.packets}
            holder.real_backhaul(world, uav, spec, step_end)
            if not holder.record:
                return
            second = int(round(float(world.t))) + 1
            for packet in world.packets:
                old = before[packet.id]
                now = (packet.holder, float(packet.remaining_bits), packet.delivered_s)
                if old == now:
                    continue
                delivered = packet.holder == "shore" and old[0] != "shore"
                if delivered and not (second - 1.0 - 1e-6 <= float(packet.delivered_s) <= second + 1e-6):
                    raise RuntimeError(f"{packet.id} arrived at {packet.delivered_s} outside second {second}")
                holder.events.append(
                    {
                        "arm": holder.arm,
                        "kind": "deliver" if delivered else "backhaul",
                        "t_s": second,
                        "id": packet.id,
                        "buoy": int(packet.buoy),
                        "bits": float(packet.bits),
                        "uav": int(uav.index),
                        "action": holder.action.get(int(uav.index), ""),
                        "delivered_s": "" if packet.delivered_s is None else float(packet.delivered_s),
                    }
                )

        world_mod._move = move
        world_mod._collect_bulk = collect
        world_mod._backhaul = backhaul
        run_w4_mod.advance = advance

    def restore(self) -> None:
        world_mod._move = self.real_move
        world_mod._collect_bulk = self.real_collect
        world_mod._backhaul = self.real_backhaul
        run_w4_mod.advance = self.real_advance


def _ledger(world, events: list[dict], mission: float) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for event in events:
        row = rows.setdefault(event["id"], {"id": event["id"]})
        if event["kind"] == "collect" and "collected_s" not in row:
            row["collected_s"] = int(event["t_s"])
            row["collector"] = int(event["uav"])
            row["collect_action"] = event["action"]
        if event["kind"] in ("backhaul", "deliver") and "forward_s" not in row:
            row["forward_s"] = int(event["t_s"])
            row["forward_uav"] = int(event["uav"])
            row["forward_action"] = event["action"]
        if event["kind"] == "deliver":
            row["delivered_event_s"] = float(event["delivered_s"])
    for packet in world.packets:
        row = rows.setdefault(packet.id, {"id": packet.id})
        row.update(
            {
                "buoy": int(packet.buoy),
                "bits": float(packet.bits),
                "created_s": float(packet.created_s),
                "deadline_s": float(packet.deadline_s),
                "final_holder": packet.holder,
                "delivered_s": "" if packet.delivered_s is None else float(packet.delivered_s),
                "on_time": int(_on_time(packet, mission)),
            }
        )
    return rows


def _diff_rows(reference: dict[str, dict], other: dict[str, dict], arm: str) -> list[dict]:
    rows = []
    for packet_id in sorted(set(reference) | set(other)):
        left = reference[packet_id]
        right = other[packet_id]
        changed = any(left.get(key) != right.get(key) for key in ("collected_s", "collector", "forward_s", "forward_uav", "delivered_s", "on_time", "final_holder"))
        if not changed:
            continue
        if int(left["on_time"]) == 1 and int(right["on_time"]) == 0:
            status = "missing"
        elif int(left["on_time"]) == 0 and int(right["on_time"]) == 1:
            status = "added"
        else:
            status = "shifted"
        rows.append(
            {
                "alternative": arm,
                "status": status,
                "packet": packet_id,
                "buoy": left["buoy"],
                "bits": left["bits"],
                "created_s": left["created_s"],
                "baseline_collected_s": left.get("collected_s", ""),
                "baseline_collector": left.get("collector", ""),
                "baseline_collect_action": left.get("collect_action", ""),
                "baseline_forward_s": left.get("forward_s", ""),
                "baseline_forward_action": left.get("forward_action", ""),
                "baseline_delivered_s": left.get("delivered_s", ""),
                "baseline_on_time": left["on_time"],
                "baseline_holder": left["final_holder"],
                "other_collected_s": right.get("collected_s", ""),
                "other_collector": right.get("collector", ""),
                "other_collect_action": right.get("collect_action", ""),
                "other_forward_s": right.get("forward_s", ""),
                "other_forward_action": right.get("forward_action", ""),
                "other_delivered_s": right.get("delivered_s", ""),
                "other_on_time": right["on_time"],
                "other_holder": right["final_holder"],
                "deadline_s": left["deadline_s"],
            }
        )
    return rows


def _milestones(events: list[dict], packet_id: str) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for event in events:
        if event["id"] != packet_id:
            continue
        if event["kind"] == "collect" and "collect" not in found:
            found["collect"] = event
        elif event["kind"] == "backhaul" and "forward" not in found:
            found["forward"] = event
        elif event["kind"] == "deliver" and "deliver" not in found:
            found["deliver"] = event
    return found


def _earliest(reference_events: list[dict], other_events: list[dict], diff: list[dict]) -> dict | None:
    found = []
    for row in diff:
        if row["status"] not in ("missing", "added"):
            continue
        left = _milestones(reference_events, row["packet"])
        right = _milestones(other_events, row["packet"])
        for kind in ("collect", "forward", "deliver"):
            base = left.get(kind)
            alt = right.get(kind)
            same = base is not None and alt is not None and int(base["t_s"]) == int(alt["t_s"]) and int(base["uav"]) == int(alt["uav"])
            if same:
                continue
            stamp = min(int(item["t_s"]) for item in (base, alt) if item is not None)
            actor = base if base is not None and int(base["t_s"]) == stamp else alt
            found.append(
                {
                    "t_s": stamp,
                    "kind": kind,
                    "packet": row["packet"],
                    "status": row["status"],
                    "bits": row["bits"],
                    "baseline_action": "" if base is None else base["action"],
                    "other_action": "" if alt is None else alt["action"],
                    "baseline_uav": "" if base is None else int(base["uav"]),
                    "other_uav": "" if alt is None else int(alt["uav"]),
                    "actor_action": "" if actor is None else actor["action"],
                    "actor_uav": "" if actor is None else int(actor["uav"]),
                }
            )
            break
    if not found:
        return None
    found.sort(key=lambda item: (int(item["t_s"]), item["packet"]))
    first = found[0]["t_s"]
    same_step = [item for item in found if int(item["t_s"]) == int(first)]
    return {"t_s": first, "packets": same_step, "bits": sum(float(item["bits"]) for item in same_step)}


def _score(base_poses: dict, remote_poses: dict, query: dict, latent: Field, seed: str, protocol: dict, kernels: dict, condition: dict, field_cfg: dict) -> dict[int, tuple[float, float]]:
    clock = protocol["timeline"]
    samples = []
    for index, sample_t in enumerate((0, int(clock["choose_s"]), int(clock["sample_s"]), int(clock["usable_s"]))):
        x_m, y_m = base_poses[(sample_t, 1)]
        epsilon = _noise(seed, "onboard", sample_t, float(protocol["observation_sigma_m_s"]))
        samples.append(_observe(condition, field_cfg, latent, x_m, y_m, float(sample_t), epsilon, 100 + index))
    x_m, y_m = remote_poses[(int(clock["sample_s"]), 0)]
    epsilon = _noise(seed, "remote", int(clock["sample_s"]), float(protocol["observation_sigma_m_s"]))
    samples.append(_observe(condition, field_cfg, latent, x_m, y_m, float(clock["sample_s"]), epsilon, 200))
    errors = {}
    prior = {"global": protocol["prior_weight_global"], "local": protocol["prior_weight_local"]}
    for horizon, target_t in zip(clock["horizons_s"], clock["score_s"]):
        if int(target_t) < int(clock["usable_s"]):
            raise RuntimeError("score precedes the usable message")
        predicted = query[int(target_t)]
        ripple = _ripple(condition, field_cfg, predicted[0], predicted[1], float(target_t))
        forecast, _weights = _forecast(samples, {"id": -1, "x": predicted[0], "y": predicted[1], "t": float(target_t)}, kernels, prior, ripple)
        actual = base_poses[(int(target_t), 1)]
        truth = _truth(condition, field_cfg, latent, actual[0], actual[1], float(target_t))
        errors[int(horizon)] = (forecast[0] - truth[0], forecast[1] - truth[1])
    return errors


def _mission(world, spec: dict) -> dict:
    mission = float(spec["time"]["mission_s"])
    outcome = assess(world, spec, set(), 3, 0.0)
    terminals = mission_terminal(world, spec)
    times = [float(packet.delivered_s) for packet in world.packets if _on_time(packet, mission)]
    reserve = float(spec["energy"]["reserve_j"])
    return {
        "bits": float(outcome["on_time_bits"]),
        "packets": len(times),
        "last_s": None if not times else max(times),
        "energy_j": float(outcome["energy_j"]),
        "returned": [bool(item["returned"]) for item in terminals],
        "energy_left_j": [float(item["energy_j"]) for item in terminals],
        "reserve_margin_j": [float(item["energy_j"]) - reserve for item in terminals],
        "reserve_j": reserve,
    }


def _balance(rows: list[dict], protocol: dict) -> list[str]:
    fails = []
    bits_tol = float(protocol["bits_tolerance"])
    energy_tol = float(protocol["energy_tolerance_j"])
    rmse_tol = float(protocol["rmse_tolerance_m_s"])
    for row in rows:
        if float(row["bits"]) + bits_tol < float(row["baseline_bits"]):
            fails.append(f"{row['field']}:{row['seed']}:bits")
        if int(row["packets"]) < int(row["baseline_packets"]):
            fails.append(f"{row['field']}:{row['seed']}:packets")
        if not all(bool(item) for item in row["returned"]):
            fails.append(f"{row['field']}:{row['seed']}:return")
        if float(row["energy_j"]) > float(row["baseline_energy_j"]) + energy_tol:
            fails.append(f"{row['field']}:{row['seed']}:energy")
        if int(row["bytes"]) > int(row["baseline_bytes"]):
            fails.append(f"{row['field']}:{row['seed']}:bytes")
        if row["field"] != "local":
            continue
        for horizon in (30, 60):
            if float(row[f"rmse_{horizon}"]) > float(row[f"baseline_rmse_{horizon}"]) + rmse_tol:
                fails.append(f"{row['seed']}:h{horizon}")
        improved = any(float(row[f"rmse_{horizon}"]) + rmse_tol < float(row[f"baseline_rmse_{horizon}"]) for horizon in (30, 60))
        if not improved:
            fails.append(f"{row['seed']}:no_gain")
    return fails


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    protocol, digest = _load()
    out = out_dir or (ROOT / "results" / "wind_observation_w81" / RUN_ID)
    if "wind_observation_w81" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w81")
    archived = json.loads((ROOT / str(protocol["w8_result"])).read_text(encoding="utf-8"))
    w8 = json.loads((ROOT / "configs" / "experiments" / "wind_observation_w8.json").read_text(encoding="utf-8"))
    w7 = json.loads((ROOT / str(w8["w7_result"])).read_text(encoding="utf-8"))
    w7_cases = {row["scenario_id"]: row for row in w7["scenarios"]}
    frozen_tasks = [row for row in archived["tasks"] if row["scenario_id"] == protocol["scenario_id"]]
    frozen_base = frozen_tasks[0]
    _cfg, archive = load_w4()
    condition = _condition(archive, str(w8["condition"]))
    wind_at = mean_wind_fn(None, archive["cfg"]["field"], float(w8["prior_std_m_s"]))
    kernels = {"global": _kernel(w8, 0.0), "local": _kernel(w8, w8["local_field"]["amplitude_m_s"])}
    clock = w8["timeline"]
    points: dict[tuple, dict] = {}
    scenarios = []
    trace = _Trace()
    trace.install()
    try:
        for scenario_id in [row["scenario_id"] for row in w7["scenarios"]]:
            case, _manifest = _case_for(scenario_id)
            case = dict(case)
            case["condition_row"] = condition
            trace.record = scenario_id == protocol["scenario_id"]
            trace.arm = "service:0"
            print(f"scenario {scenario_id}", flush=True)
            played, poses, _committed = _fly(case, condition)
            trace.record = False
            outcome = assess(played["world"], case["spec"], set(), 3, 0.0)
            reference = w7_cases[scenario_id]
            if abs(float(outcome["on_time_bits"]) - float(reference["bits"])) > 1e-4 or abs(float(outcome["energy_j"]) - float(reference["energy_j"])) > 1e-3:
                raise RuntimeError(f"{scenario_id} left the W8 baseline")
            state = _state(played, int(clock["choose_s"]))
            committed = [_requested(played, int(clock["choose_s"]), index) for index in range(2)]
            positions, _air = _plan(state, committed, case, wind_at, str(w8["method"]), 4)

            def add_point(x: float, y: float, t: float) -> None:
                key = _key(x, y, t)
                points.setdefault(key, {"key": key, "x": float(x), "y": float(y), "t": float(t)})

            for t_s in (0, int(clock["choose_s"]), int(clock["sample_s"]), int(clock["usable_s"])):
                add_point(poses[(t_s, 1)][0], poses[(t_s, 1)][1], t_s)
            add_point(poses[(int(clock["sample_s"]), 0)][0], poses[(int(clock["sample_s"]), 0)][1], int(clock["sample_s"]))
            for t_s in clock["score_s"]:
                add_point(poses[(int(t_s), 1)][0], poses[(int(t_s), 1)][1], int(t_s))
                add_point(positions[int(t_s)][0], positions[int(t_s)][1], int(t_s))
            legal, _mapping = selectable_candidates(case["spec"], state["world"], state["beliefs"], 0, include_calibrate=False, factor=float(case["factor"]))
            for action in list(legal) + [committed[0]]:
                endpoint = _endpoint(state["world"].uavs[0], action, state["beliefs"][0], case["spec"], wind_at, float(clock["choose_s"]))
                add_point(endpoint[0], endpoint[1], int(clock["sample_s"]))
            scenarios.append({"id": scenario_id, "case": case, "played": played, "poses": poses, "plan": positions, "state": state})
        focus = next(item for item in scenarios if item["id"] == protocol["scenario_id"])
        drawn = _draw_deltas(list(points.values()), kernels["local"], [int(seed) for seed in w8["local_field"]["seeds"]], float(w8["local_field"]["generation_jitter"]))
        fields = [("control", "control", Field(list(points.values()), None, kernels["local"], "control", 0.0))]
        for seed, table in drawn.items():
            values = {point["key"]: table[point["key"]] for point in points.values()}
            fields.append(("local", str(seed), Field(list(points.values()), values, kernels["local"], str(seed), float(w8["local_field"]["generation_jitter"]))))
        choices = []
        for field_name, seed, latent in fields:
            before = len(latent.points)
            choice = _choose(focus["case"], focus["played"], focus["poses"], w8, wind_at, kernels, latent, seed)
            if len(latent.points) != before:
                raise RuntimeError("candidate scoring read a wind value outside the registered draw")
            frozen = next(row for row in frozen_tasks if row["field"] == field_name and row["seed"] == seed)
            got = [float(item["score"]) for item in choice["candidates"]]
            labels = [item["action"] for item in choice["candidates"]]
            if set(labels) != set(protocol["actions"]) or labels[0] != protocol["reference_action"]:
                raise RuntimeError(f"candidate list changed: {labels}")
            if any(abs(left - right) > 1e-6 for left, right in zip(got, frozen["scores"])):
                raise RuntimeError(f"{field_name} {seed} variance left the W8 score")
            choices.append({"field": field_name, "seed": seed, "latent": latent, "scores": {item["action"]: float(item["score"]) for item in choice["candidates"]}})
        state = focus["state"]
        saved_world = state["world"].snapshot()
        saved_beliefs = [item.copy() for item in state["beliefs"]]
        companion = _requested(focus["played"], int(clock["choose_s"]), 1)
        flights = {"service:0": {"played": focus["played"], "poses": focus["poses"]}}
        for action_label in protocol["actions"][1:]:
            trace.record = True
            trace.arm = action_label
            print(f"replay {action_label}", flush=True)
            played, extra = _continue(
                focus["case"],
                condition,
                saved_world.snapshot(),
                [item.copy() for item in saved_beliefs],
                int(clock["choose_s"]) // int(w8["decision_s"]),
                [_parse(action_label), companion],
            )
            trace.record = False
            flights[action_label] = {"played": played, "poses": _merge_poses(focus["poses"], extra)}
        mission = float(focus["case"]["spec"]["time"]["mission_s"])
        prefix = [event for event in trace.events if event["arm"] == "service:0" and int(event["t_s"]) <= int(clock["choose_s"])]
        ledgers = {}
        missions = {}
        for action_label, flight in flights.items():
            later = [event for event in trace.events if event["arm"] == action_label and int(event["t_s"]) > int(clock["choose_s"])]
            shared = prefix if action_label != "service:0" else [event for event in trace.events if event["arm"] == "service:0"]
            ledgers[action_label] = _ledger(flight["played"]["world"], shared + later if action_label != "service:0" else shared, mission)
            missions[action_label] = _mission(flight["played"]["world"], focus["case"]["spec"])
        base_mission = missions["service:0"]
        chosen_mission = missions["service:1"]
        if abs(base_mission["bits"] - float(frozen_base["bits_baseline"])) > float(protocol["bits_tolerance"]) or base_mission["packets"] != int(frozen_base["packets_baseline"]):
            raise RuntimeError("baseline packet count left W8")
        scenario_energy = float(w7_cases[protocol["scenario_id"]]["energy_j"])
        if abs(base_mission["energy_j"] - float(scenario_energy)) > float(protocol["energy_tolerance_j"]):
            raise RuntimeError("baseline energy left W8")
        if abs(chosen_mission["bits"] - float(frozen_base["bits_experiment"])) > float(protocol["bits_tolerance"]) or chosen_mission["packets"] != int(frozen_base["packets_experiment"]):
            raise RuntimeError("chosen delivery left W8")
        if abs(chosen_mission["energy_j"] - (float(scenario_energy) + float(frozen_base["extra_energy_j"]))) > float(protocol["energy_tolerance_j"]):
            raise RuntimeError("chosen energy left W8")
        airtime = float(message_cost_j(int(w8["payload_bytes"]))["airtime_s"])
        if airtime <= 0.0 or airtime >= float(clock["usable_s"]) - float(clock["send_s"]):
            raise RuntimeError("airtime does not fit the registered timeline")
        messages = 1
        payload = int(w8["payload_bytes"])
        with (ROOT / str(protocol["w8_pairs"])).open(encoding="utf-8") as handle:
            frozen_pairs = list(csv.DictReader(handle))
        pair_index = {
            (row["field"], row["seed"], row["arm"], int(row["horizon_s"])): (float(row["dx"]), float(row["dy"]))
            for row in frozen_pairs
            if row["scenario_id"] == protocol["scenario_id"]
        }
        arm_name = {"service:0": "baseline", "service:1": "experiment"}
        candidate_rows = []
        for choice in choices:
            for action_label in protocol["actions"]:
                errors = _score(
                    focus["poses"],
                    flights[action_label]["poses"],
                    focus["plan"],
                    choice["latent"],
                    choice["seed"],
                    w8,
                    kernels,
                    condition,
                    focus["case"]["cfg"]["field"],
                )
                if action_label in arm_name:
                    for horizon, error in errors.items():
                        frozen_error = pair_index[(choice["field"], choice["seed"], arm_name[action_label], horizon)]
                        if abs(error[0] - frozen_error[0]) > float(protocol["rmse_tolerance_m_s"]) or abs(error[1] - frozen_error[1]) > float(protocol["rmse_tolerance_m_s"]):
                            raise RuntimeError(f"{action_label} {choice['seed']} h{horizon} left the W8 forecast")
                mission_row = missions[action_label]
                baseline_errors = None
                candidate_rows.append(
                    {
                        "field": choice["field"],
                        "seed": choice["seed"],
                        "action": action_label,
                        "variance": choice["scores"][action_label],
                        "errors": errors,
                        "mission": mission_row,
                        "messages": messages,
                        "bytes": messages * payload,
                    }
                )
                baseline_errors = errors if action_label == "service:0" else baseline_errors
        by_key = {(row["field"], row["seed"], row["action"]): row for row in candidate_rows}
        flat = []
        grouped: dict[str, list[dict]] = {}
        for row in candidate_rows:
            base = by_key[(row["field"], row["seed"], "service:0")]
            flat_row = {
                "field": row["field"],
                "seed": row["seed"],
                "action": row["action"],
                "variance": row["variance"],
                "baseline_variance": base["variance"],
                "messages": row["messages"],
                "bytes": row["bytes"],
                "baseline_bytes": base["bytes"],
                "bits": row["mission"]["bits"],
                "baseline_bits": base["mission"]["bits"],
                "packets": row["mission"]["packets"],
                "baseline_packets": base["mission"]["packets"],
                "last_s": row["mission"]["last_s"],
                "baseline_last_s": base["mission"]["last_s"],
                "energy_j": row["mission"]["energy_j"],
                "baseline_energy_j": base["mission"]["energy_j"],
                "returned": row["mission"]["returned"],
                "energy_left_j": row["mission"]["energy_left_j"],
                "reserve_margin_j": row["mission"]["reserve_margin_j"],
            }
            for horizon in (30, 60):
                flat_row[f"rmse_{horizon}"] = (row["errors"][horizon][0] ** 2 + row["errors"][horizon][1] ** 2) ** 0.5
                flat_row[f"baseline_rmse_{horizon}"] = (base["errors"][horizon][0] ** 2 + base["errors"][horizon][1] ** 2) ** 0.5
                flat_row[f"d_rmse_{horizon}"] = flat_row[f"rmse_{horizon}"] - flat_row[f"baseline_rmse_{horizon}"]
            flat_row["d_bits"] = flat_row["bits"] - flat_row["baseline_bits"]
            flat_row["d_packets"] = flat_row["packets"] - flat_row["baseline_packets"]
            flat_row["d_energy_j"] = flat_row["energy_j"] - flat_row["baseline_energy_j"]
            grouped.setdefault(row["action"], []).append(flat_row)
            flat.append(flat_row)
        for action_label, rows in grouped.items():
            reasons = [] if action_label == "service:0" else list(dict.fromkeys(_balance(rows, protocol)))
            for row in rows:
                row["balances"] = int(action_label != "service:0" and not reasons)
                row["failed"] = "" if action_label == "service:0" else ";".join(reasons)
        packet_rows = []
        earliest = {}
        for action_label in protocol["actions"][1:]:
            other_events = prefix + [event for event in trace.events if event["arm"] == action_label and int(event["t_s"]) > int(clock["choose_s"])]
            base_events = [event for event in trace.events if event["arm"] == "service:0"]
            diff = _diff_rows(ledgers["service:0"], ledgers[action_label], action_label)
            packet_rows.extend(diff)
            earliest[action_label] = _earliest(base_events, other_events, diff)
        first_action = None
        for second in range(int(clock["choose_s"]) + 1, int(mission) + 1):
            left = [trace.actions.get(("service:0", second, uav), "") for uav in (0, 1)]
            right = [trace.actions.get(("service:1", second, uav), "") for uav in (0, 1)]
            if left != right:
                first_action = {"t_s": second, "baseline": left, "chosen": right}
                break
        out.mkdir(parents=True, exist_ok=True)
        _write_csv(out / "packets.csv", packet_rows)
        export = []
        for row in flat:
            export.append({key: (json.dumps(value) if isinstance(value, list) else value) for key, value in row.items()})
        _write_csv(out / "candidates.csv", export)
        summary = {
            "schema_version": 1,
            "parent_commit": _parent(),
            "config_sha256": digest,
            "config_status": protocol["status"],
            "baseline_reuse": True,
            "chosen_reuse": True,
            "budget_timeline": True,
            "default_changed": False,
            "scenario_id": protocol["scenario_id"],
            "airtime_s": airtime,
            "messages": messages,
            "bytes": messages * payload,
            "forwarding": protocol["forwarding"],
            "first_action_difference_s": first_action,
            "earliest": earliest,
            "candidates": flat,
            "packet_diff_counts": {
                action_label: {
                    "missing": sum(1 for row in packet_rows if row["alternative"] == action_label and row["status"] == "missing"),
                    "added": sum(1 for row in packet_rows if row["alternative"] == action_label and row["status"] == "added"),
                    "shifted": sum(1 for row in packet_rows if row["alternative"] == action_label and row["status"] == "shifted"),
                }
                for action_label in protocol["actions"][1:]
            },
            "elapsed_s": time.perf_counter() - started,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"first_action": first_action, "earliest": earliest, "counts": summary["packet_diff_counts"]}, ensure_ascii=False), flush=True)
        return summary
    finally:
        trace.restore()


if __name__ == "__main__":
    run()
