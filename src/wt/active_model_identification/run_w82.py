"""Replay one later decision after the buoy-0 packets are already onboard.

baseline replay: continuing the original action leaves the W8.1 packets or energy.

snapshot shared: a candidate does not start from the same 60-second queue.

budget timeline: the remote sample enters a forecast before 120 seconds, or the arms send different bytes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

from wt.active_model_identification import policies as policies_mod
from wt.active_model_identification import world as world_mod
from wt.active_model_identification.belief import prior_belief
from wt.active_model_identification.policies import action_rank, selectable_candidates
from wt.active_model_identification.run_w1 import _label
from wt.active_model_identification import run_w4 as run_w4_mod
from wt.active_model_identification.run_w4 import _condition, assess, load_w4, play
from wt.active_model_identification.run_w5 import _draw_deltas, _ripple
from wt.active_model_identification.run_w7 import _case_for, _fly
from wt.active_model_identification.run_w8 import (
    Field,
    _endpoint,
    _forecast,
    _kernel,
    _key,
    _merge_poses,
    _mixture_variance,
    _noise,
    _observe,
    _plan,
    _requested,
    _state,
    _truth,
    _weights,
)
from wt.active_model_identification.run_w81 import _Trace, _diff_rows, _earliest, _ledger, _mission
from wt.active_model_identification.spec import ROOT
from wt.active_model_identification.wind import mean_wind_fn, message_cost_j

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w82.json"
RUN_ID = "w82_20261118"


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
    archived = json.loads((ROOT / str(protocol["w8_config"])).read_text(encoding="utf-8"))
    for key in ("min_separation_m", "max_alternatives", "tie_variance", "payload_bytes"):
        if protocol[key] != archived[key]:
            raise RuntimeError(f"{key} left the W8 registration")
    return protocol, hashlib.sha256(raw).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _continue_live(case: dict, condition: dict, world, beliefs: list, epoch: int, preset: list, trace: _Trace) -> tuple[dict, dict]:
    holder = {"world": world}
    poses: dict[tuple[int, int], tuple[float, float]] = {}
    original = world_mod._move

    def wrapped(uav, action, spec, theta, phase_s, wind_xy=(0.0, 0.0)):
        original(uav, action, spec, theta, phase_s, wind_xy)
        if not trace.record:
            return
        current = holder["world"]
        t_now = int(round(float(current.t)))
        poses[(t_now + 1, int(uav.index))] = (float(uav.x), float(uav.y))

    world_mod._move = wrapped
    try:
        played = play(case, condition, world, beliefs, epoch, preset)
    finally:
        world_mod._move = original
    return played, poses


def _queue(world) -> list[dict]:
    return [
        {"id": packet.id, "holder": packet.holder, "remaining_bits": float(packet.remaining_bits)}
        for packet in world.packets
    ]


def _signature(world) -> tuple:
    return tuple(sorted((packet.id, packet.holder, round(float(packet.remaining_bits), 6)) for packet in world.packets))


def _hyp(error: tuple[float, float]) -> float:
    return math.hypot(error[0], error[1])


def _watch(ledger: dict[str, dict], names: list[str]) -> list[dict]:
    rows = []
    for name in names:
        row = ledger[name]
        rows.append(
            {
                "packet": name,
                "forward_s": row.get("forward_s", ""),
                "delivered_s": row.get("delivered_s", ""),
                "on_time": row.get("on_time", 0),
                "holder": row.get("final_holder", ""),
            }
        )
    return rows


def _geometry(case: dict, played: dict, protocol: dict, wind_at, w8: dict) -> dict:
    clock = protocol["timeline"]
    choose_t = int(clock["choose_s"])
    state = _state(played, choose_t)
    world = state["world"].snapshot()
    beliefs = [item.copy() for item in state["beliefs"]]
    spec = case["spec"]
    committed = [_requested(played, choose_t, index) for index in range(2)]
    positions, airborne = _plan(state, committed, case, wind_at, str(w8["method"]), 4)
    both_air = all(bool(uav.airborne) for uav in world.uavs)
    telemetry = any(abs(item.mean() - prior_belief(spec).mean()) > 1e-6 for item in beliefs)
    receiver_ready = bool(airborne.get(int(clock["score_s"][0]), False))
    qualified = both_air and telemetry and receiver_ready
    original = committed[0]
    origin = _endpoint(world.uavs[0], original, beliefs[0], spec, wind_at, float(choose_t))
    pool = [{"action": original, "position": origin, "separation_m": 0.0, "alternative": False}]
    legal_labels = []
    rejected = []
    if qualified:
        legal, _mapping = selectable_candidates(spec, world, beliefs, 0, include_calibrate=False, factor=float(case["factor"]))
        legal_labels = [_label(action) for action in legal]
        found = []
        for action in legal:
            if action == original:
                continue
            position = _endpoint(world.uavs[0], action, beliefs[0], spec, wind_at, float(choose_t))
            gap = math.hypot(position[0] - origin[0], position[1] - origin[1])
            if gap + 1e-9 >= float(protocol["min_separation_m"]):
                found.append((gap, action_rank(action), action, position))
            else:
                rejected.append({"action": _label(action), "separation_m": gap, "reason": "inside_300_m"})
        found.sort(key=lambda item: (-item[0], item[1]))
        for gap, _rank, action, position in found[: int(protocol["max_alternatives"])]:
            pool.append({"action": action, "position": position, "separation_m": gap, "alternative": True})
        for gap, _rank, action, _position in found[int(protocol["max_alternatives"]) :]:
            rejected.append({"action": _label(action), "separation_m": gap, "reason": "beyond_cap"})
    reason = ""
    if not both_air:
        reason = "aircraft_not_airborne"
    elif not telemetry:
        reason = "belief_still_at_prior"
    elif not receiver_ready:
        reason = "receiver_not_airborne_at_first_score"
    elif not any(item["alternative"] for item in pool):
        reason = "no_separated_alternative"
    return {
        "state": state,
        "world": world,
        "beliefs": beliefs,
        "committed": committed,
        "positions": positions,
        "airborne": {str(key): value for key, value in airborne.items()},
        "qualified": qualified and reason == "",
        "reason": reason,
        "legal": legal_labels,
        "rejected": rejected,
        "pool": pool,
    }


def _variance(case: dict, poses: dict, pool: list[dict], positions: dict, latent: Field, seed: str, protocol: dict, kernels: dict, w8: dict) -> None:
    clock = protocol["timeline"]
    onboard = []
    for index, sample_t in enumerate(clock["choice_onboard_s"]):
        x_m, y_m = poses[(int(sample_t), 1)]
        epsilon = _noise(seed, "onboard", sample_t, float(w8["observation_sigma_m_s"]))
        onboard.append(_observe(case["condition_row"], case["cfg"]["field"], latent, x_m, y_m, float(sample_t), epsilon, index + 1))
    weights = _weights(onboard, kernels, {"global": w8["prior_weight_global"], "local": w8["prior_weight_local"]})
    targets = [{"id": -1, "x": positions[int(item)][0], "y": positions[int(item)][1], "t": float(item)} for item in clock["score_s"]]
    for item in pool:
        candidate = {"id": -10, "x": item["position"][0], "y": item["position"][1], "t": float(clock["sample_s"])}
        item["score"] = _mixture_variance(onboard, candidate, targets, kernels, weights)
        item["weights"] = weights


def _errors(base_poses: dict, remote_poses: dict, query: dict, latent: Field, seed: str, protocol: dict, kernels: dict, condition: dict, field_cfg: dict, w8: dict) -> dict[int, tuple[float, float]]:
    clock = protocol["timeline"]
    if int(clock["send_s"]) >= int(clock["usable_s"]) or any(int(item) < int(clock["usable_s"]) for item in clock["score_s"]):
        raise RuntimeError("score precedes the usable message")
    samples = []
    for index, sample_t in enumerate(clock["forecast_onboard_s"]):
        x_m, y_m = base_poses[(int(sample_t), 1)]
        epsilon = _noise(seed, "onboard", sample_t, float(w8["observation_sigma_m_s"]))
        samples.append(_observe(condition, field_cfg, latent, x_m, y_m, float(sample_t), epsilon, 100 + index))
    x_m, y_m = remote_poses[(int(clock["sample_s"]), 0)]
    epsilon = _noise(seed, "remote", int(clock["sample_s"]), float(w8["observation_sigma_m_s"]))
    samples.append(_observe(condition, field_cfg, latent, x_m, y_m, float(clock["sample_s"]), epsilon, 200))
    prior = {"global": w8["prior_weight_global"], "local": w8["prior_weight_local"]}
    found = {}
    for horizon, target_t in zip(clock["horizons_s"], clock["score_s"]):
        predicted = query[int(target_t)]
        ripple = _ripple(condition, field_cfg, predicted[0], predicted[1], float(target_t))
        forecast, _weights = _forecast(samples, {"id": -1, "x": predicted[0], "y": predicted[1], "t": float(target_t)}, kernels, prior, ripple)
        actual = base_poses[(int(target_t), 1)]
        truth = _truth(condition, field_cfg, latent, actual[0], actual[1], float(target_t))
        found[int(horizon)] = (forecast[0] - truth[0], forecast[1] - truth[1])
    return found


def _clone(latent: Field) -> Field:
    values = None if latent.control else dict(latent.values)
    return Field([dict(point) for point in latent.points], values, latent.kernel, latent.seed, latent.jitter)


def _gate(rows: list[dict], protocol: dict, evaluable: set[int]) -> list[str]:
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
        if not row["scored"]:
            fails.append(f"{row['seed']}:unscored")
            continue
        judged = []
        for horizon in (30, 60):
            if horizon not in evaluable:
                fails.append(f"{row['seed']}:h{horizon}_not_airborne")
                continue
            judged.append(horizon)
            if float(row[f"rmse_{horizon}"]) > float(row[f"baseline_rmse_{horizon}"]) + rmse_tol:
                fails.append(f"{row['seed']}:h{horizon}")
        if not judged:
            fails.append(f"{row['seed']}:no_airborne_horizon")
        elif not any(float(row[f"rmse_{horizon}"]) + rmse_tol < float(row[f"baseline_rmse_{horizon}"]) for horizon in judged):
            fails.append(f"{row['seed']}:no_gain")
    return fails


def _finish(out: Path, summary: dict, packets: list[dict], candidates: list[dict]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "packets.csv", packets)
    export = [{key: (json.dumps(value) if isinstance(value, (list, dict)) else value) for key, value in row.items()} for row in candidates]
    _write_csv(out / "candidates.csv", export)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"stopped": summary["stopped"], "reason": summary["stop_reason"], "passed": summary.get("passed", [])}, ensure_ascii=False), flush=True)
    return summary


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    protocol, digest = _load()
    out = out_dir or (ROOT / "results" / "wind_observation_w82" / RUN_ID)
    if "wind_observation_w82" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w82")
    w8 = json.loads((ROOT / str(protocol["w8_config"])).read_text(encoding="utf-8"))
    w81 = json.loads((ROOT / str(protocol["w81_result"])).read_text(encoding="utf-8"))
    frozen = next(row for row in w81["candidates"] if row["action"] == "service:0" and row["field"] == "control")
    _cfg, archive = load_w4()
    condition = _condition(archive, str(w8["condition"]))
    wind_at = mean_wind_fn(None, archive["cfg"]["field"], float(w8["prior_std_m_s"]))
    kernels = {"global": _kernel(w8, 0.0), "local": _kernel(w8, w8["local_field"]["amplitude_m_s"])}
    clock = protocol["timeline"]
    case, _manifest = _case_for(str(protocol["scenario_id"]))
    case = dict(case)
    case["condition_row"] = condition
    trace = _Trace()
    trace.install()
    real_select = policies_mod.select_lookahead

    def _select_without_trace(*args, **kwargs):
        previous = trace.record
        trace.record = False
        try:
            return real_select(*args, **kwargs)
        finally:
            trace.record = previous

    policies_mod.select_lookahead = _select_without_trace
    run_w4_mod.select_lookahead = _select_without_trace
    try:
        trace.record = True
        trace.arm = "baseline"
        print("baseline", flush=True)
        played, poses, _committed = _fly(case, condition)
        trace.record = False
        mission = _mission(played["world"], case["spec"])
        mission_s = float(case["spec"]["time"]["mission_s"])
        base_events = [event for event in trace.events if event["arm"] == "baseline"]
        base_ledger = _ledger(played["world"], base_events, mission_s)
        on_time_ids = sorted(packet.id for packet in played["world"].packets if packet.holder == "shore" and packet.delivered_s is not None and float(packet.delivered_s) <= mission_s + 1e-9)
        reference_ids = [name for name in protocol["watched_packets"]]
        if abs(mission["bits"] - float(frozen["bits"])) > float(protocol["bits_tolerance"]) or mission["packets"] != int(frozen["packets"]):
            raise RuntimeError("original route left the W8.1 delivery")
        if abs(mission["energy_j"] - float(frozen["energy_j"])) > float(protocol["energy_tolerance_j"]):
            raise RuntimeError("original route left the W8.1 energy")
        state = _state(played, int(clock["choose_s"]))
        onboard = [packet for packet in state["world"].packets if packet.id in reference_ids]
        packets_ready = all(packet.holder == "uav:0" for packet in onboard) and len(onboard) == len(reference_ids)
        geometry = _geometry(case, played, protocol, wind_at, w8)
        same_queue = _signature(state["world"]) == _signature(geometry["world"])
        summary = {
            "schema_version": 1,
            "parent_commit": _parent(),
            "config_sha256": digest,
            "config_status": protocol["status"],
            "baseline_replay": True,
            "snapshot_shared": bool(same_queue),
            "budget_timeline": True,
            "default_changed": False,
            "scenario_id": protocol["scenario_id"],
            "timeline": clock,
            "onboard_choice_s": clock["choice_onboard_s"],
            "onboard_forecast_s": clock["forecast_onboard_s"],
            "field_draw": protocol["field_draw"],
            "queue_at_choice": _queue(state["world"]),
            "actions_at_choice": [_label(action) for action in geometry["committed"]],
            "legal": geometry["legal"],
            "rejected": geometry["rejected"],
            "choice_epoch": [
                {"uav": int(row["uav"]), "requested": row["requested"], "executed": row["executed"]}
                for row in played["rows"]
                if int(row["epoch"]) == int(clock["choose_s"]) // int(w8["decision_s"])
            ],
            "choice_seconds": [
                {"t_s": second, "uav0": trace.actions.get(("baseline", second, 0), ""), "uav1": trace.actions.get(("baseline", second, 1), "")}
                for second in range(int(clock["choose_s"]) + 1, int(clock["sample_s"]) + 1)
            ],
            "predicted_airborne": geometry["airborne"],
            "packets_on_uav0": packets_ready,
            "watched_baseline": _watch(base_ledger, reference_ids),
            "stopped": False,
            "stop_reason": "",
            "passed": [],
            "on_time_ids": on_time_ids,
        }
        if not same_queue:
            raise RuntimeError("choice snapshot left the replayed queue")
        if not packets_ready:
            summary["stopped"] = True
            summary["stop_reason"] = "watched_packets_not_on_uav0"
            summary["elapsed_s"] = time.perf_counter() - started
            return _finish(out, summary, [], [])
        if geometry["reason"]:
            summary["stopped"] = True
            summary["stop_reason"] = geometry["reason"]
            summary["candidates_geometry"] = [_label(item["action"]) for item in geometry["pool"]]
            summary["elapsed_s"] = time.perf_counter() - started
            return _finish(out, summary, [], [])
        points: dict[tuple, dict] = {}

        def add_point(x: float, y: float, t: float) -> None:
            key = _key(x, y, t)
            points.setdefault(key, {"key": key, "x": float(x), "y": float(y), "t": float(t)})

        for sample_t in list(clock["forecast_onboard_s"]) + list(clock["score_s"]):
            add_point(poses[(int(sample_t), 1)][0], poses[(int(sample_t), 1)][1], int(sample_t))
        add_point(poses[(int(clock["sample_s"]), 0)][0], poses[(int(clock["sample_s"]), 0)][1], int(clock["sample_s"]))
        for target_t in clock["score_s"]:
            add_point(geometry["positions"][int(target_t)][0], geometry["positions"][int(target_t)][1], int(target_t))
        for item in geometry["pool"]:
            add_point(item["position"][0], item["position"][1], int(clock["sample_s"]))
        drawn = _draw_deltas(list(points.values()), kernels["local"], [int(seed) for seed in w8["local_field"]["seeds"]], float(w8["local_field"]["generation_jitter"]))
        fields = [("control", "control", Field(list(points.values()), None, kernels["local"], "control", 0.0))]
        for seed, table in drawn.items():
            values = {point["key"]: table[point["key"]] for point in points.values()}
            fields.append(("local", str(seed), Field(list(points.values()), values, kernels["local"], str(seed), float(w8["local_field"]["generation_jitter"]))))
        summary["field_points"] = len(points)
        flights = {"baseline": {"played": played, "poses": poses, "ledger": base_ledger, "mission": mission}}
        saved_world = state["world"].snapshot()
        saved_beliefs = [item.copy() for item in state["beliefs"]]
        saved_signature = _signature(saved_world)
        companion = geometry["committed"][1]
        for item in geometry["pool"]:
            if not item["alternative"]:
                continue
            label = _label(item["action"])
            trace.record = True
            trace.arm = label
            print(f"replay {label}", flush=True)
            continued, extra = _continue_live(
                case,
                condition,
                saved_world.snapshot(),
                [item.copy() for item in saved_beliefs],
                int(clock["choose_s"]) // int(w8["decision_s"]),
                [item["action"], companion],
                trace,
            )
            trace.record = False
            if _signature(continued["states"][0]["world"]) != saved_signature:
                raise RuntimeError(f"{label} left the shared snapshot")
            prefix = [event for event in base_events if int(event["t_s"]) <= int(clock["choose_s"])]
            later = [event for event in trace.events if event["arm"] == label and int(event["t_s"]) > int(clock["choose_s"])]
            ledger = _ledger(continued["world"], prefix + later, mission_s)
            flights[label] = {
                "played": continued,
                "poses": _merge_poses(poses, extra),
                "ledger": ledger,
                "mission": _mission(continued["world"], case["spec"]),
            }
        airtime = float(message_cost_j(int(protocol["payload_bytes"]))["airtime_s"])
        if airtime <= 0.0 or airtime >= float(clock["usable_s"]) - float(clock["send_s"]):
            raise RuntimeError("airtime does not fit the registered timeline")
        messages = 1
        payload = messages * int(protocol["payload_bytes"])
        packet_rows = []
        candidate_rows = []
        extensions = []
        receiver_air = {}
        for target_t in clock["score_s"]:
            receiver_air[str(target_t)] = bool(_state(played, float(target_t))["world"].uavs[1].airborne)
        summary["receiver_airborne_actual"] = receiver_air
        evaluable = {int(horizon) for horizon, target_t in zip(clock["horizons_s"], clock["score_s"]) if receiver_air[str(target_t)]}
        summary["evaluable_horizons_s"] = sorted(evaluable)
        original_label = _label(geometry["committed"][0])
        for field_name, seed, latent in fields:
            before = len(latent.points)
            _variance(case, poses, geometry["pool"], geometry["positions"], latent, seed, protocol, kernels, w8)
            if len(latent.points) != before:
                raise RuntimeError("variance read a wind point outside the draw")
            base_errors = _errors(poses, poses, geometry["positions"], latent, seed, protocol, kernels, condition, case["cfg"]["field"], w8)
            if len(latent.points) != before:
                raise RuntimeError("baseline truth extended the shared draw")
            for item in geometry["pool"]:
                label = _label(item["action"])
                flight = flights["baseline" if not item["alternative"] else label]
                sender_air = bool(_state(flight["played"], float(clock["sample_s"]))["world"].uavs[0].airborne)
                row = {
                    "field": field_name,
                    "seed": seed,
                    "action": label,
                    "separation_m": item["separation_m"],
                    "variance": item["score"],
                    "messages": messages,
                    "bytes": payload,
                    "baseline_bytes": payload,
                    "bits": flight["mission"]["bits"],
                    "baseline_bits": mission["bits"],
                    "packets": flight["mission"]["packets"],
                    "baseline_packets": mission["packets"],
                    "energy_j": flight["mission"]["energy_j"],
                    "baseline_energy_j": mission["energy_j"],
                    "returned": flight["mission"]["returned"],
                    "energy_left_j": flight["mission"]["energy_left_j"],
                    "reserve_margin_j": flight["mission"]["reserve_margin_j"],
                    "last_s": flight["mission"]["last_s"],
                    "watched": _watch(flight["ledger"], reference_ids),
                    "sender_airborne_at_sample": sender_air,
                    "mission_ok": False,
                    "scored": False,
                    "earliest": None,
                    "first_action_s": "",
                    "balances": 0,
                    "failed": "",
                }
                mission_ok = (
                    float(row["bits"]) + float(protocol["bits_tolerance"]) >= float(row["baseline_bits"])
                    and int(row["packets"]) >= int(row["baseline_packets"])
                    and all(row["returned"])
                    and float(row["energy_j"]) <= float(row["baseline_energy_j"]) + float(protocol["energy_tolerance_j"])
                    and int(row["bytes"]) <= int(row["baseline_bytes"])
                    and sender_air
                )
                row["mission_ok"] = mission_ok
                if item["alternative"] and mission_ok and evaluable:
                    copy = _clone(latent)
                    errors = _errors(poses, flight["poses"], geometry["positions"], copy, seed, protocol, kernels, condition, case["cfg"]["field"], w8)
                    if len(copy.points) != before:
                        extensions.append({"field": field_name, "seed": seed, "action": label, "added": len(copy.points) - before})
                    row["scored"] = True
                else:
                    errors = base_errors if not item["alternative"] else {}
                    row["scored"] = not item["alternative"]
                for horizon in (30, 60):
                    if horizon in errors:
                        row[f"rmse_{horizon}"] = _hyp(errors[horizon])
                        row[f"baseline_rmse_{horizon}"] = _hyp(base_errors[horizon])
                        row[f"d_rmse_{horizon}"] = row[f"rmse_{horizon}"] - row[f"baseline_rmse_{horizon}"]
                    else:
                        row[f"rmse_{horizon}"] = ""
                        row[f"baseline_rmse_{horizon}"] = _hyp(base_errors[horizon])
                        row[f"d_rmse_{horizon}"] = ""
                row["d_bits"] = row["bits"] - row["baseline_bits"]
                row["d_packets"] = row["packets"] - row["baseline_packets"]
                row["d_energy_j"] = row["energy_j"] - row["baseline_energy_j"]
                if item["alternative"]:
                    diff = _diff_rows(base_ledger, flight["ledger"], label)
                    if field_name == "control":
                        packet_rows.extend(diff)
                    other_events = [event for event in base_events if int(event["t_s"]) <= int(clock["choose_s"])] + [event for event in trace.events if event["arm"] == label and int(event["t_s"]) > int(clock["choose_s"])]
                    row["earliest"] = _earliest(base_events, other_events, diff)
                    changed = [second for second in range(int(clock["choose_s"]) + 1, int(mission_s) + 1) if trace.actions.get(("baseline", second, 0)) != trace.actions.get((label, second, 0))]
                    row["first_action_s"] = "" if not changed else int(changed[0])
                candidate_rows.append(row)
        grouped: dict[str, list[dict]] = {}
        for row in candidate_rows:
            grouped.setdefault(row["action"], []).append(row)
        passed = []
        for action, rows in grouped.items():
            if action == original_label:
                continue
            reasons = list(dict.fromkeys(_gate(rows, protocol, evaluable)))
            for row in rows:
                row["balances"] = int(not reasons)
                row["failed"] = ";".join(reasons)
            if not reasons:
                passed.append(action)
        orders = {}
        for row in candidate_rows:
            orders.setdefault(f"{row['field']}:{row['seed']}", []).append(row)
        variance_order = {}
        selectable = []
        for key, rows in orders.items():
            ranked = sorted(rows, key=lambda item: (float(item["variance"]), action_rank(next(pool["action"] for pool in geometry["pool"] if _label(pool["action"]) == item["action"]))))
            variance_order[key] = [item["action"] for item in ranked]
        for action in passed:
            picks = []
            for key, rows in orders.items():
                ranked = sorted(rows, key=lambda item: (float(item["variance"]), action_rank(next(pool["action"] for pool in geometry["pool"] if _label(pool["action"]) == item["action"]))))
                origin = next(item for item in ranked if item["action"] == original_label)
                best = ranked[0]
                picks.append(best["action"] == action and float(best["variance"]) + float(protocol["tie_variance"]) < float(origin["variance"]))
            if all(picks):
                selectable.append(action)
        summary.update(
            {
                "airtime_s": airtime,
                "messages": messages,
                "bytes": payload,
                "extensions": extensions,
                "variance_order": variance_order,
                "passed": passed,
                "selectable": selectable,
                "candidates": candidate_rows,
                "elapsed_s": time.perf_counter() - started,
            }
        )
        return _finish(out, summary, packet_rows, candidate_rows)
    finally:
        policies_mod.select_lookahead = real_select
        run_w4_mod.select_lookahead = real_select
        trace.restore()


if __name__ == "__main__":
    run()
