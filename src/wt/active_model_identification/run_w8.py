"""Choose one observation point for UAV 0 and score the paired wind forecast.

baseline replay: the original scheduler leaves the W7 delivery or energy.

budget match: the two arms do not send the same number of bytes.

timeline: a forecast uses the remote sample before the registered usable time.
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

from wt.active_model_identification.belief import prior_belief
from wt.active_model_identification.policies import action_rank, select_lookahead, selectable_candidates
from wt.active_model_identification.run_w1 import _label
from wt.active_model_identification.run_w4 import _condition, assess, play
from wt.active_model_identification.run_w5 import _draw_deltas, _ripple, _rmse, _score_candidates, posterior_mean
from wt.active_model_identification.run_w6 import _loglik, _normalize, _predict
from wt.active_model_identification.run_w7 import _advance_recorded, _by_uav, _case_for, _fly, forecast_hold
from wt.active_model_identification.spec import ROOT
from wt.active_model_identification.wind import mean_wind_fn, message_cost_j
from wt.active_model_identification.world import mission_terminal
from wt.active_model_identification import world as world_mod

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w8.json"
RUN_ID = "w8_20261118"


def _parent() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _load_protocol() -> tuple[dict, str]:
    raw = CONFIG_PATH.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("status") != "registered_before_run":
        raise SystemExit("registration moved")
    archived = json.loads((ROOT / str(protocol["w62_config"])).read_text(encoding="utf-8"))
    for key in ("observation_sigma_m_s", "prior_std_m_s", "prior_weight_global", "prior_weight_local"):
        if protocol[key] != archived[key]:
            raise RuntimeError(f"{key} left the W6.2 registration")
    local = protocol["local_field"]
    old = archived["local_field"]
    for key in ("amplitude_m_s", "length_m", "time_s", "generation_jitter"):
        if local[key] != old[key]:
            raise RuntimeError(f"local {key} left the W6.2 registration")
    if set(local["seeds"]) & set(old["seeds"] + archived["holdout_seeds"]):
        raise RuntimeError("wind seeds overlap an earlier round")
    return protocol, hashlib.sha256(raw).hexdigest()


def _kernel(protocol: dict, amplitude: float) -> dict:
    local = protocol["local_field"]
    return {
        "prior_std_m_s": float(protocol["prior_std_m_s"]),
        "sigma_m_s": float(protocol["observation_sigma_m_s"]),
        "amplitude_m_s": float(amplitude),
        "length_m": float(local["length_m"]),
        "time_s": float(local["time_s"]),
    }


def _key(x: float, y: float, t: float) -> tuple[float, float, float]:
    return (round(float(t), 6), round(float(x), 6), round(float(y), 6))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _state(played: dict, t_s: float) -> dict:
    return next(item for item in played["states"] if abs(float(item["t_s"]) - t_s) <= 1e-9)


def _requested(played: dict, t_s: int, uav: int) -> tuple:
    epoch = int(t_s) // 30
    row = next(item for item in played["rows"] if int(item["epoch"]) == epoch and int(item["uav"]) == uav)
    kind, target = str(row["requested"]).split(":")
    return (kind, None if target == "None" else int(target))


def _path_length(poses: dict, uav: int, mission: int) -> float:
    total = 0.0
    for second in range(mission):
        left = poses[(second, uav)]
        right = poses[(second + 1, uav)]
        total += math.hypot(right[0] - left[0], right[1] - left[1])
    return total


def _continue(case: dict, condition: dict, world, beliefs: list, epoch: int, preset: list) -> tuple[dict, dict]:
    holder = {"world": world}
    poses: dict[tuple[int, int], tuple[float, float]] = {}
    original = world_mod._move

    def wrapped(uav, action, spec, theta, phase_s, wind_xy=(0.0, 0.0)):
        original(uav, action, spec, theta, phase_s, wind_xy)
        current = holder["world"]
        t_now = int(round(float(current.t)))
        poses[(t_now + 1, int(uav.index))] = (float(uav.x), float(uav.y))

    world_mod._move = wrapped
    try:
        played = play(case, condition, world, beliefs, epoch, preset)
    finally:
        world_mod._move = original
    return played, poses


def _plan(state: dict, committed: list, case: dict, wind_at, method: str, steps: int) -> tuple[dict, dict]:
    spec = case["spec"]
    factor = float(case["factor"])
    world = state["world"].snapshot()
    beliefs = [item.copy() for item in state["beliefs"]]
    actions = list(committed)
    epoch = int(round(float(state["t_s"]) / float(spec["time"]["decision_s"])))
    positions: dict[int, tuple[float, float]] = {}
    airborne: dict[int, bool] = {}
    for step in range(steps):
        trace, _theta = _advance_recorded(world, actions, beliefs, spec, factor, wind_at, epoch + step)
        grouped = _by_uav(trace, int(spec["time"]["decision_s"]))
        start = int(round(float(state["t_s"]))) + step * int(spec["time"]["decision_s"])
        for second, row in enumerate(grouped[1], start=1):
            positions[start + second] = (float(row["x"]), float(row["y"]))
        airborne[start + int(spec["time"]["decision_s"])] = bool(world.uavs[1].airborne)
        requested, _tables = select_lookahead(method, world, [item.copy() for item in beliefs], spec, factor, None)
        actions = list(requested)
    return positions, airborne


def _endpoint(uav, action: tuple, belief, spec: dict, wind_at, t0: float) -> tuple[float, float]:
    points, _theta = forecast_hold([uav], [action], [belief], spec, wind_at, t0, 30)
    return points[int(uav.index)][30]


def _local_cov(left: dict, right: dict, kernel: dict, jitter: float) -> float:
    distance = math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))
    dt = abs(float(left["t"]) - float(right["t"]))
    value = (float(kernel["amplitude_m_s"]) ** 2) * math.exp(-distance / float(kernel["length_m"])) * math.exp(-dt / float(kernel["time_s"]))
    if jitter and left is right:
        value += jitter
    return value


class Field:
    def __init__(self, points: list[dict], values: dict[tuple, tuple[float, float]] | None, kernel: dict, seed: str, jitter: float):
        self.points = points
        self.values = {} if values is None else dict(values)
        self.kernel = kernel
        self.seed = seed
        self.jitter = jitter
        self.control = values is None

    def at(self, x: float, y: float, t: float) -> tuple[float, float]:
        if self.control:
            return 0.0, 0.0
        key = _key(x, y, t)
        if key in self.values:
            return self.values[key]
        target = {"x": x, "y": y, "t": t}
        size = len(self.points)
        matrix = [[_local_cov(self.points[row], self.points[col], self.kernel, self.jitter if row == col else 0.0) for col in range(size)] for row in range(size)]
        lower = _cholesky(matrix)
        cross = [_local_cov(target, point, self.kernel, 0.0) for point in self.points]
        weight = _solve(lower, cross)
        prior = _local_cov(target, target, self.kernel, 0.0)
        variance = max(0.0, prior - sum(cross[index] * weight[index] for index in range(size)))
        mean_x = sum(weight[index] * self.values[self.points[index]["key"]][0] for index in range(size))
        mean_y = sum(weight[index] * self.values[self.points[index]["key"]][1] for index in range(size))
        generator = random.Random(f"w8-extend-{self.seed}-{key[0]}-{key[1]}-{key[2]}")
        drawn = (mean_x + math.sqrt(variance) * generator.gauss(0.0, 1.0), mean_y + math.sqrt(variance) * generator.gauss(0.0, 1.0))
        self.values[key] = drawn
        self.points.append({"key": key, "x": x, "y": y, "t": t})
        return drawn


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for col in range(size):
        diagonal = matrix[col][col] - sum(lower[col][k] * lower[col][k] for k in range(col))
        if diagonal <= 1e-14:
            diagonal = 1e-14
        lower[col][col] = math.sqrt(diagonal)
        for row in range(col + 1, size):
            total = matrix[row][col] - sum(lower[row][k] * lower[col][k] for k in range(col))
            lower[row][col] = total / lower[col][col]
    return lower


def _solve(lower: list[list[float]], rhs: list[float]) -> list[float]:
    forward = [0.0] * len(rhs)
    for row in range(len(rhs)):
        forward[row] = (rhs[row] - sum(lower[row][k] * forward[k] for k in range(row))) / lower[row][row]
    backward = [0.0] * len(rhs)
    for row in range(len(rhs) - 1, -1, -1):
        backward[row] = (forward[row] - sum(lower[k][row] * backward[k] for k in range(row + 1, len(rhs)))) / lower[row][row]
    return backward


def _noise(seed: str, role: str, t: float, sigma: float) -> tuple[float, float]:
    generator = random.Random(f"w8-{seed}-{role}-{int(t)}")
    return generator.gauss(0.0, sigma), generator.gauss(0.0, sigma)


def _observe(condition: dict, field_cfg: dict, latent: Field, x: float, y: float, t: float, epsilon: tuple[float, float], sample_id: int) -> dict:
    delta = latent.at(x, y, t)
    return {
        "id": sample_id,
        "x": float(x),
        "y": float(y),
        "t": float(t),
        "rx": float(condition["ax"]) + delta[0] + epsilon[0],
        "ry": float(condition["ay"]) + delta[1] + epsilon[1],
    }


def _weights(observations: list[dict], kernels: dict[str, dict], prior: dict[str, float]) -> dict[str, float]:
    log_weights = {name: math.log(float(prior[name])) for name in prior}
    kept: list[dict] = []
    for sample in observations:
        for name, kernel in kernels.items():
            mean, variance = _predict(kept, sample, kernel)
            log_weights[name] += _loglik(sample, mean, variance)
        kept.append(sample)
    return _normalize(log_weights)


def _mixture_variance(observations: list[dict], candidate: dict, targets: list[dict], kernels: dict[str, dict], weights: dict[str, float]) -> float:
    total = 0.0
    for name, kernel in kernels.items():
        total += float(weights[name]) * _score_candidates(observations, [candidate], targets, kernel)[0]
    return 2.0 * total


def _choose(case: dict, played: dict, poses: dict, protocol: dict, wind_at, kernels: dict[str, dict], latent: Field, seed: str) -> dict:
    clock = protocol["timeline"]
    choose_t = int(clock["choose_s"])
    sample_t = int(clock["sample_s"])
    state = _state(played, choose_t)
    world = state["world"].snapshot()
    beliefs = [item.copy() for item in state["beliefs"]]
    spec = case["spec"]
    method = str(protocol["method"])
    both_air = all(bool(uav.airborne) for uav in world.uavs)
    prior_mean = prior_belief(spec).mean()
    telemetry = any(abs(item.mean() - prior_mean) > 1e-6 for item in beliefs)
    committed = [_requested(played, choose_t, index) for index in range(2)]
    positions, airborne = _plan(state, committed, case, wind_at, method, 4)
    qualified = both_air and telemetry and bool(airborne.get(int(clock["score_s"][0]), False))
    original = committed[0]
    origin = _endpoint(world.uavs[0], original, beliefs[0], spec, wind_at, float(choose_t))
    pool = [{"action": original, "position": origin, "separation_m": 0.0, "alternative": False}]
    if qualified:
        legal, _mapping = selectable_candidates(spec, world, beliefs, 0, include_calibrate=False, factor=float(case["factor"]))
        found = []
        for action in legal:
            if action == original:
                continue
            position = _endpoint(world.uavs[0], action, beliefs[0], spec, wind_at, float(choose_t))
            gap = math.hypot(position[0] - origin[0], position[1] - origin[1])
            if gap + 1e-9 >= float(protocol["min_separation_m"]):
                found.append((gap, action_rank(action), action, position))
        found.sort(key=lambda item: (-item[0], item[1]))
        for gap, _rank, action, position in found[: int(protocol["max_alternatives"])]:
            pool.append({"action": action, "position": position, "separation_m": gap, "alternative": True})
    onboard = []
    for sample_t_i, sample_id in ((0, 1), (choose_t, 2)):
        x_m, y_m = poses[(sample_t_i, 1)]
        epsilon = _noise(seed, "onboard", sample_t_i, float(protocol["observation_sigma_m_s"]))
        onboard.append(_observe(case["condition_row"], case["cfg"]["field"], latent, x_m, y_m, float(sample_t_i), epsilon, sample_id))
    weights = _weights(onboard, kernels, {"global": protocol["prior_weight_global"], "local": protocol["prior_weight_local"]}) if qualified else {"global": 0.5, "local": 0.5}
    targets = [{"id": -1, "x": positions[int(item)][0], "y": positions[int(item)][1], "t": float(item)} for item in clock["score_s"]]
    for item in pool:
        candidate = {"id": -10, "x": item["position"][0], "y": item["position"][1], "t": float(sample_t)}
        item["score"] = _mixture_variance(onboard, candidate, targets, kernels, weights) if qualified else 0.0
    best = pool[0]
    if qualified:
        for item in pool[1:]:
            if item["score"] + float(protocol["tie_variance"]) < best["score"]:
                best = item
    return {
        "qualified": qualified,
        "both_air": both_air,
        "telemetry": telemetry,
        "predicted_airborne": {str(key): value for key, value in airborne.items()},
        "predicted_receiver": {str(key): positions[key] for key in clock["score_s"]},
        "candidates": [{"action": _label(item["action"]), "x": item["position"][0], "y": item["position"][1], "separation_m": item["separation_m"], "score": item["score"]} for item in pool],
        "chosen": best["action"],
        "changed": bool(qualified and best["alternative"]),
        "weights": weights,
    }


def _forecast(samples: list[dict], query: dict, kernels: dict[str, dict], prior: dict[str, float], ripple: tuple[float, float]) -> tuple[tuple[float, float], dict[str, float]]:
    weights = _weights(samples, kernels, prior)
    mixed = [0.0, 0.0]
    for name, kernel in kernels.items():
        mean = posterior_mean(samples, query, kernel)
        mixed[0] += weights[name] * (ripple[0] + mean[0])
        mixed[1] += weights[name] * (ripple[1] + mean[1])
    return (mixed[0], mixed[1]), weights


def _truth(condition: dict, field_cfg: dict, latent: Field, x: float, y: float, t: float) -> tuple[float, float]:
    ripple = _ripple(condition, field_cfg, x, y, t)
    delta = latent.at(x, y, t)
    return float(condition["ax"]) + ripple[0] + delta[0], float(condition["ay"]) + ripple[1] + delta[1]


def _merge_poses(base: dict, extra: dict) -> dict:
    merged = dict(base)
    merged.update(extra)
    return merged


def _packets(world, mission: float) -> dict:
    times = []
    for packet in world.packets:
        delivered = packet.delivered_s is not None and packet.holder == "shore" and float(packet.delivered_s) <= mission + 1e-9
        if delivered:
            times.append(float(packet.delivered_s))
    return {"n": len(times), "last_s": None if not times else max(times)}


def _terminals(world, spec: dict) -> list[dict]:
    return [
        {"uav": int(item["uav"]), "energy_j": float(item["energy_j"]), "returned": bool(item["returned"])}
        for item in mission_terminal(world, spec)
    ]


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    protocol, digest = _load_protocol()
    out = out_dir or (ROOT / "results" / "wind_observation_w8" / RUN_ID)
    if "wind_observation_w8" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w8")
    archived = json.loads((ROOT / str(protocol["w7_result"])).read_text(encoding="utf-8"))
    archived_case = {row["scenario_id"]: row for row in archived["scenarios"]}
    _cfg, archive = __import__("wt.active_model_identification.run_w4", fromlist=["load_w4"]).load_w4()
    if str(archive["cfg"]["method"]) != str(protocol["method"]):
        raise RuntimeError("registered method left the frozen scheduler")
    condition = _condition(archive, str(protocol["condition"]))
    wind_at = mean_wind_fn(None, archive["cfg"]["field"], float(protocol["prior_std_m_s"]))
    kernels = {"global": _kernel(protocol, 0.0), "local": _kernel(protocol, protocol["local_field"]["amplitude_m_s"])}
    clock = protocol["timeline"]
    scenarios = []
    points: dict[tuple, dict] = {}

    def add_point(x: float, y: float, t: float) -> None:
        key = _key(x, y, t)
        points.setdefault(key, {"key": key, "x": float(x), "y": float(y), "t": float(t)})

    for scenario_id in [row["scenario_id"] for row in archived["scenarios"]]:
        case, _manifest = _case_for(scenario_id)
        case = dict(case)
        case["condition_row"] = condition
        print(f"scenario {scenario_id}", flush=True)
        played, poses, _committed = _fly(case, condition)
        outcome = assess(played["world"], case["spec"], set(), 3, 0.0)
        reference = archived_case[scenario_id]
        if abs(float(outcome["on_time_bits"]) - float(reference["bits"])) > 1e-4 or abs(float(outcome["energy_j"]) - float(reference["energy_j"])) > 1e-3:
            raise RuntimeError(f"{scenario_id} left the W7 trajectory")
        state = _state(played, int(clock["choose_s"]))
        committed = [_requested(played, int(clock["choose_s"]), index) for index in range(2)]
        positions, _air = _plan(state, committed, case, wind_at, str(protocol["method"]), 4)
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
        scenarios.append({"id": scenario_id, "case": case, "played": played, "poses": poses, "outcome": outcome, "plan": positions})
    drawn = _draw_deltas(
        list(points.values()),
        kernels["local"],
        [int(seed) for seed in protocol["local_field"]["seeds"]],
        float(protocol["local_field"]["generation_jitter"]),
    )
    fields = [("control", "control", Field(list(points.values()), None, kernels["local"], "control", 0.0))]
    for seed, table in drawn.items():
        values = {point["key"]: table[point["key"]] for point in points.values()}
        fields.append(("local", str(seed), Field(list(points.values()), values, kernels["local"], str(seed), float(protocol["local_field"]["generation_jitter"]))))
    airtime = float(message_cost_j(int(protocol["payload_bytes"]))["airtime_s"])
    if airtime <= 0.0 or airtime >= float(clock["usable_s"]) - float(clock["send_s"]):
        raise RuntimeError("airtime does not fit the registered timeline")
    if any(int(item) < int(clock["usable_s"]) for item in clock["score_s"]):
        raise RuntimeError("a score time precedes the usable message")
    tasks = []
    pairs = []
    missions: dict[tuple, dict] = {}
    for scenario in scenarios:
        state = _state(scenario["played"], int(clock["choose_s"]))
        saved_world = state["world"].snapshot()
        saved_beliefs = [item.copy() for item in state["beliefs"]]
        for field_name, seed, latent in fields:
            choice = _choose(scenario["case"], scenario["played"], scenario["poses"], protocol, wind_at, kernels, latent, seed)
            if int(clock["send_s"]) >= int(clock["usable_s"]):
                raise RuntimeError("remote sample is usable at the send time")
            key = (scenario["id"], _label(choice["chosen"]))
            if key not in missions:
                if choice["changed"]:
                    preset = [choice["chosen"], _requested(scenario["played"], int(clock["choose_s"]), 1)]
                    played, extra = _continue(
                        scenario["case"],
                        condition,
                        saved_world.snapshot(),
                        [item.copy() for item in saved_beliefs],
                        int(clock["choose_s"]) // int(protocol["decision_s"]),
                        preset,
                    )
                    poses = _merge_poses(scenario["poses"], extra)
                else:
                    played = scenario["played"]
                    poses = scenario["poses"]
                missions[key] = {"played": played, "poses": poses}
            flown = missions[key]
            spec = scenario["case"]["spec"]
            mission = float(spec["time"]["mission_s"])
            base_packets = _packets(scenario["played"]["world"], mission)
            exp_packets = _packets(flown["played"]["world"], mission)
            base_terms = _terminals(scenario["played"]["world"], spec)
            exp_terms = _terminals(flown["played"]["world"], spec)
            exp_outcome = assess(flown["played"]["world"], spec, set(), 3, 0.0)
            messages = 1 if choice["qualified"] else 0
            if messages != (1 if choice["qualified"] else 0):
                raise RuntimeError("budget")
            remote_epsilon = _noise(seed, "remote", int(clock["sample_s"]), float(protocol["observation_sigma_m_s"]))
            query = scenario["plan"]
            errors = {}
            weight_row = {}
            for arm, poses_arm, remote_uav_track in (
                ("baseline", scenario["poses"], scenario["poses"]),
                ("experiment", flown["poses"], flown["poses"]),
            ):
                samples = []
                for index, sample_t in enumerate((0, int(clock["choose_s"]), int(clock["sample_s"]), int(clock["usable_s"]))):
                    x_m, y_m = scenario["poses"][(sample_t, 1)]
                    epsilon = _noise(seed, "onboard", sample_t, float(protocol["observation_sigma_m_s"]))
                    samples.append(_observe(condition, scenario["case"]["cfg"]["field"], latent, x_m, y_m, float(sample_t), epsilon, 100 + index))
                if choice["qualified"]:
                    x_m, y_m = remote_uav_track[(int(clock["sample_s"]), 0)]
                    samples.append(_observe(condition, scenario["case"]["cfg"]["field"], latent, x_m, y_m, float(clock["sample_s"]), remote_epsilon, 200))
                for horizon, target_t in zip(clock["horizons_s"], clock["score_s"]):
                    predicted = query[int(target_t)]
                    ripple = _ripple(condition, scenario["case"]["cfg"]["field"], predicted[0], predicted[1], float(target_t))
                    forecast, weights = _forecast(
                        samples,
                        {"id": -1, "x": predicted[0], "y": predicted[1], "t": float(target_t)},
                        kernels,
                        {"global": protocol["prior_weight_global"], "local": protocol["prior_weight_local"]},
                        ripple,
                    )
                    actual = scenario["poses"][(int(target_t), 1)]
                    truth = _truth(condition, scenario["case"]["cfg"]["field"], latent, actual[0], actual[1], float(target_t))
                    error = (forecast[0] - truth[0], forecast[1] - truth[1])
                    own = poses_arm[(int(target_t), 1)]
                    own_truth = _truth(condition, scenario["case"]["cfg"]["field"], latent, own[0], own[1], float(target_t))
                    own_error = (forecast[0] - own_truth[0], forecast[1] - own_truth[1])
                    errors[(arm, int(horizon))] = error
                    pairs.append(
                        {
                            "scenario_id": scenario["id"],
                            "field": field_name,
                            "seed": seed,
                            "arm": arm,
                            "horizon_s": int(horizon),
                            "changed": int(choice["changed"]),
                            "qualified": int(choice["qualified"]),
                            "receiver_airborne": int(bool(_state(scenario["played"], float(target_t))["world"].uavs[1].airborne)),
                            "error_m_s": math.hypot(*error),
                            "actual_track_error_m_s": math.hypot(*own_error),
                            "track_offset_m_s": math.hypot(*own_error) - math.hypot(*error),
                            "dx": error[0],
                            "dy": error[1],
                        }
                    )
                    weight_row[arm] = weights
            base_length = _path_length(scenario["poses"], 0, int(mission))
            exp_length = _path_length(flown["poses"], 0, int(mission))
            tasks.append(
                {
                    "scenario_id": scenario["id"],
                    "field": field_name,
                    "seed": seed,
                    "opportunities": int(choice["qualified"]),
                    "n_candidates": len(choice["candidates"]),
                    "changed": int(choice["changed"]),
                    "chosen": _label(choice["chosen"]),
                    "original": choice["candidates"][0]["action"],
                    "separations_m": [item["separation_m"] for item in choice["candidates"]],
                    "scores": [item["score"] for item in choice["candidates"]],
                    "messages": messages,
                    "bytes": messages * int(protocol["payload_bytes"]),
                    "airtime_s": airtime,
                    "use_delay_s": float(clock["usable_s"]) - float(clock["send_s"]),
                    "valid_predictions": len(clock["horizons_s"]),
                    "extra_path_m": exp_length - base_length,
                    "extra_energy_j": float(exp_outcome["energy_j"]) - float(scenario["outcome"]["energy_j"]),
                    "bits_baseline": float(scenario["outcome"]["on_time_bits"]),
                    "bits_experiment": float(exp_outcome["on_time_bits"]),
                    "packets_baseline": base_packets["n"],
                    "packets_experiment": exp_packets["n"],
                    "last_delivery_baseline_s": base_packets["last_s"],
                    "last_delivery_experiment_s": exp_packets["last_s"],
                    "returned_baseline": [item["returned"] for item in base_terms],
                    "returned_experiment": [item["returned"] for item in exp_terms],
                    "energy_left_baseline_j": [item["energy_j"] for item in base_terms],
                    "energy_left_experiment_j": [item["energy_j"] for item in exp_terms],
                    "weight_global_experiment": weight_row["experiment"]["global"],
                }
            )
    for task in tasks:
        if task["messages"] * int(protocol["payload_bytes"]) != task["bytes"]:
            raise RuntimeError("budget")
        twin = next(item for item in tasks if item["scenario_id"] == task["scenario_id"] and item["field"] == task["field"] and item["seed"] == task["seed"])
        if twin["messages"] != task["messages"]:
            raise RuntimeError("arms disagree on the message budget")
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "pairs.csv", pairs)
    flat_tasks = []
    for task in tasks:
        flat_tasks.append({key: (json.dumps(value) if isinstance(value, (list, dict)) else value) for key, value in task.items()})
    _write_csv(out / "tasks.csv", flat_tasks)

    def rmse_of(changed: int | None, horizon: int, arm: str) -> dict:
        chosen = [
            (float(row["dx"]), float(row["dy"]))
            for row in pairs
            if int(row["horizon_s"]) == horizon and row["arm"] == arm and (changed is None or int(row["changed"]) == changed)
        ]
        return {"n": len(chosen), "rmse_m_s": None if not chosen else _rmse(chosen)}

    summary = {
        "schema_version": 1,
        "parent_commit": _parent(),
        "config_sha256": digest,
        "config_status": protocol["status"],
        "baseline_replay": True,
        "budget_match": True,
        "timeline_ok": True,
        "default_changed": False,
        "timeline": clock,
        "airtime_s": airtime,
        "centralized_assumption": protocol["centralized_assumption"],
        "opportunities": {scenario["id"]: int(any(task["opportunities"] for task in tasks if task["scenario_id"] == scenario["id"])) for scenario in scenarios},
        "n_changed": sum(int(task["changed"]) for task in tasks),
        "n_tasks": len(tasks),
        "rmse": {
            "all": {str(horizon): {"baseline": rmse_of(None, int(horizon), "baseline"), "experiment": rmse_of(None, int(horizon), "experiment")} for horizon in clock["horizons_s"]},
            "changed": {str(horizon): {"baseline": rmse_of(1, int(horizon), "baseline"), "experiment": rmse_of(1, int(horizon), "experiment")} for horizon in clock["horizons_s"]},
        },
        "tasks": tasks,
        "elapsed_s": time.perf_counter() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"changed {summary['n_changed']}/{summary['n_tasks']} "
        f"rmse30 {summary['rmse']['all']['30']['baseline']['rmse_m_s']} -> {summary['rmse']['all']['30']['experiment']['rmse_m_s']}",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run()
