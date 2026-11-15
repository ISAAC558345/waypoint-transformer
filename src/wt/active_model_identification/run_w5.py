"""Choose one wind sample for a future leg on the frozen W4 trajectory.

frozen trajectory: on-time bits or team energy leave the W4 headwind baseline.

future isolation: a send uses a sample taken after the send time, or the choice reads a measurement.

equal budget: the two sending methods do not use one equal payload at every decision.

zero amplitude: amplitude 0 changes the chosen sample, or the posterior leaves the location-free mean.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

from wt.active_model_identification import run_w4 as run_w4_mod
from wt.active_model_identification import world as world_mod
from wt.active_model_identification.run_w4 import _condition, assess, baseline, load_w4
from wt.active_model_identification.spec import ROOT
from wt.active_model_identification.wind import message_cost_j, true_wind

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w5.json"
RUN_ID = "w5_20261115"
ARCHIVED_BITS = 140000.0
ARCHIVED_ENERGY_J = 111310.32858681791
TIE_VARIANCE = 1e-9


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    digest = hashlib.sha256(raw).hexdigest()
    protocol = json.loads(raw)
    if protocol.get("status") != "registered_before_run":
        raise SystemExit("registration moved")
    return protocol, digest


def _kernel(protocol: dict, amplitude: float) -> dict:
    local = protocol["local_field"]
    return {
        "prior_std_m_s": float(protocol["prior_std_m_s"]),
        "sigma_m_s": float(protocol["observation_sigma_m_s"]),
        "amplitude_m_s": float(amplitude),
        "length_m": float(local["length_m"]),
        "time_s": float(local["time_s"]),
    }


def _cov(left: dict, right: dict, kernel: dict, *, noise: bool) -> float:
    distance = math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))
    dt = abs(float(left["t"]) - float(right["t"]))
    local = (float(kernel["amplitude_m_s"]) ** 2) * math.exp(-distance / float(kernel["length_m"])) * math.exp(
        -dt / float(kernel["time_s"])
    )
    value = float(kernel["prior_std_m_s"]) ** 2 + local
    if noise and int(left["id"]) == int(right["id"]):
        value += float(kernel["sigma_m_s"]) ** 2
    return value


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for col in range(size):
        diagonal = matrix[col][col] - sum(lower[col][k] * lower[col][k] for k in range(col))
        if diagonal <= 0.0:
            raise ValueError(f"covariance not positive at {col}")
        lower[col][col] = math.sqrt(diagonal)
        for row in range(col + 1, size):
            total = matrix[row][col] - sum(lower[row][k] * lower[col][k] for k in range(col))
            lower[row][col] = total / lower[col][col]
    return lower


def _solve(lower: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    forward = [0.0] * size
    for row in range(size):
        forward[row] = (rhs[row] - sum(lower[row][k] * forward[k] for k in range(row))) / lower[row][row]
    backward = [0.0] * size
    for row in range(size - 1, -1, -1):
        backward[row] = (forward[row] - sum(lower[k][row] * backward[k] for k in range(row + 1, size))) / lower[row][row]
    return backward


def _factor_observations(observations: list[dict], kernel: dict) -> list[list[float]]:
    size = len(observations)
    matrix = [[_cov(observations[row], observations[col], kernel, noise=True) for col in range(size)] for row in range(size)]
    return _cholesky(matrix)


def global_mean(values: list[float], prior_std: float, sigma: float) -> float:
    """Location-free mean. Amplitude 0 has to match this."""
    count = len(values)
    if count == 0:
        return 0.0
    prior_var = prior_std ** 2
    noise_var = sigma ** 2
    return (prior_var * sum(values)) / (noise_var + count * prior_var)


def posterior_mean(observations: list[dict], target: dict, kernel: dict) -> tuple[float, float]:
    """Mean of ax+delta at one point. Duplicate sample ids are rejected."""
    seen: set[int] = set()
    for item in observations:
        sample_id = int(item["id"])
        if sample_id in seen:
            raise ValueError("duplicate sample")
        seen.add(sample_id)
    if not observations:
        return 0.0, 0.0
    lower = _factor_observations(observations, kernel)
    cross = [_cov(target, item, kernel, noise=False) for item in observations]
    weight = _solve(lower, cross)
    east = sum(weight[index] * float(observations[index]["rx"]) for index in range(len(observations)))
    north = sum(weight[index] * float(observations[index]["ry"]) for index in range(len(observations)))
    return east, north


def _score_candidates(arrived: list[dict], candidates: list[dict], targets: list[dict], kernel: dict) -> list[float]:
    """Mean predictive variance at the targets after adding one candidate. Lower is better."""
    if not targets:
        raise ValueError("no predicted point")
    if not arrived:
        scores = []
        for candidate in candidates:
            own = _cov(candidate, candidate, kernel, noise=True)
            total = 0.0
            for target in targets:
                prior = _cov(target, target, kernel, noise=False)
                cross = _cov(target, candidate, kernel, noise=False)
                total += prior - (cross * cross) / own
            scores.append(total / len(targets))
        return scores
    lower = _factor_observations(arrived, kernel)
    target_weights = []
    target_prior = []
    for target in targets:
        cross = [_cov(target, item, kernel, noise=False) for item in arrived]
        target_weights.append(_solve(lower, cross))
        target_prior.append(_cov(target, target, kernel, noise=False))
    scores = []
    for candidate in candidates:
        own_cross = [_cov(candidate, item, kernel, noise=False) for item in arrived]
        own_weight = _solve(lower, own_cross)
        own_var = _cov(candidate, candidate, kernel, noise=True) - sum(
            own_cross[index] * own_weight[index] for index in range(len(arrived))
        )
        total = 0.0
        for target, weights, prior in zip(targets, target_weights, target_prior):
            latent_cross = _cov(target, candidate, kernel, noise=False) - sum(
                _cov(target, arrived[index], kernel, noise=False) * own_weight[index] for index in range(len(arrived))
            )
            base = prior - sum(
                _cov(target, arrived[index], kernel, noise=False) * weights[index] for index in range(len(arrived))
            )
            if own_var <= 1e-12:
                total += base
            else:
                total += base - (latent_cross * latent_cross) / own_var
        scores.append(total / len(targets))
    return scores


def select_sample(samples: list[dict], arrived_ids: set[int], predicted: list[dict], now: float, kernel: dict) -> dict:
    """Pick one sample. Measurements and the true future track are not inputs."""
    available = []
    for sample in samples:
        if float(sample["t"]) > float(now) + 1e-9:
            raise ValueError("future sample")
        available.append(sample)
    if len(available) < 1:
        raise ValueError("no sample")
    fresh = [sample for sample in available if int(sample["id"]) not in arrived_ids]
    if not fresh:
        fresh = list(available)
    ordered = sorted(fresh, key=lambda sample: (float(sample["t"]), int(sample["id"])))
    scores = _score_candidates(
        [sample for sample in available if int(sample["id"]) in arrived_ids],
        ordered,
        predicted,
        kernel,
    )
    best = ordered[0]
    best_score = scores[0]
    for sample, score in zip(ordered, scores):
        newer = (float(sample["t"]), int(sample["id"])) >= (float(best["t"]), int(best["id"]))
        if score < best_score - TIE_VARIANCE or (score <= best_score + TIE_VARIANCE and newer):
            best = sample
            best_score = score
    latest = max(available, key=lambda sample: (float(sample["t"]), int(sample["id"])))
    return {"chosen": best, "latest": latest, "available": len(available), "fresh": len(fresh)}


def _extrapolate(uav, action: tuple, spec: dict, horizons: list[int]) -> dict[int, tuple[float, float]]:
    clone = copy.deepcopy(uav)
    points: dict[int, tuple[float, float]] = {}
    done = 0
    for horizon in sorted(int(item) for item in horizons):
        for second in range(done, horizon):
            world_mod._move(clone, action, spec, 1.0, second % 30, (0.0, 0.0))
        points[horizon] = (float(clone.x), float(clone.y))
        done = horizon
    return points


def _replay(case: dict, condition: dict) -> tuple[dict, dict, dict]:
    """Record the frozen path every second without changing the scheduler."""
    holder: dict[str, object] = {"world": None}
    poses: dict[tuple[int, int], tuple[float, float]] = {}
    committed: dict[tuple[int, int], tuple] = {}
    original_move = world_mod._move
    original_advance = run_w4_mod.advance

    def wrapped_advance(world, *args, **kwargs):
        holder["world"] = world
        return original_advance(world, *args, **kwargs)

    def wrapped_move(uav, action, spec, theta, phase_s, wind_xy=(0.0, 0.0)):
        original_move(uav, action, spec, theta, phase_s, wind_xy)
        world = holder["world"]
        if world is None:
            return
        t_now = int(round(float(world.t)))
        poses[(t_now + 1, int(uav.index))] = (float(uav.x), float(uav.y))
        if int(phase_s) == 0:
            target = None if action[1] is None else int(action[1])
            committed[(t_now, int(uav.index))] = (str(action[0]), target)

    world_mod._move = wrapped_move
    run_w4_mod.advance = wrapped_advance
    try:
        played = baseline(case, condition)
    finally:
        world_mod._move = original_move
        run_w4_mod.advance = original_advance
    return played, poses, committed


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def _ripple(condition: dict, field: dict, x: float, y: float, t: float) -> tuple[float, float]:
    base = true_wind(condition, x, y, t, field)
    return base[0] - float(condition["ax"]), base[1] - float(condition["ay"])


def _draw_deltas(points: list[dict], kernel: dict, seeds: list[int], jitter: float) -> dict[int, dict[tuple, tuple[float, float]]]:
    size = len(points)
    matrix = []
    for row in range(size):
        matrix.append([])
        for col in range(size):
            value = _cov(points[row], points[col], kernel, noise=False) - float(kernel["prior_std_m_s"]) ** 2
            if row == col:
                value += jitter
            matrix[row].append(value)
    lower = _cholesky(matrix)
    drawn: dict[int, dict[tuple, tuple[float, float]]] = {}
    for seed in seeds:
        east_rng = random.Random(f"w5-local-{seed}-x")
        north_rng = random.Random(f"w5-local-{seed}-y")
        east_z = [east_rng.gauss(0.0, 1.0) for _ in range(size)]
        north_z = [north_rng.gauss(0.0, 1.0) for _ in range(size)]
        east = [sum(lower[row][col] * east_z[col] for col in range(row + 1)) for row in range(size)]
        north = [sum(lower[row][col] * north_z[col] for col in range(row + 1)) for row in range(size)]
        table: dict[tuple, tuple[float, float]] = {}
        for index, point in enumerate(points):
            table[point["key"]] = (east[index], north[index])
        drawn[int(seed)] = table
    return drawn


def _rmse(errors: list[tuple[float, float]]) -> float:
    if not errors:
        raise ValueError("no scored point")
    return math.sqrt(sum(dx * dx + dy * dy for dx, dy in errors) / len(errors))


def _choice_table(samples, decisions, kernel, airtime: float) -> list[dict]:
    sends: list[tuple[float, int]] = []
    rows = []
    for decision in decisions:
        now = float(decision["t"])
        arrived = {sample_id for send_t, sample_id in sends if send_t + airtime <= now + 1e-9 and now > send_t + 1e-9}
        visible = [
            {"id": sample["id"], "t": sample["t"], "x": sample["x"], "y": sample["y"]}
            for sample in samples
            if float(sample["t"]) <= now + 1e-9
        ]
        predicted = [{"x": x, "y": y, "t": now + horizon, "id": -1} for horizon, (x, y) in sorted(decision["predicted"].items())]
        picked = select_sample(visible, arrived, predicted, now, kernel)
        chosen = picked["chosen"]
        latest = picked["latest"]
        if float(chosen["t"]) > now + 1e-9 or float(latest["t"]) > now + 1e-9:
            raise RuntimeError("future sample selected")
        arrived_rows = [sample for sample in visible if int(sample["id"]) in arrived]
        fresh_rows = [sample for sample in visible if int(sample["id"]) not in arrived] or list(visible)
        ordered = sorted(fresh_rows, key=lambda sample: (float(sample["t"]), int(sample["id"])))
        variances = _score_candidates(arrived_rows, ordered, predicted, kernel)
        latest_variance = variances[ordered.index(next(sample for sample in ordered if int(sample["id"]) == int(latest["id"])))]
        other_variances = [score for sample, score in zip(ordered, variances) if int(sample["id"]) != int(latest["id"])]
        advantage = min(other_variances) - latest_variance if other_variances else None
        sends.append((now, int(chosen["id"])))
        target = decision["predicted"].get(30) or next(iter(decision["predicted"].values()))
        rows.append(
            {
                "t_s": now,
                "available": int(picked["available"]),
                "fresh": int(picked["fresh"]),
                "latest_id": int(latest["id"]),
                "latest_t_s": float(latest["t"]),
                "chosen_id": int(chosen["id"]),
                "chosen_t_s": float(chosen["t"]),
                "differed": int(chosen["id"]) != int(latest["id"]),
                "chosen_distance_m": math.hypot(float(chosen["x"]) - target[0], float(chosen["y"]) - target[1]),
                "latest_distance_m": math.hypot(float(latest["x"]) - target[0], float(latest["y"]) - target[1]),
                "chosen_age_s": now - float(chosen["t"]),
                "latest_variance_advantage": advantage,
            }
        )
    return rows


def _inbox(rows: list[dict], decision_t: float, airtime: float) -> list[int]:
    ids = []
    seen: set[int] = set()
    for row in rows:
        if float(row["t_s"]) + airtime <= decision_t + 1e-9 and decision_t > float(row["t_s"]) + 1e-9:
            sample_id = int(row["chosen_id"])
            if sample_id not in seen:
                seen.add(sample_id)
                ids.append(sample_id)
    return ids


def _evaluate(protocol, case, condition, samples, decisions, choice_rows, deltas, seed, kernel) -> list[dict]:
    by_id = {int(sample["id"]): sample for sample in samples}
    field = case["cfg"]["field"]
    airtime = float(message_cost_j(int(protocol["payload_bytes"]))["airtime_s"])
    methods = {
        "no_message": None,
        "latest": [row for row in choice_rows if row["method"] == "latest"],
        "relevant": [row for row in choice_rows if row["method"] == "relevant"],
    }
    latest_rows = methods["latest"]
    relevant_rows = methods["relevant"]
    grouped: dict[int, dict[str, list[tuple[float, float]]]] = {int(h): {"no_message": [], "latest": [], "relevant": []} for h in protocol["horizons_s"]}
    inbox_errors: dict[int, dict[str, list[tuple[float, float]]]] = {int(h): {"latest": [], "relevant": []} for h in protocol["horizons_s"]}
    control_gap = 0.0
    for decision in decisions:
        now = float(decision["t"])
        inboxes = {
            "latest": _inbox(latest_rows, now, airtime),
            "relevant": _inbox(relevant_rows, now, airtime),
        }
        differ = inboxes["latest"] != inboxes["relevant"]
        for horizon, actual in decision["actual"].items():
            point = {"x": actual[0], "y": actual[1], "t": now + int(horizon), "id": -1}
            ripple = _ripple(condition, field, point["x"], point["y"], point["t"])
            delta = (0.0, 0.0) if deltas is None else deltas[(int(now + int(horizon)), 1)]
            truth = (
                float(condition["ax"]) + ripple[0] + delta[0],
                float(condition["ay"]) + ripple[1] + delta[1],
            )
            estimates = {"no_message": ripple}
            for method in ("latest", "relevant"):
                observations = []
                for sample_id in inboxes[method]:
                    sample = by_id[sample_id]
                    sample_delta = (0.0, 0.0) if deltas is None else deltas[(int(sample["t"]), 0)]
                    observations.append(
                        {
                            "id": sample_id,
                            "x": sample["x"],
                            "y": sample["y"],
                            "t": sample["t"],
                            "rx": float(sample["rx"]) + sample_delta[0],
                            "ry": float(sample["ry"]) + sample_delta[1],
                        }
                    )
                mean = posterior_mean(observations, point, kernel)
                estimates[method] = (ripple[0] + mean[0], ripple[1] + mean[1])
                if deltas is None and observations:
                    closed_x = global_mean([item["rx"] for item in observations], kernel["prior_std_m_s"], kernel["sigma_m_s"])
                    closed_y = global_mean([item["ry"] for item in observations], kernel["prior_std_m_s"], kernel["sigma_m_s"])
                    control_gap = max(control_gap, abs(mean[0] - closed_x), abs(mean[1] - closed_y))
                    other = posterior_mean(observations, {"x": point["x"] + 1000.0, "y": point["y"] - 1000.0, "t": point["t"] + 15.0, "id": -2}, kernel)
                    control_gap = max(control_gap, abs(mean[0] - other[0]), abs(mean[1] - other[1]))
            for method, estimate in estimates.items():
                error = (estimate[0] - truth[0], estimate[1] - truth[1])
                grouped[int(horizon)][method].append(error)
                if differ and method in inbox_errors[int(horizon)]:
                    inbox_errors[int(horizon)][method].append(error)
    if deltas is None and control_gap > 1e-6:
        raise RuntimeError(f"zero-amplitude posterior left the location-free mean by {control_gap}")
    rows = []
    for horizon in protocol["horizons_s"]:
        scores = {method: _rmse(grouped[int(horizon)][method]) for method in ("no_message", "latest", "relevant")}
        inbox_n = len(inbox_errors[int(horizon)]["latest"])
        inbox_paired = None
        if inbox_n:
            inbox_paired = _rmse(inbox_errors[int(horizon)]["relevant"]) - _rmse(inbox_errors[int(horizon)]["latest"])
        rows.append(
            {
                "field": "control" if deltas is None else "local",
                "seed": seed,
                "horizon_s": int(horizon),
                "n": len(grouped[int(horizon)]["latest"]),
                "rmse_no_message": scores["no_message"],
                "rmse_latest": scores["latest"],
                "rmse_relevant": scores["relevant"],
                "paired_relevant_minus_latest": scores["relevant"] - scores["latest"],
                "inbox_differs_n": inbox_n,
                "inbox_differs_paired": inbox_paired,
                "leg_error_m": sum(decision["leg_error"][int(horizon)] for decision in decisions if int(horizon) in decision["leg_error"])
                / max(1, sum(1 for decision in decisions if int(horizon) in decision["leg_error"])),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    protocol, digest = _load_protocol()
    out = out_dir or (ROOT / "results" / "wind_observation_w5" / RUN_ID)
    if "wind_observation_w5" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w5")
    out.mkdir(parents=True, exist_ok=True)
    _w4, case = load_w4()
    condition = _condition(case, str(protocol["condition"]))
    spec = case["spec"]
    mission = float(spec["time"]["mission_s"])
    decision_s = float(spec["time"]["decision_s"])
    if decision_s != float(protocol["decision_s"]):
        raise RuntimeError("decision interval changed")
    airtime = float(message_cost_j(int(protocol["payload_bytes"]))["airtime_s"])
    if airtime >= decision_s:
        raise RuntimeError("airtime does not fit before the next decision")
    print("replay", flush=True)
    played, poses, committed = _replay(case, condition)
    outcome = assess(played["world"], spec, set(), 3, 0.0)
    trajectory_match = (
        abs(float(outcome["on_time_bits"]) - ARCHIVED_BITS) < 1e-6
        and abs(float(outcome["energy_j"]) - ARCHIVED_ENERGY_J) < 1e-4
    )
    if not trajectory_match:
        raise RuntimeError(f"frozen trajectory drifted bits={outcome['on_time_bits']} energy={outcome['energy_j']}")
    receiver = int(protocol["receiver_uav"])
    observer = int(protocol["observer_uav"])
    for state in played["states"]:
        t_now = int(round(float(state["t_s"])))
        for uav in state["world"].uavs:
            if t_now == 0:
                poses[(0, int(uav.index))] = (float(uav.x), float(uav.y))
            else:
                recorded = poses[(t_now, int(uav.index))]
                if math.hypot(recorded[0] - uav.x, recorded[1] - uav.y) > 1e-6:
                    raise RuntimeError("recorded pose left the decision snapshot")
    final_t = int(round(float(played["world"].t)))
    for uav in played["world"].uavs:
        recorded = poses[(final_t, int(uav.index))]
        if math.hypot(recorded[0] - uav.x, recorded[1] - uav.y) > 1e-6:
            raise RuntimeError("recorded pose left the final snapshot")
    sample_period = int(protocol["sample_period_s"])
    samples = []
    sample_t = 0
    while sample_t <= mission - decision_s + 1e-9:
        x_m, y_m = poses[(sample_t, observer)]
        samples.append({"id": len(samples), "t": float(sample_t), "x": x_m, "y": y_m})
        sample_t += sample_period
    separations = []
    decisions = []
    for state in played["states"]:
        t_now = int(round(float(state["t_s"])))
        uavs = state["world"].uavs
        separations.append(math.hypot(uavs[0].x - uavs[1].x, uavs[0].y - uavs[1].y))
        horizons = [int(h) for h in protocol["horizons_s"] if t_now + int(h) <= mission + 1e-9]
        predicted = _extrapolate(uavs[receiver], committed[(t_now, receiver)], spec, horizons)
        actual = {}
        leg_error = {}
        for horizon in horizons:
            actual_xy = poses[(t_now + horizon, receiver)]
            actual[horizon] = actual_xy
            pred_xy = predicted[horizon]
            leg_error[horizon] = math.hypot(pred_xy[0] - actual_xy[0], pred_xy[1] - actual_xy[1])
        decisions.append(
            {
                "t": float(t_now),
                "predicted": predicted,
                "actual": actual,
                "leg_error": leg_error,
            }
        )
    if len(decisions) != 60 or len(samples) <= len(decisions):
        raise RuntimeError("sample buffer does not outgrow the send opportunities")
    noise = random.Random(str(protocol["noise_token"]))
    sigma = float(protocol["observation_sigma_m_s"])
    for sample in samples:
        # Residual after the known ripple. The hidden local draw is added only in scoring.
        sample["rx"] = float(condition["ax"]) + noise.gauss(0.0, sigma)
        sample["ry"] = float(condition["ay"]) + noise.gauss(0.0, sigma)
    points = []
    for sample in samples:
        points.append({"key": (int(sample["t"]), 0), "id": int(sample["id"]), "x": sample["x"], "y": sample["y"], "t": sample["t"]})
    score_times = sorted({int(decision["t"] + horizon) for decision in decisions for horizon in decision["actual"]})
    for t_score in score_times:
        x_m, y_m = poses[(t_score, receiver)]
        points.append({"key": (t_score, 1), "id": -1, "x": x_m, "y": y_m, "t": float(t_score)})
    local_kernel = _kernel(protocol, protocol["local_field"]["amplitude_m_s"])
    control_kernel = _kernel(protocol, 0.0)
    print("choices", flush=True)
    control_choices = _choice_table(samples, decisions, control_kernel, airtime)
    local_choices = _choice_table(samples, decisions, local_kernel, airtime)
    for left, right in zip(control_choices, local_choices):
        if int(left["latest_id"]) != int(right["latest_id"]) or float(left["t_s"]) != float(right["t_s"]):
            raise RuntimeError("latest sample depended on the kernel")
    if any(row["differed"] for row in control_choices):
        raise RuntimeError("zero amplitude changed the choice")
    for row in control_choices:
        row["field"] = "control"
        row["method"] = "relevant"
    for row in local_choices:
        row["field"] = "local"
        row["method"] = "relevant"
    latest_rows = []
    for source in control_choices:
        latest_rows.append({**source, "method": "latest", "chosen_id": source["latest_id"], "chosen_t_s": source["latest_t_s"], "differed": False})
    choice_rows = control_choices + local_choices + latest_rows
    print("fields", flush=True)
    deltas = _draw_deltas(points, local_kernel, [int(seed) for seed in protocol["local_field"]["seeds"]], float(protocol["local_field"]["generation_jitter"]))
    scores = _evaluate(protocol, case, condition, samples, decisions, [row for row in choice_rows if row["field"] == "control" or row["method"] == "latest"], None, None, control_kernel)
    for seed, table in deltas.items():
        tagged = []
        for row in local_choices:
            tagged.append({**row, "method": "relevant"})
        for row in latest_rows:
            tagged.append({**row, "field": "local", "method": "latest"})
        scores.extend(_evaluate(protocol, case, condition, samples, decisions, tagged, table, seed, local_kernel))
    floor = float(protocol["support_improvement_m_s"])
    supported = []
    for horizon in protocol["horizons_s"]:
        paired = [
            row["paired_relevant_minus_latest"]
            for row in scores
            if row["field"] == "local" and int(row["horizon_s"]) == int(horizon)
        ]
        if paired and all(value < -floor for value in paired):
            supported.append(int(horizon))
    step_lengths = [
        math.hypot(samples[index + 1]["x"] - samples[index]["x"], samples[index + 1]["y"] - samples[index]["y"])
        for index in range(len(samples) - 1)
    ]
    legs = {30: [], 60: []}
    for decision in decisions:
        for horizon, length_rows in legs.items():
            if horizon in decision["actual"]:
                origin = poses[(int(decision["t"]), receiver)]
                length_rows.append(math.hypot(decision["actual"][horizon][0] - origin[0], decision["actual"][horizon][1] - origin[1]))
    differed = [row for row in local_choices if row["differed"]]
    advantages = [float(row["latest_variance_advantage"]) for row in local_choices if row["latest_variance_advantage"] is not None]
    if advantages and not differed and min(advantages) < -TIE_VARIANCE:
        raise RuntimeError("variance ranking disagreed with the recorded choice")
    leg_nonzero = {horizon: sum(1 for decision in decisions if decision["leg_error"].get(horizon, 0.0) > 1e-6) for horizon in (30, 60)}
    leg_max = {horizon: max((decision["leg_error"][horizon] for decision in decisions if horizon in decision["leg_error"]), default=0.0) for horizon in (30, 60)}
    summary = {
        "schema_version": 1,
        "parent_commit": _parent(),
        "config_sha256": digest,
        "config_status": protocol["status"],
        "trajectory_match": True,
        "on_time_bits": float(outcome["on_time_bits"]),
        "energy_j": float(outcome["energy_j"]),
        "n_decisions": len(decisions),
        "n_samples": len(samples),
        "payload_bytes": int(protocol["payload_bytes"]),
        "airtime_s": airtime,
        "send_count": {"no_message": 0, "latest": len(decisions), "relevant": len(decisions)},
        "future_choice_count": 0,
        "default_changed": False,
        "single_sample_times_s": [row["t_s"] for row in local_choices if row["available"] < 2],
        "geometry": {
            "separation_min_m": min(separations),
            "separation_median_m": _quantile(separations, 0.5),
            "separation_max_m": max(separations),
            "sample_step_median_m": _quantile(step_lengths, 0.5),
            "sample_step_max_m": max(step_lengths),
            "receiver_leg_30_median_m": _quantile(legs[30], 0.5),
            "receiver_leg_30_max_m": max(legs[30]),
            "receiver_leg_60_median_m": _quantile(legs[60], 0.5),
            "receiver_leg_60_max_m": max(legs[60]),
        },
        "choices": {
            "control": {
                "possible": sum(1 for row in control_choices if row["available"] >= 2),
                "differed": sum(1 for row in control_choices if row["differed"]),
            },
            "local": {
                "possible": sum(1 for row in local_choices if row["available"] >= 2),
                "differed": len(differed),
                "mean_chosen_distance_m": (sum(row["chosen_distance_m"] for row in differed) / len(differed)) if differed else None,
                "mean_latest_distance_m": (sum(row["latest_distance_m"] for row in differed) / len(differed)) if differed else None,
                "mean_chosen_age_s": (sum(row["chosen_age_s"] for row in differed) / len(differed)) if differed else None,
                "latest_variance_advantage_min": min(advantages) if advantages else None,
                "latest_variance_advantage_median": _quantile(advantages, 0.5) if advantages else None,
            },
        },
        "leg_error_max_m": {str(horizon): leg_max[horizon] for horizon in (30, 60)},
        "leg_error_nonzero": {str(horizon): leg_nonzero[horizon] for horizon in (30, 60)},
        "scores": scores,
        "supported_horizons_s": supported,
        "continue_observation": bool(supported),
        "elapsed_s": time.perf_counter() - started,
    }
    _write_csv(out / "scores.csv", scores)
    public_choices = []
    for row in control_choices + local_choices:
        public_choices.append({key: row[key] for key in row if key != "method"})
    _write_csv(out / "choices.csv", public_choices)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"match {summary['trajectory_match']} differed {summary['choices']['local']['differed']} "
        f"advantage {summary['choices']['local']['latest_variance_advantage_min']} supported {supported}",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run()
