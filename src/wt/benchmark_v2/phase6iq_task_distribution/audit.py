"""Task-distribution audit for the frozen low-level trainer and scheme C."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
from wt.benchmark_v2.phase6e.generators import build_pools
from wt.benchmark_v2.phase6g_lowlevel_mappo.evaluate import _actor_move
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, low_level_reward, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import load_primary
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _key
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import clip_unit_step
from wt.lowlevel_v2.observation import build_actor_observation, communication_features
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL, PPO

from .protocol import (
    BIT_GAP,
    BIT_WINDOWS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DECISION_TIMES,
    DISTANCE_GAP_M,
    EVAL_POOL,
    HOLD,
    INTERFERENCE_REL_GAP,
    OFFICIAL_ALPHA,
    OUTPUT,
    QUEUE_REL_GAP,
    RADIUS_GAP,
    ROLLOUT_DRAWS,
    SINR_GAP,
    SOURCE_ACTOR,
    STATIC_DRAWS,
    SUPPORT_OUTSIDE_LIMIT,
    TRAIN_PROBE,
    assert_audit_scenario,
    assert_phase_output,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _restore(env: L3BBitQueueEnvV2, snap: dict, initial_q: np.ndarray) -> None:
    restore_state(env, snap)
    env.initial_q_bits = np.asarray(initial_q, dtype=np.float64).copy()


def _features(env: L3BBitQueueEnvV2, assignment: np.ndarray, remaining_hold: int) -> dict:
    chosen = np.asarray(assignment, dtype=np.int64).copy()
    saved = np.asarray(env.assignment, dtype=np.int64).copy()
    env.assignment[:] = chosen
    comm = communication_features(env)
    env.assignment[:] = saved
    radius = float(env.config.service_radius_m)
    distances = []
    queues = []
    inside = []
    sinr = []
    interference = []
    for uav, buoy in enumerate(chosen):
        if int(buoy) < 0:
            continue
        dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
        distances.append(dist)
        queues.append(float(env.q_bits[int(buoy)]))
        inside.append(float(dist <= radius))
        sinr.append(float(comm["sinr"][uav]))
        interference.append(float(comm["total_interference"][uav]))
    active = len(distances)
    return {
        "n_active": int(active),
        "idle_fraction": float(1.0 - active / max(int(env.config.num_uavs), 1)),
        "distance_m": float(np.mean(distances)) if distances else 0.0,
        "in_radius_fraction": float(np.mean(inside)) if inside else 0.0,
        "queue_bits": float(np.mean(queues)) if queues else 0.0,
        "remaining_hold": int(remaining_hold),
        "sinr": float(np.mean(sinr)) if sinr else 0.0,
        "interference": float(np.mean(interference)) if interference else 0.0,
    }


def _rollout_bits(env: L3BBitQueueEnvV2, actor, assignment: np.ndarray, steps: int) -> float:
    snap = capture_state(env)
    initial_q = np.asarray(env.initial_q_bits, dtype=np.float64).copy()
    option = make_option(env, np.asarray(assignment, dtype=np.int64))
    if not np.array_equal(option.association, option.targets):
        _restore(env, snap, initial_q)
        raise RuntimeError("INTERFACE")
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    served = 0.0
    payload_before = float(env.payload_bits)
    for _ in range(int(steps)):
        if int(env.t) >= int(env.config.episode_horizon):
            break
        movement, _flag, _norm = _actor_move(env, actor, option, last)
        info = env.step(movement, np.asarray(option.association, dtype=np.int64))
        if float(np.max(np.abs(env.conservation_error_bits()))) > float(env.tol_bits):
            _restore(env, snap, initial_q)
            raise RuntimeError("ACCOUNTING")
        served += float(np.sum(info["served_bits"]))
        last = np.asarray(movement, dtype=np.float64)
        option.remaining_hold -= 1
    if abs(float(env.payload_bits) - payload_before - served) > float(env.tol_bits):
        _restore(env, snap, initial_q)
        raise RuntimeError("ACCOUNTING")
    _restore(env, snap, initial_q)
    return float(served)


def _sample(pool: list[np.ndarray], count: int, rng: np.random.Generator) -> list[np.ndarray]:
    if not pool:
        raise RuntimeError("empty candidate pool")
    choice = rng.choice(len(pool), size=min(int(count), len(pool)), replace=False)
    return [np.asarray(pool[int(index)], dtype=np.int64).copy() for index in choice]


def _mean_features(env, assignments: list[np.ndarray], hold: int) -> dict:
    rows = [_features(env, item, hold) for item in assignments]
    keys = ("n_active", "idle_fraction", "distance_m", "in_radius_fraction", "queue_bits", "sinr", "interference")
    out = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    out["remaining_hold"] = int(hold)
    out["samples"] = int(len(rows))
    return out


def _decision_row(env, actor, assignment, steps: int, *, source: str, scenario: int, hold: int, in_k5: int, pool_k5: int, pool_k6: int) -> dict:
    feat = _features(env, assignment, hold)
    bits = _rollout_bits(env, actor, assignment, steps) if int(steps) > 0 else 0.0
    return {
        "scenario_id": int(scenario),
        "time": int(env.t),
        "source": source,
        "window_steps": int(steps),
        "in_k5_pool": int(in_k5),
        "pool_k5": int(pool_k5),
        "pool_k6": int(pool_k6),
        "bits15": float(bits),
        **feat,
    }


def audit_scenarios(planner: FormalPlanner, scenarios: tuple[int, ...], rng: np.random.Generator) -> list[dict]:
    actor = planner.actor
    actor.eval()
    rows = []
    for scenario in scenarios:
        assert_audit_scenario(int(scenario))
        env = L3BBitQueueEnvV2()
        install_light_info(env)
        env.reset(seed=int(scenario))
        position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
        predecessor = None
        limit = int(env.config.episode_horizon)
        while int(env.t) < limit:
            if int(env.t) not in DECISION_TIMES:
                raise RuntimeError("decision time left the hold grid")
            selection = planner.select_assignment(env, position_prev, predecessor)
            if selection.status != "selected" or selection.assignment is None or selection.option is None:
                raise RuntimeError(selection.reason or selection.status)
            pools, _packed, illegal = build_pools(env, (("G1_RATE", 5), ("G1_RATE", 6)))
            if int(illegal) != 0:
                raise RuntimeError("ILLEGAL")
            k5 = pools[("G1_RATE", 5)]
            k6 = pools[("G1_RATE", 6)]
            k5_keys = {_key(item) for item in k5}
            chosen = np.asarray(selection.assignment, dtype=np.int64)
            chosen_key = _key(chosen)
            steps = min(int(HOLD), limit - int(env.t))
            state_t = int(env.t)
            state_payload = float(env.payload_bits)
            state_queue = np.asarray(env.q_bits, dtype=np.float64).copy()
            formal = _decision_row(
                env,
                actor,
                chosen,
                steps,
                source="scheme_c",
                scenario=int(scenario),
                hold=int(selection.option.remaining_hold),
                in_k5=int(chosen_key in k5_keys),
                pool_k5=len(k5),
                pool_k6=len(k6),
            )
            if not np.array_equal(selection.option.association, selection.option.targets):
                raise RuntimeError("INTERFACE")
            if int(selection.option.remaining_hold) != int(HOLD):
                raise RuntimeError("INTERFACE")
            k5_draw = _sample(k5, STATIC_DRAWS, rng)
            k6_draw = _sample(k6, STATIC_DRAWS, rng)
            train_feat = _mean_features(env, k5_draw, int(HOLD))
            wide_feat = _mean_features(env, k6_draw, int(HOLD))
            train_bits = float(np.mean([_rollout_bits(env, actor, item, steps) for item in k5_draw[:ROLLOUT_DRAWS]]))
            wide_bits = float(np.mean([_rollout_bits(env, actor, item, steps) for item in k6_draw[:ROLLOUT_DRAWS]]))
            if int(env.t) != state_t or abs(float(env.payload_bits) - state_payload) > 1.0e-6 or not np.allclose(env.q_bits, state_queue):
                raise RuntimeError("INTERFACE")
            formal["train_k5"] = {**train_feat, "bits15": train_bits}
            formal["uniform_k6"] = {**wide_feat, "bits15": wide_bits}
            rows.append(formal)
            option = selection.option
            last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
            live_served = 0.0
            for _ in range(steps):
                position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
                predecessor = int(env.t)
                movement, _flag, _norm = _actor_move(env, actor, option, last)
                info = env.step(movement, np.asarray(option.association, dtype=np.int64))
                if float(np.max(np.abs(env.conservation_error_bits()))) > float(env.tol_bits):
                    raise RuntimeError("ACCOUNTING")
                if abs(float(info["payload_increment_bits"]) - float(np.sum(info["served_bits"]))) > float(env.tol_bits):
                    raise RuntimeError("ACCOUNTING")
                live_served += float(np.sum(info["served_bits"]))
                last = np.asarray(movement, dtype=np.float64)
                option.remaining_hold -= 1
            if abs(live_served - float(formal["bits15"])) > float(env.tol_bits):
                raise RuntimeError("ACCOUNTING")
            if int(env.t) < limit and int(option.remaining_hold) != 0:
                raise RuntimeError("INTERFACE")
        print(f"audited {scenario} decisions={sum(row['scenario_id'] == int(scenario) for row in rows)}", flush=True)
    return rows


def _bootstrap_mean(values: np.ndarray) -> dict:
    rng = np.random.default_rng(int(BOOTSTRAP_SEED))
    draws = rng.integers(0, len(values), size=(int(BOOTSTRAP_DRAWS), len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _scenario_gap(rows: list[dict], field: str, other: str, *, times: tuple[int, ...] | None) -> dict:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if times is not None and int(row["time"]) not in times:
            continue
        left = float(row[field]) if other == "scheme_c" else float(row[other][field])
        right = float(row["train_k5"][field])
        if other != "scheme_c":
            left = float(row[other][field])
        grouped.setdefault(int(row["scenario_id"]), []).append(left - right)
    paired = np.asarray([float(np.mean(items)) for items in grouped.values()], dtype=np.float64)
    summary = _bootstrap_mean(paired)
    summary["field"] = field
    summary["comparison"] = f"{other} minus train_k5"
    return summary


def _gap_triggered(summary: dict, absolute: float, *, relative_base: float | None = None) -> bool:
    mean = float(summary["mean"])
    excludes = float(summary["ci_low"]) > 0.0 or float(summary["ci_high"]) < 0.0
    if relative_base is None:
        return bool(excludes and abs(mean) > float(absolute))
    base = max(abs(float(relative_base)), 1.0e-9)
    return bool(excludes and abs(mean) / base > float(absolute))


def interface_report(actor) -> dict:
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=321096)
    pools, _packed, illegal = build_pools(env, (("G1_RATE", 5),))
    if int(illegal) != 0:
        raise RuntimeError("ILLEGAL")
    assignment = np.asarray(pools[("G1_RATE", 5)][0], dtype=np.int64)
    option = make_option(env, assignment)
    obs = build_actor_observation(env, option, np.zeros((env.config.num_uavs, 2)))
    with torch.no_grad():
        mu, _log_std = actor(
            torch.as_tensor(obs["self"], dtype=torch.float32),
            torch.as_tensor(obs["task"], dtype=torch.float32),
            torch.as_tensor(obs["communication"], dtype=torch.float32),
            torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
            torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
        )
    pre = mu.detach().cpu().numpy().astype(np.float64)
    delta = float(ALPHA_RESIDUAL) * np.tanh(pre)
    desired = clip_unit_step(fly_toward(env, option.association) + delta)
    movement, _flag, _norm = _actor_move(env, actor, option, np.zeros((env.config.num_uavs, 2)))
    zeros = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    task, comm = low_level_reward(1.0e6, zeros, zeros, zeros, env.block_bits)
    hold_sequence = [int(HOLD) - step for step in range(int(HOLD))]
    return {
        "same_make_option": True,
        "association_equals_targets": bool(np.array_equal(option.association, option.targets)),
        "initial_hold": int(option.remaining_hold),
        "hold_sequence": hold_sequence,
        "alpha": float(ALPHA_RESIDUAL),
        "gamma": float(PPO.gamma),
        "deterministic_move_matches_formula": bool(np.allclose(movement, desired)),
        "train_action_samples_gaussian": True,
        "formal_action_uses_mean": True,
        "queue_update": "L3BBitQueueEnvV2.step apply_service_then_arrival",
        "service_window_steps": int(HOLD),
        "task_reward_per_million_bits": float(task),
        "comm_reward_per_million_bits": float(comm),
        "mapping_equal": bool(
            np.array_equal(option.association, option.targets)
            and int(option.remaining_hold) == int(HOLD)
            and np.allclose(movement, desired)
            and abs(float(ALPHA_RESIDUAL) - float(OFFICIAL_ALPHA)) <= 1.0e-12
        ),
    }


def _level(rows: list[dict]) -> dict:
    def col(source: str, field: str, times: tuple[int, ...] | None = None) -> float:
        picked = []
        for row in rows:
            if times is not None and int(row["time"]) not in times:
                continue
            picked.append(float(row[field] if source == "scheme_c" else row[source][field]))
        return float(np.mean(picked)) if picked else 0.0

    fields = ("distance_m", "in_radius_fraction", "queue_bits", "remaining_hold", "sinr", "interference", "idle_fraction", "n_active")
    table = {}
    for source in ("scheme_c", "train_k5", "uniform_k6"):
        table[source] = {field: col(source, field) for field in fields}
        table[source]["bits15"] = col(source, "bits15", BIT_WINDOWS)
    outside = [1 - int(row["in_k5_pool"]) for row in rows]
    table["scheme_c_outside_k5_fraction"] = float(np.mean(outside)) if outside else 0.0
    table["mean_pool_k5"] = float(np.mean([row["pool_k5"] for row in rows]))
    table["mean_pool_k6"] = float(np.mean([row["pool_k6"] for row in rows]))
    return table


def judge(rows: list[dict]) -> dict:
    bits = _scenario_gap(rows, "bits15", "scheme_c", times=BIT_WINDOWS)
    distance = _scenario_gap(rows, "distance_m", "scheme_c", times=None)
    radius = _scenario_gap(rows, "in_radius_fraction", "scheme_c", times=None)
    queue = _scenario_gap(rows, "queue_bits", "scheme_c", times=None)
    sinr = _scenario_gap(rows, "sinr", "scheme_c", times=None)
    interference = _scenario_gap(rows, "interference", "scheme_c", times=None)
    wide_bits = _scenario_gap(rows, "bits15", "uniform_k6", times=BIT_WINDOWS)
    outside = _bootstrap_mean(np.asarray([
        float(np.mean([1 - int(row["in_k5_pool"]) for row in rows if int(row["scenario_id"]) == scenario]))
        for scenario in sorted({int(row["scenario_id"]) for row in rows})
    ], dtype=np.float64))
    queue_base = float(np.mean([row["train_k5"]["queue_bits"] for row in rows]))
    interference_base = float(np.mean([row["train_k5"]["interference"] for row in rows]))
    triggered = {
        "bits15": _gap_triggered(bits, BIT_GAP),
        "distance_m": _gap_triggered(distance, DISTANCE_GAP_M),
        "in_radius_fraction": _gap_triggered(radius, RADIUS_GAP),
        "queue_bits": _gap_triggered(queue, QUEUE_REL_GAP, relative_base=queue_base),
        "sinr": _gap_triggered(sinr, SINR_GAP),
        "interference": _gap_triggered(interference, INTERFERENCE_REL_GAP, relative_base=interference_base),
        "outside_k5": bool(float(outside["mean"]) > float(SUPPORT_OUTSIDE_LIMIT) and (float(outside["ci_low"]) > float(SUPPORT_OUTSIDE_LIMIT))),
    }
    feature_hits = sum(int(triggered[name]) for name in ("distance_m", "in_radius_fraction", "queue_bits", "sinr", "interference"))
    confirmed = bool(triggered["bits15"] or feature_hits >= 2 or (triggered["outside_k5"] and feature_hits >= 1))
    return {
        "confirmed": confirmed,
        "triggered": triggered,
        "scheme_c_minus_train_k5": {
            "bits15": bits,
            "distance_m": distance,
            "in_radius_fraction": radius,
            "queue_bits": queue,
            "sinr": sinr,
            "interference": interference,
            "outside_k5": outside,
        },
        "uniform_k6_minus_train_k5_bits15": wide_bits,
    }


def seed_analysis() -> dict:
    root = Path(__file__).resolve().parents[4] / "results" / "phase6ip_potential_shaping"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    payload = json.loads((root / "scenario_rows.json").read_text(encoding="utf-8"))
    baseline = {int(row["scenario_id"]): row for row in payload["baseline"]}
    seeds = []
    for item in payload["trained"]:
        name = Path(item["checkpoint"]).name
        deltas = []
        for row in item["rows"]:
            base = baseline[int(row["scenario_id"])]
            deltas.append(
                {
                    "scenario_id": int(row["scenario_id"]),
                    "delta_payload": float(row["payload_bits"]) - float(base["payload_bits"]),
                    "task_reward": float(row["task_reward_sum"]),
                    "shaping": float(row["shaping_sum"]),
                    "baseline_task_reward": float(base["task_reward_sum"]),
                    "baseline_shaping": float(base["shaping_sum"]),
                    "sinr": float(row["sinr_mean"]),
                    "baseline_sinr": float(base["sinr_mean"]),
                    "service_steps": int(row["service_steps"]),
                    "baseline_service_steps": int(base["service_steps"]),
                    "entry": float(row["mean_first_entry_steps"]),
                    "baseline_entry": float(base["mean_first_entry_steps"]),
                }
            )
        values = np.asarray([row["delta_payload"] for row in deltas], dtype=np.float64)
        shaping_gap = np.asarray([row["shaping"] - row["baseline_shaping"] for row in deltas], dtype=np.float64)
        task_gap = np.asarray([row["task_reward"] - row["baseline_task_reward"] for row in deltas], dtype=np.float64)
        positive = values[values > 1.0]
        hist = json.loads((root / "checkpoints" / name.replace(".pt", ".history.json")).read_text(encoding="utf-8"))
        early = hist[:10]
        late = hist[-10:]
        seeds.append(
            {
                "checkpoint": name,
                "sha256": item["sha256"],
                "paired_mean": float(item["paired"]["mean"]),
                "ci_low": float(item["paired"]["ci_low"]),
                "ci_high": float(item["paired"]["ci_high"]),
                "positive": int(item["paired"]["positive"]),
                "negative": int(item["paired"]["negative"]),
                "top_positive_share": float(item["top_positive_share"]),
                "mean_without_best": float(item["mean_without_best"]),
                "mean_task_reward": float(np.mean([row["task_reward"] for row in deltas])),
                "mean_shaping": float(np.mean([row["shaping"] for row in deltas])),
                "mean_baseline_task_reward": float(np.mean([row["baseline_task_reward"] for row in deltas])),
                "mean_baseline_shaping": float(np.mean([row["baseline_shaping"] for row in deltas])),
                "shaping_fraction_of_task": float(np.mean([row["shaping"] / max(row["task_reward"], 1.0e-9) for row in deltas])),
                "payload_delta_vs_task_gap_corr": float(np.corrcoef(values, task_gap)[0, 1]),
                "payload_delta_vs_shaping_gap_corr": float(np.corrcoef(values, shaping_gap)[0, 1]),
                "mean_shaping_gap": float(np.mean(shaping_gap)),
                "mean_task_gap": float(np.mean(task_gap)),
                "largest_positive": max(deltas, key=lambda row: row["delta_payload"]),
                "largest_negative": min(deltas, key=lambda row: row["delta_payload"]),
                "positive_mass": float(np.sum(positive)) if positive.size else 0.0,
                "train_served_early": float(np.mean([row["mean_served_bits"] for row in early])),
                "train_served_late": float(np.mean([row["mean_served_bits"] for row in late])),
                "train_shaping_early": float(np.mean([row["mean_shaping"] for row in early])),
                "train_shaping_late": float(np.mean([row["mean_shaping"] for row in late])),
                "train_task_early": float(np.mean([row["mean_task_reward"] for row in early])),
                "train_task_late": float(np.mean([row["mean_task_reward"] for row in late])),
                "train_entropy_early": float(np.mean([row["entropy"] for row in early])),
                "train_entropy_late": float(np.mean([row["entropy"] for row in late])),
                "train_grad_mean": float(np.mean([row["gradient_norm_preclip"] for row in hist])),
            }
        )
    by_scenario = {}
    for seed in seeds:
        pass
    matrices = []
    for item in payload["trained"]:
        ordered = sorted(item["rows"], key=lambda row: int(row["scenario_id"]))
        matrices.append(np.asarray([
            float(row["payload_bits"]) - float(baseline[int(row["scenario_id"])]["payload_bits"]) for row in ordered
        ], dtype=np.float64))
    corr = np.corrcoef(np.stack(matrices, axis=0))
    signs = np.sign(np.stack(matrices, axis=0))
    return {
        "source_summary_gates": summary["gates"],
        "seeds": seeds,
        "delta_correlation": corr.tolist(),
        "all_positive": int(np.sum(np.all(signs > 0, axis=0))),
        "all_negative": int(np.sum(np.all(signs < 0, axis=0))),
        "seed0_only_positive": int(np.sum((signs[0] > 0) & (signs[1] < 0) & (signs[2] < 0))),
    }


def run_audit() -> dict:
    before = _sha256(SOURCE_ACTOR)
    actor = load_primary()
    interface = interface_report(actor)
    if not interface["mapping_equal"]:
        raise RuntimeError("INTERFACE")
    planner = FormalPlanner("scheme_c")
    if _sha256(SOURCE_ACTOR) != before:
        raise RuntimeError("source checkpoint changed")
    rng = np.random.default_rng(0)
    dev_rows = audit_scenarios(planner, EVAL_POOL, rng)
    train_rows = audit_scenarios(planner, TRAIN_PROBE, rng)
    verdict = judge(dev_rows)
    outcome = "TASK_DISTRIBUTION_MISMATCH_CONFIRMED" if verdict["confirmed"] else "TASK_DISTRIBUTION_MISMATCH_NOT_CONFIRMED"
    return {
        "gates": ["TASK_DISTRIBUTION_AUDIT_COMPLETE", outcome],
        "source_sha256": before,
        "source_unchanged": int(_sha256(SOURCE_ACTOR) == before),
        "interface": interface,
        "dev_levels": _level(dev_rows),
        "train_probe_levels": _level(train_rows),
        "verdict": verdict,
        "dev_rows": dev_rows,
        "train_rows": train_rows,
        "seeds": seed_analysis(),
        "replaced_official_low_level": False,
        "training_started": False,
    }


__all__ = ["run_audit"]
