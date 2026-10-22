"""Collect scheme-C decisions. Recorded window totals are not training rewards."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
from wt.benchmark_v2.phase6e.generators import build_pools, is_legal
from wt.benchmark_v2.phase6g_lowlevel_mappo.evaluate import _actor_move
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _key
from wt.benchmark_v2.phase6iy_candidate_adapter.adapter import frozen_candidate_keys
from wt.lowlevel_v2.observation import FORBIDDEN_KEYS, build_actor_observation, communication_features

from .protocol import (
    DATASET_DIR,
    DECISION_TIMES,
    EVAL_POOL,
    HOLD,
    K5_FEATURE_DRAWS,
    K5_ROLLOUT_DRAWS,
    TRAIN_SCENARIOS,
    assert_phase_output,
    assert_split_scenario,
)

STATE_FIELDS = (
    "uav_positions",
    "assignment",
    "progress",
    "served_k",
    "q_bits",
    "queue",
    "service_remainder_bits",
    "completed_block_equivalent",
    "cumulative_served_bits",
    "cumulative_dropped_bits",
    "cumulative_generated_arrival_bits",
    "cumulative_snap_bits",
    "last_sinr",
    "last_rate",
    "legacy_progress_completions",
)


def _restore(env: L3BBitQueueEnvV2, snap: dict, initial_q: np.ndarray) -> None:
    restore_state(env, snap)
    env.initial_q_bits = np.asarray(initial_q, dtype=np.float64).copy()


def _task_features(env: L3BBitQueueEnvV2, assignment: np.ndarray) -> dict:
    chosen = np.asarray(assignment, dtype=np.int64).copy()
    saved = np.asarray(env.assignment, dtype=np.int64).copy()
    env.assignment[:] = chosen
    comm = communication_features(env)
    env.assignment[:] = saved
    radius = float(env.config.service_radius_m)
    distance, inside, queues, sinr, rate, interference = [], [], [], [], [], []
    for uav, buoy in enumerate(chosen):
        if int(buoy) < 0:
            continue
        dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
        distance.append(dist)
        inside.append(float(dist <= radius))
        queues.append(float(env.q_bits[int(buoy)]))
        sinr.append(float(comm["sinr"][uav]))
        rate.append(float(comm["rate"][uav]))
        interference.append(float(comm["total_interference"][uav]))
    active = len(distance)
    return {
        "n_active": int(active),
        "idle_fraction": float(1.0 - active / max(int(env.config.num_uavs), 1)),
        "distance_m": float(np.mean(distance)) if distance else 0.0,
        "in_radius_fraction": float(np.mean(inside)) if inside else 0.0,
        "queue_bits": float(np.mean(queues)) if queues else 0.0,
        "sinr": float(np.mean(sinr)) if sinr else 0.0,
        "rate_bps": float(np.mean(rate)) if rate else 0.0,
        "interference": float(np.mean(interference)) if interference else 0.0,
    }


def _rollout_reference(env, actor, assignment: np.ndarray, steps: int, last_movement: np.ndarray) -> float:
    snap = capture_state(env)
    initial_q = np.asarray(env.initial_q_bits, dtype=np.float64).copy()
    option = make_option(env, np.asarray(assignment, dtype=np.int64))
    last = np.asarray(last_movement, dtype=np.float64).copy()
    served = 0.0
    for _ in range(int(steps)):
        if int(env.t) >= int(env.config.episode_horizon):
            break
        movement, _flag, _norm = _actor_move(env, actor, option, last)
        info = env.step(movement, np.asarray(option.association, dtype=np.int64))
        served += float(np.sum(info["served_bits"]))
        last = np.asarray(movement, dtype=np.float64)
        option.remaining_hold -= 1
    _restore(env, snap, initial_q)
    return float(served)


def collect_split(planner: FormalPlanner, scenarios: tuple[int, ...], split: str, rng: np.random.Generator) -> list[dict]:
    actor = planner.actor
    actor.eval()
    rows = []
    for scenario in scenarios:
        assert_split_scenario(int(scenario), split)
        env = L3BBitQueueEnvV2()
        install_light_info(env)
        env.reset(seed=int(scenario))
        position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
        predecessor = -1
        last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
        limit = int(env.config.episode_horizon)
        while int(env.t) < limit:
            if int(env.t) not in DECISION_TIMES:
                raise RuntimeError("decision time left the hold grid")
            pred = None if int(predecessor) < 0 else int(predecessor)
            select_position_prev = np.asarray(position_prev, dtype=np.float64).copy()
            select_predecessor = int(predecessor)
            selection = planner.select_assignment(env, select_position_prev, pred)
            if selection.status != "selected" or selection.assignment is None or selection.option is None:
                raise RuntimeError(selection.reason or selection.status)
            chosen = np.asarray(selection.assignment, dtype=np.int64).copy()
            if not is_legal(chosen, int(env.config.num_uavs), int(env.config.num_buoys)):
                raise RuntimeError("ILLEGAL")
            if not np.array_equal(selection.option.association, selection.option.targets):
                raise RuntimeError("INTERFACE")
            if int(selection.option.remaining_hold) != int(HOLD):
                raise RuntimeError("INTERFACE")
            keys, _raw = frozen_candidate_keys(env)
            if _key(chosen) not in set(keys) or int(selection.candidate_count) != len(keys):
                raise RuntimeError("scheme C assignment is outside the k=6 pool")
            pools, _packed, illegal = build_pools(env, (("G1_RATE", 5),))
            if int(illegal) != 0:
                raise RuntimeError("ILLEGAL")
            k5 = pools[("G1_RATE", 5)]
            k5_keys = {_key(item) for item in k5}
            draw = [np.asarray(k5[int(index)], dtype=np.int64) for index in rng.choice(len(k5), size=min(K5_FEATURE_DRAWS, len(k5)), replace=False)]
            k5_features = [_task_features(env, item) for item in draw]
            steps = min(int(HOLD), limit - int(env.t))
            snap = capture_state(env)
            initial_q = np.asarray(env.initial_q_bits, dtype=np.float64).copy()
            features = _task_features(env, chosen)
            obs = build_actor_observation(env, selection.option, last)
            if any(key in obs for key in FORBIDDEN_KEYS) or bool(obs["leaks_future_arrivals"]):
                raise RuntimeError("future information reached the actor observation")
            reference_bits = float(
                np.mean([_rollout_reference(env, actor, item, steps, last) for item in draw[:K5_ROLLOUT_DRAWS]])
            )
            if int(env.t) != int(snap["t"]) or not np.allclose(env.q_bits, snap["q_bits"]):
                raise RuntimeError("INTERFACE")
            payload_before = float(env.payload_bits)
            decision_last = np.asarray(last, dtype=np.float64).copy()
            option = selection.option
            served = 0.0
            executed_steps = 0
            for _ in range(steps):
                position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
                predecessor = int(env.t)
                movement, _flag, _norm = _actor_move(env, actor, option, last)
                info = env.step(movement, np.asarray(option.association, dtype=np.int64))
                if float(np.max(np.abs(env.conservation_error_bits()))) > float(env.tol_bits):
                    raise RuntimeError("ACCOUNTING")
                served += float(np.sum(info["served_bits"]))
                last = np.asarray(movement, dtype=np.float64)
                option.remaining_hold -= 1
                executed_steps += 1
            if abs(float(env.payload_bits) - payload_before - served) > float(env.tol_bits):
                raise RuntimeError("ACCOUNTING")
            if int(env.t) != int(snap["t"]) + executed_steps:
                raise RuntimeError("window advanced the wrong number of steps")
            rows.append(
                {
                    "split": split,
                    "scenario_id": int(scenario),
                    "time": int(snap["t"]),
                    "scenario_seed": int(scenario),
                    "window_steps": int(executed_steps),
                    "assignment": chosen,
                    "pool_keys": np.asarray(keys, dtype=np.int8),
                    "pool_hash": str(selection.pool_hash),
                    "pool_count": int(selection.candidate_count),
                    "score_index": int(selection.score_index),
                    "remaining_hold": int(HOLD),
                    "in_k5_pool": int(_key(chosen) in k5_keys),
                    "bits15": float(served),
                    "payload_before": float(payload_before),
                    "payload_after": float(env.payload_bits),
                    "k5_bits15": float(reference_bits),
                    "k5_features": {key: float(np.mean([item[key] for item in k5_features])) for key in k5_features[0]},
                    "features": features,
                    "snap": snap,
                    "initial_q_bits": initial_q,
                    "buoy_positions": np.asarray(env.buoy_positions, dtype=np.float64).copy(),
                    "last_movement": decision_last,
                    "position_prev": select_position_prev,
                    "predecessor": int(select_predecessor),
                }
            )
        print(f"collected {split} {scenario} windows={sum(row['scenario_id'] == int(scenario) for row in rows)}", flush=True)
    return rows


def _stack_state(rows: list[dict], name: str) -> np.ndarray:
    return np.stack([np.asarray(row["snap"][name]) for row in rows])


def save_rows(rows: list[dict], split: str, manifest: dict) -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    path = assert_phase_output(DATASET_DIR / f"{split}.npz")
    pools = [np.asarray(row["pool_keys"], dtype=np.int8) for row in rows]
    counts = {int(item.shape[0]) for item in pools}
    if len(counts) == 1:
        pool_keys = np.stack(pools)
        pool_offsets = np.zeros(len(rows), dtype=np.int64)
    else:
        pool_keys = np.concatenate(pools, axis=0)
        pool_offsets = np.cumsum([0] + [int(item.shape[0]) for item in pools[:-1]], dtype=np.int64)
    payload = {
        "scenario_id": np.asarray([row["scenario_id"] for row in rows], dtype=np.int64),
        "scenario_seed": np.asarray([row["scenario_seed"] for row in rows], dtype=np.int64),
        "time": np.asarray([row["time"] for row in rows], dtype=np.int64),
        "window_steps": np.asarray([row["window_steps"] for row in rows], dtype=np.int64),
        "chosen_assignment": np.stack([row["assignment"] for row in rows]).astype(np.int64),
        "pool_hash": np.asarray([row["pool_hash"] for row in rows]),
        "pool_count": np.asarray([row["pool_count"] for row in rows], dtype=np.int64),
        "score_index": np.asarray([row["score_index"] for row in rows], dtype=np.int64),
        "remaining_hold": np.asarray([row["remaining_hold"] for row in rows], dtype=np.int64),
        "in_k5_pool": np.asarray([row["in_k5_pool"] for row in rows], dtype=np.int64),
        "bits15": np.asarray([row["bits15"] for row in rows], dtype=np.float64),
        "payload_before": np.asarray([row["payload_before"] for row in rows], dtype=np.float64),
        "payload_after": np.asarray([row["payload_after"] for row in rows], dtype=np.float64),
        "k5_bits15": np.asarray([row["k5_bits15"] for row in rows], dtype=np.float64),
        "initial_q_bits": np.stack([row["initial_q_bits"] for row in rows]),
        "buoy_positions": np.stack([row["buoy_positions"] for row in rows]),
        "last_movement": np.stack([row["last_movement"] for row in rows]),
        "position_prev": np.stack([row["position_prev"] for row in rows]),
        "predecessor": np.asarray([row["predecessor"] for row in rows], dtype=np.int64),
        "pool_keys": pool_keys,
        "pool_offsets": pool_offsets,
        "t": np.asarray([int(row["snap"]["t"]) for row in rows], dtype=np.int64),
        "payload_bits": np.asarray([float(row["snap"]["payload_bits"]) for row in rows], dtype=np.float64),
        "dropped": np.asarray([float(row["snap"]["dropped"]) for row in rows], dtype=np.float64),
    }
    for name in STATE_FIELDS:
        payload[f"snap_{name}"] = _stack_state(rows, name)
    for key in ("n_active", "idle_fraction", "distance_m", "in_radius_fraction", "queue_bits", "sinr", "rate_bps", "interference"):
        payload[f"feature_{key}"] = np.asarray([row["features"][key] for row in rows], dtype=np.float64)
        payload[f"k5_{key}"] = np.asarray([row["k5_features"][key] for row in rows], dtype=np.float64)
    np.savez_compressed(path, **payload)
    assert_phase_output(DATASET_DIR / f"{split}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_rows(split: str) -> list[dict]:
    path = DATASET_DIR / f"{split}.npz"
    packed = np.load(path, allow_pickle=False)
    rows = []
    offsets = np.asarray(packed["pool_offsets"], dtype=np.int64)
    keys = np.asarray(packed["pool_keys"])
    dense = keys.ndim == 3
    for index in range(int(packed["scenario_id"].shape[0])):
        snap = {name: np.asarray(packed[f"snap_{name}"][index]).copy() for name in STATE_FIELDS}
        snap["t"] = int(packed["t"][index])
        snap["payload_bits"] = float(packed["payload_bits"][index])
        snap["dropped"] = float(packed["dropped"][index])
        if dense:
            pool = np.asarray(keys[index], dtype=np.int64)
        else:
            start = int(offsets[index])
            stop = int(offsets[index + 1]) if index + 1 < len(offsets) else int(keys.shape[0])
            pool = np.asarray(keys[start:stop], dtype=np.int64)
        rows.append(
            {
                "split": split,
                "scenario_id": int(packed["scenario_id"][index]),
                "scenario_seed": int(packed["scenario_seed"][index]),
                "time": int(packed["time"][index]),
                "window_steps": int(packed["window_steps"][index]),
                "assignment": np.asarray(packed["chosen_assignment"][index], dtype=np.int64).copy(),
                "pool_keys": pool,
                "pool_hash": str(packed["pool_hash"][index]),
                "pool_count": int(packed["pool_count"][index]),
                "score_index": int(packed["score_index"][index]),
                "remaining_hold": int(packed["remaining_hold"][index]),
                "in_k5_pool": int(packed["in_k5_pool"][index]),
                "bits15": float(packed["bits15"][index]),
                "payload_before": float(packed["payload_before"][index]),
                "payload_after": float(packed["payload_after"][index]),
                "k5_bits15": float(packed["k5_bits15"][index]),
                "initial_q_bits": np.asarray(packed["initial_q_bits"][index], dtype=np.float64).copy(),
                "buoy_positions": np.asarray(packed["buoy_positions"][index], dtype=np.float64).copy(),
                "last_movement": np.asarray(packed["last_movement"][index], dtype=np.float64).copy(),
                "position_prev": np.asarray(packed["position_prev"][index], dtype=np.float64).copy(),
                "predecessor": int(packed["predecessor"][index]),
                "snap": snap,
                "features": {
                    key: float(packed[f"feature_{key}"][index])
                    for key in ("n_active", "idle_fraction", "distance_m", "in_radius_fraction", "queue_bits", "sinr", "rate_bps", "interference")
                },
                "k5_features": {
                    key: float(packed[f"k5_{key}"][index])
                    for key in ("n_active", "idle_fraction", "distance_m", "in_radius_fraction", "queue_bits", "sinr", "rate_bps", "interference")
                },
            }
        )
    return rows


def _open_env(row: dict) -> L3BBitQueueEnvV2:
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(row["scenario_id"]))
    if not np.allclose(env.buoy_positions, row["buoy_positions"]):
        raise RuntimeError("INTERFACE")
    _restore(env, row["snap"], row["initial_q_bits"])
    if int(env.t) != int(row["time"]):
        raise RuntimeError("INTERFACE")
    return env


def validate_rows(rows: list[dict], split: str, planner: FormalPlanner) -> dict:
    if not rows:
        raise RuntimeError("empty scheme C dataset")
    allowed = set(TRAIN_SCENARIOS if split == "train" else EVAL_POOL)
    other = set(EVAL_POOL if split == "train" else TRAIN_SCENARIOS)
    seen = [int(row["scenario_id"]) for row in rows]
    from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST

    if set(seen) & set(RESERVED_TEST) or set(seen) & other or not set(seen) <= allowed:
        raise RuntimeError("train and validation scenarios are mixed")
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["scenario_id"]), []).append(row)
    checked = 0
    for scenario, items in grouped.items():
        items.sort(key=lambda item: int(item["time"]))
        times = [int(item["time"]) for item in items]
        if times != list(DECISION_TIMES):
            raise RuntimeError(f"windows are mixed for scenario {scenario}")
        total_bits = float(sum(float(item["bits15"]) for item in items))
        if abs(float(items[0]["payload_before"])) > 1.0e-6 or abs(float(items[-1]["payload_after"]) - total_bits) > 1.0e-4:
            raise RuntimeError("ACCOUNTING")
        for index, row in enumerate(items):
            expected_steps = 3 if int(row["time"]) == 45 else int(HOLD)
            if int(row["window_steps"]) != expected_steps:
                raise RuntimeError("window length does not match the decision grid")
            if index + 1 < len(items) and int(row["time"]) + int(row["window_steps"]) != int(items[index + 1]["time"]):
                raise RuntimeError("adjacent windows overlap or leave a gap")
            if abs(float(row["payload_after"]) - float(row["payload_before"]) - float(row["bits15"])) > 1.0e-6:
                raise RuntimeError("ACCOUNTING")
            if _key(row["pool_keys"][int(row["score_index"])]) != _key(row["assignment"]):
                raise RuntimeError("score index does not point at the scheme C assignment")
            env = _open_env(row)
            option = make_option(env, row["assignment"])
            if not np.array_equal(option.association, option.targets) or not np.array_equal(option.association, row["assignment"]):
                raise RuntimeError("INTERFACE")
            if int(option.remaining_hold) != int(row["remaining_hold"]) or int(option.remaining_hold) != int(HOLD):
                raise RuntimeError("INTERFACE")
            obs = build_actor_observation(env, option, row["last_movement"])
            if any(key in obs for key in FORBIDDEN_KEYS) or bool(obs["leaks_future_arrivals"]):
                raise RuntimeError("future information reached the actor observation")
            if "bits15" in obs or "payload" in obs:
                raise RuntimeError("recorded return leaked into the actor observation")
            pool = { _key(item) for item in row["pool_keys"] }
            if _key(row["assignment"]) not in pool or len(pool) != int(row["pool_count"]):
                raise RuntimeError("stored pool does not contain the scheme C assignment")
            pred = None if int(row["predecessor"]) < 0 else int(row["predecessor"])
            selection = planner.select_assignment(env, row["position_prev"], pred)
            if selection.status != "selected" or _key(selection.assignment) != _key(row["assignment"]):
                raise RuntimeError("restored state does not reproduce the scheme C assignment")
            if int(selection.candidate_count) != int(row["pool_count"]) or str(selection.pool_hash) != str(row["pool_hash"]):
                raise RuntimeError("candidate pool hash changed")
            checked += 1
    return {
        "split": split,
        "rows": int(len(rows)),
        "scenarios": int(len(grouped)),
        "reselected": int(checked),
        "accounting_ok": True,
        "future_inputs": 0,
        "windows_aligned": True,
    }


def distribution_summary(rows: list[dict]) -> dict:
    def mean(key: str, source: str) -> float:
        if source == "scheme_c":
            return float(np.mean([row["features"][key] for row in rows]))
        return float(np.mean([row["k5_features"][key] for row in rows]))

    full = [row for row in rows if int(row["time"]) != 45]
    return {
        "decisions": int(len(rows)),
        "scheme_c_outside_k5_fraction": float(np.mean([1 - int(row["in_k5_pool"]) for row in rows])),
        "mean_pool_count": float(np.mean([row["pool_count"] for row in rows])),
        "scheme_c": {key: mean(key, "scheme_c") for key in ("n_active", "idle_fraction", "distance_m", "in_radius_fraction", "queue_bits", "sinr", "rate_bps", "interference")},
        "train_k5": {key: mean(key, "k5") for key in ("n_active", "idle_fraction", "distance_m", "in_radius_fraction", "queue_bits", "sinr", "rate_bps", "interference")},
        "scheme_c_bits15": float(np.mean([row["bits15"] for row in full])),
        "train_k5_bits15": float(np.mean([row["k5_bits15"] for row in full])),
    }


def repair_legacy_dataset(split: str, manifest: dict) -> bool:
    """Separate a chosen assignment that an earlier save stored over the snapshot."""
    path = DATASET_DIR / f"{split}.npz"
    packed = np.load(path, allow_pickle=False)
    if "chosen_assignment" in packed.files:
        packed.close()
        return False
    offsets = np.asarray(packed["pool_offsets"], dtype=np.int64)
    keys = np.asarray(packed["pool_keys"])
    dense = keys.ndim == 3
    names = ("n_active", "idle_fraction", "distance_m", "in_radius_fraction", "queue_bits", "sinr", "rate_bps", "interference")
    rows = []
    count = int(packed["scenario_id"].shape[0])
    for index in range(count):
        snap = {name: np.asarray(packed[name][index]).copy() for name in STATE_FIELDS}
        snap["t"] = int(packed["t"][index])
        snap["payload_bits"] = float(packed["payload_bits"][index])
        snap["dropped"] = float(packed["dropped"][index])
        if dense:
            pool = np.asarray(keys[index], dtype=np.int64)
        else:
            start = int(offsets[index])
            stop = int(offsets[index + 1]) if index + 1 < len(offsets) else int(keys.shape[0])
            pool = np.asarray(keys[start:stop], dtype=np.int64)
        chosen = np.asarray(pool[int(packed["score_index"][index])], dtype=np.int64).copy()
        rows.append(
            {
                "split": split,
                "scenario_id": int(packed["scenario_id"][index]),
                "scenario_seed": int(packed["scenario_seed"][index]),
                "time": int(packed["time"][index]),
                "window_steps": int(packed["window_steps"][index]),
                "assignment": chosen,
                "pool_keys": pool,
                "pool_hash": str(packed["pool_hash"][index]),
                "pool_count": int(packed["pool_count"][index]),
                "score_index": int(packed["score_index"][index]),
                "remaining_hold": int(packed["remaining_hold"][index]),
                "in_k5_pool": int(packed["in_k5_pool"][index]),
                "bits15": float(packed["bits15"][index]),
                "payload_before": float(packed["payload_before"][index]),
                "payload_after": float(packed["payload_after"][index]),
                "k5_bits15": float(packed["k5_bits15"][index]),
                "initial_q_bits": np.asarray(packed["initial_q_bits"][index], dtype=np.float64).copy(),
                "buoy_positions": np.asarray(packed["buoy_positions"][index], dtype=np.float64).copy(),
                "last_movement": np.asarray(packed["last_movement"][index], dtype=np.float64).copy(),
                "position_prev": np.asarray(packed["position_prev"][index], dtype=np.float64).copy(),
                "predecessor": int(packed["predecessor"][index]),
                "snap": snap,
                "features": {key: float(packed[f"feature_{key}"][index]) for key in names},
                "k5_features": {key: float(packed[f"k5_{key}"][index]) for key in names},
            }
        )
    packed.close()
    save_rows(rows, split, manifest)
    return True


__all__ = ["collect_split", "distribution_summary", "load_rows", "repair_legacy_dataset", "save_rows", "validate_rows"]
