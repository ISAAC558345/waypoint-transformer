"""State shards. Each worker owns its environment and MAPPO copy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import geometry, link_rates
from wt.benchmark_v2.phase6c.rollout import choose_assignment
from wt.benchmark_v2.phase6e.generators import build_pools, feature_matrices, is_legal, proposal_scores
from wt.benchmark_v2.phase6e.value import argmax_value
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads, load_primary
from wt.benchmark_v2.phase6h_executor_conditioned_macro.jobs import _guard, _live_step
from wt.benchmark_v2.phase6h_executor_conditioned_macro.labels import batch_j15_mappo
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import (
    EPS_BITS,
    GENERATOR,
    HOLD,
    LABEL_NAME,
    PHASE,
    PRIMARY_TIMES,
    SELECTED_K,
    TRAIN_POOL,
    VALIDATION_POOL,
    assert_allowed,
    integrity_selected,
)
from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_arrays, state_hash


_CACHED_ACTOR = None


def _frozen_actor():
    """One frozen seed-0 actor per worker process. The weights stay read-only."""
    global _CACHED_ACTOR
    if _CACHED_ACTOR is None:
        limit_threads()
        _CACHED_ACTOR = load_primary()
    return _CACHED_ACTOR


def _key(assignment) -> tuple[int, ...]:
    return tuple(int(item) for item in np.asarray(assignment, dtype=np.int64).reshape(-1))


def _split_of(scenario: int) -> str:
    if int(scenario) in TRAIN_POOL:
        return "train"
    if int(scenario) in VALIDATION_POOL:
        return "validation"
    raise RuntimeError(f"phase-6ib refuses scenario {scenario}")


def _directory(scenario: int) -> Path:
    return PHASE / "dataset" / _split_of(scenario) / f"scenario_{int(scenario)}"


def _paths(scenario: int, time_step: int) -> dict[str, Path]:
    folder = _directory(scenario)
    stem = f"t{int(time_step):03d}"
    return {"state": folder / f"{stem}_state.npz", "candidates": folder / f"{stem}_candidates.npz", "manifest": folder / f"{stem}_manifest.json"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return _sha(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _dense_groups(keys: list[tuple[int, ...]], values: dict[tuple[int, ...], float]) -> tuple[np.ndarray, np.ndarray, int]:
    ordered = sorted(keys, key=lambda key: (-float(values[key]), key))
    ranks = np.zeros(len(keys), dtype=np.int32)
    groups = np.zeros(len(keys), dtype=np.int32)
    index = {key: position for position, key in enumerate(keys)}
    group_value = None
    group_id = -1
    dense = 0
    for key in ordered:
        value = float(values[key])
        if group_value is None or value < float(group_value) - EPS_BITS:
            dense += 1
            group_id += 1
            group_value = value
        ranks[index[key]] = dense
        groups[index[key]] = group_id
    return ranks, groups, int(group_id + 1)


def _entropy(groups: np.ndarray) -> float:
    if groups.size == 0:
        return 0.0
    counts = np.bincount(groups.astype(np.int64))
    probability = counts[counts > 0] / float(groups.size)
    return float(-np.sum(probability * np.log(probability)))


def valid_manifest(scenario: int, time_step: int) -> dict | None:
    paths = _paths(scenario, time_step)
    if not paths["manifest"].is_file() or not paths["state"].is_file() or not paths["candidates"].is_file():
        return None
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if payload.get("checksum_state") != _sha(paths["state"]) or payload.get("checksum_candidates") != _sha(paths["candidates"]):
        return None
    if int(payload.get("candidate_count", 0)) <= 0 or int(payload.get("illegal", 1)) != 0:
        return None
    return payload


def _score_current(env, actor, scenario: int, time_step: int) -> dict:
    pools, _packed, illegal = build_pools(env, ((GENERATOR, SELECTED_K),))
    if int(illegal) != 0:
        raise RuntimeError("ILLEGAL")
    raw = pools[(GENERATOR, SELECTED_K)]
    keys = sorted({_key(item) for item in raw})
    n_uavs = int(env.config.num_uavs)
    n_buoys = int(env.config.num_buoys)
    if any(not is_legal(np.asarray(key, dtype=np.int64), n_uavs, n_buoys) for key in keys):
        raise RuntimeError("ILLEGAL")
    horizon = min(HOLD, int(env.config.episode_horizon) - int(time_step))
    if int(horizon) != HOLD:
        raise RuntimeError("TRUNCATED")
    arrays = state_arrays(env)
    digest = state_hash(scenario, arrays)

    def _evaluate():
        values, seconds, _first, error = batch_j15_mappo(env, actor, [np.asarray(key, dtype=np.int64) for key in keys], horizon)
        if float(error) > float(env.tol_bits):
            raise RuntimeError("ACCOUNTING")
        if not np.all(np.isfinite(np.fromiter((values[key] for key in keys), dtype=np.float64, count=len(keys)))):
            raise RuntimeError("NONFINITE")
        return values, float(seconds), float(error)

    values, seconds, error = _guard(env, actor, None, _evaluate)
    ranks, groups, unique_groups = _dense_groups(keys, values)
    labels = np.asarray([float(values[key]) for key in keys], dtype=np.float64)
    best, best_value = argmax_value([np.asarray(key, dtype=np.int64) for key in keys], values)
    order = np.argsort(-labels, kind="mergesort")
    top2 = float(labels[order[1]]) if labels.size > 1 else float(labels[order[0]])
    top5 = float(labels[order[min(4, labels.size - 1)]])
    rates = proposal_scores(feature_matrices(env))[GENERATOR]
    additive = []
    for key in keys:
        total = 0.0
        for uav, buoy in enumerate(key):
            if int(buoy) >= 0:
                total += float(rates[uav, int(buoy)])
        additive.append(total)
    additive_key = keys[int(np.argmax(np.asarray(additive, dtype=np.float64)))]
    geo = geometry(env)
    max_rate, _meta = choose_assignment("max_rate_independent", env, np.random.default_rng(0), geo)
    max_rate_key = _key(max_rate)
    _sinr, _rate, mean_interference = link_rates(
        geo["gain"],
        np.asarray(max_rate_key, dtype=np.int64),
        env.queue,
        geo["in_range"],
        bandwidth_hz=float(env.config.bandwidth_hz),
        tx_power=float(env.config.tx_power),
        noise_power=float(env.config.noise_power),
        kappa_same=float(env.config.kappa_same_cluster),
        kappa_cross=float(env.config.kappa_cross_cluster),
    )
    edge_rank = 0
    rank_order = np.argsort(-rates, axis=1, kind="mergesort")
    for uav, buoy in enumerate(best):
        if int(buoy) < 0:
            continue
        edge_rank = max(edge_rank, int(np.where(rank_order[uav] == int(buoy))[0][0]) + 1)
    assignments = np.asarray(keys, dtype=np.int16)
    return {
        "arrays": arrays,
        "assignments": assignments,
        "labels": labels,
        "ranks": ranks,
        "groups": groups,
        "keys": keys,
        "values": values,
        "summary": {
            "scenario_id": int(scenario),
            "time_step": int(time_step),
            "split": _split_of(scenario),
            "state_id": f"{int(scenario)}_t{int(time_step):03d}",
            "state_hash": digest,
            "candidate_count": int(len(keys)),
            "illegal": 0,
            "conservation_error": float(error),
            "shadow_state_contamination": 0,
            "future_realized_arrivals_used": 0,
            "label": LABEL_NAME,
            "reuse_existing_label": 0,
            "seconds": float(seconds),
            "j15_mean": float(np.mean(labels)),
            "j15_std": float(np.std(labels)),
            "j15_min": float(np.min(labels)),
            "j15_max": float(np.max(labels)),
            "unique_j15_groups": int(unique_groups),
            "tie_fraction": float(np.mean(np.bincount(groups)[groups] > 1)),
            "all_candidates_tied": int(unique_groups == 1),
            "top1_j15": float(best_value),
            "top2_j15": top2,
            "top1_top2_margin": float(best_value - top2),
            "top1_top5_margin": float(best_value - top5),
            "rank_entropy": _entropy(groups),
            "argmax_assignment": ",".join(str(item) for item in best),
            "additive_assignment": ",".join(str(item) for item in additive_key),
            "additive_j15": float(values[additive_key]),
            "assignment_disagreement": int(additive_key != best),
            "j15_regret_bits": float(best_value - float(values[additive_key])),
            "max_rate_assignment": ",".join(str(item) for item in max_rate_key),
            "max_rate_in_pool": int(max_rate_key in values),
            "max_rate_j15": float(values[max_rate_key]) if max_rate_key in values else "",
            "max_edge_rank": int(edge_rank),
            "mean_interference": float(mean_interference),
            "q_bits_mean": float(np.mean(arrays["q_bits"])),
            "distance_mean": float(np.mean(arrays["uav_buoy_distance_m"])),
            "rate_mean": float(np.mean(arrays["uav_buoy_hypothetical_rate"])),
        },
    }


def _store(scenario: int, time_step: int, scored: dict) -> dict:
    paths = _paths(scenario, time_step)
    state_sha = _write_npz(paths["state"], scored["arrays"])
    candidate_sha = _write_npz(
        paths["candidates"],
        {
            "candidate_id": np.arange(scored["assignments"].shape[0], dtype=np.int32),
            "assignment": scored["assignments"],
            "j15_bits": scored["labels"],
            "utility_rank": scored["ranks"],
            "tie_group_id": scored["groups"],
        },
    )
    block = float(L3BBitQueueEnvV2().config.block_payload_bits)
    payload = dict(scored["summary"])
    payload["j15_block_equivalent_mean"] = float(np.mean(scored["labels"]) / block)
    payload["checksum_state"] = state_sha
    payload["checksum_candidates"] = candidate_sha
    _write_json(paths["manifest"], payload)
    return payload


def generate_scenario(scenario: int) -> dict:
    limit_threads()
    scenario = int(scenario)
    split = _split_of(scenario)
    assert_allowed(scenario, split=split)
    limit_threads()
    actor = _frozen_actor()
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=scenario)
    env._phase6h_last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    option = None
    pending = 0
    rows = []
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            if int(env.t) in PRIMARY_TIMES:
                cached = valid_manifest(scenario, int(env.t))
                if cached is None:
                    print(f"label {scenario} t={int(env.t)}", flush=True)
                    cached = _store(scenario, int(env.t), _score_current(env, actor, scenario, int(env.t)))
                rows.append(cached)
            if int(env.t) >= PRIMARY_TIMES[-1]:
                break
            geo = geometry(env)
            max_rate, _meta = choose_assignment("max_rate_independent", env, np.random.default_rng(0), geo)
            option = make_option(env, np.asarray(_key(max_rate), dtype=np.int64))
            pending = HOLD
        _live_step(env, actor, option, use_fly=False)
        pending -= 1
    print(f"scenario {scenario}", flush=True)
    return {"scenario_id": scenario, "split": split, "states": rows}


def equivalence_scenario(scenario: int) -> dict:
    """Recompute one scenario in this process and compare it with the serial shard."""
    limit_threads()
    scenario = int(scenario)
    assert_allowed(scenario, split="pilot")
    limit_threads()
    actor = _frozen_actor()
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=scenario)
    env._phase6h_last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    option = None
    pending = 0
    rows = []
    worst = 0.0
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            if int(env.t) in PRIMARY_TIMES:
                scored = _score_current(env, actor, scenario, int(env.t))
                with np.load(_paths(scenario, int(env.t))["candidates"]) as shard:
                    stored = np.asarray(shard["j15_bits"], dtype=np.float64)
                gap = float(np.max(np.abs(stored - scored["labels"]))) if stored.shape == scored["labels"].shape else float("inf")
                worst = max(worst, gap)
                manifest = valid_manifest(scenario, int(env.t))
                rows.append(
                    {
                        "time_step": int(env.t),
                        "max_abs_error": gap,
                        "argmax_match": int(manifest["argmax_assignment"] == scored["summary"]["argmax_assignment"]),
                        "count_match": int(int(manifest["candidate_count"]) == int(scored["summary"]["candidate_count"])),
                        "state_hash_match": int(manifest["state_hash"] == scored["summary"]["state_hash"]),
                    }
                )
            if int(env.t) >= PRIMARY_TIMES[-1]:
                break
            geo = geometry(env)
            max_rate, _meta = choose_assignment("max_rate_independent", env, np.random.default_rng(0), geo)
            option = make_option(env, np.asarray(_key(max_rate), dtype=np.int64))
            pending = HOLD
        _live_step(env, actor, option, use_fly=False)
        pending -= 1
    passed = int(worst <= EPS_BITS and all(row["argmax_match"] and row["count_match"] and row["state_hash_match"] for row in rows))
    return {"scenario_id": scenario, "states": rows, "max_abs_error": float(worst), "passed": passed}


def integrity_scenario(scenario: int) -> dict:
    limit_threads()
    scenario = int(scenario)
    assert_allowed(scenario, split=_split_of(scenario))
    limit_threads()
    actor = _frozen_actor()
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=scenario)
    env._phase6h_last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    option = None
    pending = 0
    checked = 0
    worst = 0.0
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            if int(env.t) in PRIMARY_TIMES:
                paths = _paths(scenario, int(env.t))
                with np.load(paths["candidates"]) as shard:
                    assignment = np.asarray(shard["assignment"], dtype=np.int64)
                    stored = np.asarray(shard["j15_bits"], dtype=np.float64)
                    candidate_id = np.asarray(shard["candidate_id"], dtype=np.int64)
                chosen = [index for index, item in enumerate(candidate_id) if integrity_selected(scenario, int(env.t), int(item))]
                if chosen:
                    values, _seconds, _first, error = _guard(
                        env,
                        actor,
                        None,
                        lambda chosen=chosen: batch_j15_mappo(env, actor, [assignment[index] for index in chosen], HOLD),
                    )
                    if float(error) > float(env.tol_bits):
                        raise RuntimeError("ACCOUNTING")
                    for index in chosen:
                        key = _key(assignment[index])
                        worst = max(worst, abs(float(values[key]) - float(stored[index])))
                        checked += 1
            if int(env.t) >= PRIMARY_TIMES[-1]:
                break
            geo = geometry(env)
            max_rate, _meta = choose_assignment("max_rate_independent", env, np.random.default_rng(0), geo)
            option = make_option(env, np.asarray(_key(max_rate), dtype=np.int64))
            pending = HOLD
        _live_step(env, actor, option, use_fly=False)
        pending -= 1
    return {"scenario_id": scenario, "checked": checked, "max_abs_error": float(worst), "passed": int(worst <= EPS_BITS)}


def determinism_scenario(scenario: int) -> dict:
    limit_threads()
    scenario = int(scenario)
    assert_allowed(scenario, split=_split_of(scenario))
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=scenario)
    env._phase6h_last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    option = None
    pending = 0
    limit_threads()
    actor = _frozen_actor()
    mismatches = 0
    states = 0
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            if int(env.t) in PRIMARY_TIMES:
                pools, _packed, illegal = build_pools(env, ((GENERATOR, SELECTED_K),))
                keys = sorted({_key(item) for item in pools[(GENERATOR, SELECTED_K)]})
                paths = _paths(scenario, int(env.t))
                with np.load(paths["candidates"]) as shard:
                    stored = np.asarray(shard["assignment"], dtype=np.int64)
                rebuilt = np.asarray(keys, dtype=np.int64)
                mismatches += int(rebuilt.shape != stored.shape or not np.array_equal(rebuilt, stored) or int(illegal) != 0)
                states += 1
            if int(env.t) >= PRIMARY_TIMES[-1]:
                break
            geo = geometry(env)
            max_rate, _meta = choose_assignment("max_rate_independent", env, np.random.default_rng(0), geo)
            option = make_option(env, np.asarray(_key(max_rate), dtype=np.int64))
            pending = HOLD
        _live_step(env, actor, option, use_fly=False)
        pending -= 1
    return {"scenario_id": scenario, "states": states, "mismatches": mismatches, "passed": int(mismatches == 0)}
