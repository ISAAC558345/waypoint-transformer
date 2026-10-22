"""Replay official Phase-9.4.5 hold states and label H=15 RL-mean payload."""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.htw_phase945.candidate_pool import assignment_hash, generate_pool
from wt.htw_phase945.protocol import OUTPUT as PHASE945_OUTPUT
from wt.htw_phase945.utility_rollout import shadow_h15 as l0_shadow_h15
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward, oracle_option_assignment
from wt.lowlevel_v2.integration_phase1.shadow import shadow_h15
from wt.lowlevel_v2.integration_phase1.worker import CompatWorker
from wt.lowlevel_v2.phase1b.controller import load_frozen_high, make_networks
from wt.lowlevel_v2.phase4b.snapshot import snapshot_env

from .protocol import EXPECTED_HIGH_SHA, HOLD, LOW_SEEDS, SHARD_DIR, ckpt_path, refuse_compat

_WORKERS: dict[int, CompatWorker] | None = None


def replay_to_hold(scenario: int, time_step: int) -> dict:
    refuse_compat(int(scenario))
    env = L3BEnv()
    info = env.reset(seed=int(scenario))
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    last_movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    while int(env.t) < int(time_step) and not info.get("terminated", False):
        chosen = oracle_option_assignment(env)
        pos0 = np.asarray(env.uav_positions, dtype=np.float64).copy()
        info = env.step(fly_toward(env, chosen), chosen)
        last_arrival = np.asarray(info["arrivals_blocks"], dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        last_movement = np.asarray(env.uav_positions, dtype=np.float64) - pos0
    if int(env.t) != int(time_step):
        raise RuntimeError(f"failed to replay scenario={scenario} to t={time_step}, got t={env.t}")
    return {
        "scenario": int(scenario),
        "t": int(env.t),
        "env": snapshot_env(env),
        "last_arrival": last_arrival,
        "last_positions": last_positions,
        "last_movement": last_movement,
    }


def load_ll_worker(model_seed: int) -> CompatWorker:
    high, digest = load_frozen_high()
    if digest != EXPECTED_HIGH_SHA:
        raise RuntimeError("high SHA mismatch while loading frozen low-level")
    actor, critic = make_networks()
    payload = torch.load(ckpt_path(int(model_seed)), map_location="cpu")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return CompatWorker(high, actor, critic, service_lock=True, trust_region=True)


def shard_path(split: str, index: int) -> Path:
    return SHARD_DIR / f"{split}_{int(index):04d}.json"


def generate_state_utilities(workers: dict[int, CompatWorker], split: str, index: int, meta: dict, candidates: np.ndarray, expected_hashes: list[str] | None = None) -> dict:
    path = shard_path(split, index)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if expected_hashes is None or payload.get("hashes") == list(expected_hashes):
            return payload
    state = replay_to_hold(int(meta["scenario"]), int(meta["t"]))
    per_seed = {int(seed): [] for seed in LOW_SEEDS}
    hashes = []
    for row in np.asarray(candidates, dtype=np.int64):
        digest = assignment_hash(row)
        hashes.append(digest)
        for seed, worker in workers.items():
            stats = shadow_h15(worker, state, row, zero_residual=False)
            per_seed[int(seed)].append(float(stats["payload_increment"]))
    if expected_hashes is not None and hashes != list(expected_hashes):
        raise RuntimeError(f"candidate hash mismatch while labeling {split} {index}")
    mean = np.mean([per_seed[int(seed)] for seed in LOW_SEEDS], axis=0).tolist()
    payload = {
        "split": split,
        "index": int(index),
        "scenario": int(meta["scenario"]),
        "t": int(meta["t"]),
        "hashes": hashes,
        "U_mean": mean,
        "U_per_seed": {str(seed): per_seed[int(seed)] for seed in LOW_SEEDS},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return payload


def _init_workers() -> None:
    global _WORKERS
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    _WORKERS = {int(seed): load_ll_worker(int(seed)) for seed in LOW_SEEDS}


def _label_job(item: tuple) -> dict:
    split, index, meta, candidates, expected_hashes = item
    assert _WORKERS is not None
    return generate_state_utilities(
        _WORKERS,
        split,
        int(index),
        meta,
        np.asarray(candidates, dtype=np.int64),
        expected_hashes=list(expected_hashes),
    )


def generate_split_utilities(split: str, data: dict, workers: int = 4) -> dict:
    n = int(data["assignment"].shape[0])
    jobs = []
    for index in range(n):
        jobs.append(
            (
                split,
                int(index),
                data["meta"][index],
                np.asarray(data["candidates"][index], dtype=np.int64).tolist(),
                list(data["hashes"][index]),
            )
        )
    pending = [job for job in jobs if not shard_path(split, int(job[1])).is_file()]
    print(f"{split}: {n - len(pending)}/{n} utility shards cached, {len(pending)} remaining", flush=True)
    if pending:
        n_workers = max(1, min(int(workers), len(pending)))
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_workers) as pool:
            futures = {pool.submit(_label_job, job): int(job[1]) for job in pending}
            done = 0
            for fut in as_completed(futures):
                done += 1
                index = futures[fut]
                fut.result()
                print(f"utility {split} {done}/{len(pending)} finished index={index}", flush=True)
    u_mean = np.zeros((n, int(data["candidates"].shape[1])), dtype=np.float64)
    u_seed = {int(seed): np.zeros_like(u_mean) for seed in LOW_SEEDS}
    shards = []
    for index in range(n):
        shard = json.loads(shard_path(split, index).read_text(encoding="utf-8"))
        if shard["hashes"] != list(data["hashes"][index]):
            raise RuntimeError(f"candidate hash mismatch {split} {index}")
        u_mean[index] = np.asarray(shard["U_mean"], dtype=np.float64)
        for seed in LOW_SEEDS:
            u_seed[int(seed)][index] = np.asarray(shard["U_per_seed"][str(int(seed))], dtype=np.float64)
        shards.append(shard)
    data["utility_old"] = np.asarray(data["utility"], dtype=np.float64).copy() if "utility" in data else np.full_like(u_mean, np.nan)
    data["utility"] = u_mean
    data["utility_per_seed"] = u_seed
    return data


def reconstruct_official_split(name: str) -> dict:
    npz_path = PHASE945_OUTPUT / f"{name}.npz"
    man = json.loads((PHASE945_OUTPUT / f"{name}_manifest.json").read_text(encoding="utf-8"))
    if npz_path.is_file():
        packed = np.load(npz_path)
        data = {key: packed[key] for key in packed.files}
        data["meta"] = man["meta"]
        data["hashes"] = man["hashes"]
        data["split"] = man["split"]
        data["seeds"] = man["seeds"]
        data["pool_size"] = man["pool_size"]
        if list(data["hashes"][0]) != list(man["hashes"][0]):
            raise RuntimeError(f"{name} npz hashes do not match official manifest")
        return data
    print(f"{name}.npz missing; reconstructing official candidate/feature tensors from frozen generator", flush=True)
    keys = ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu")
    feats = {key: [] for key in keys}
    refs = []
    pools = []
    uav_pos = []
    buoy_pos = []
    l0_util = []
    for index, meta in enumerate(man["meta"]):
        refuse_compat(int(meta["scenario"]))
        env = L3BEnv()
        info = env.reset(seed=int(meta["scenario"]))
        last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        while int(env.t) < int(meta["t"]) and not info.get("terminated", False):
            chosen = oracle_option_assignment(env)
            info = env.step(fly_toward(env, chosen), chosen)
            last_arrival = np.asarray(info["arrivals_blocks"], dtype=np.float32)
            last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        if int(env.t) != int(meta["t"]):
            raise RuntimeError(f"replay failed {meta}")
        batch = _batch_from_env(info, env, last_arrival, last_positions)
        ref = np.asarray(oracle_option_assignment(env), dtype=np.int64)
        pool = generate_pool(env, seed_key=int(meta["scenario"]) * 1009 + int(meta["t"]))
        hashes = [assignment_hash(row) for row in pool]
        if hashes != list(man["hashes"][index]):
            raise RuntimeError(f"reconstructed hashes mismatch official {name} index={index} scenario={meta['scenario']} t={meta['t']}")
        for key in keys:
            feats[key].append(batch[key][0])
        refs.append(np.pad(ref, (0, max(0, 6 - ref.shape[0])), constant_values=-1)[:6])
        pools.append(pool)
        uav_pos.append(np.asarray(env.uav_positions, dtype=np.float64).copy())
        buoy_pos.append(np.asarray(env.buoy_positions, dtype=np.float64).copy())
        if man["split"] != "train_collect":
            l0_util.append([float(l0_shadow_h15(env, row)["payload_increment"]) for row in pool])
        else:
            l0_util.append([float("nan")] * len(pool))
        print(f"reconstruct {name} {index + 1}/{len(man['meta'])} scenario={meta['scenario']} t={meta['t']}", flush=True)
    packed = {key: np.stack(value, axis=0) for key, value in feats.items()}
    packed["assignment"] = np.stack(refs, axis=0)
    packed["candidates"] = np.stack(pools, axis=0)
    packed["utility"] = np.asarray(l0_util, dtype=np.float64)
    packed["uav_positions"] = np.stack(uav_pos, axis=0)
    packed["buoy_positions"] = np.stack(buoy_pos, axis=0)
    packed["meta"] = man["meta"]
    packed["hashes"] = man["hashes"]
    packed["pool_size"] = man["pool_size"]
    packed["split"] = man["split"]
    packed["seeds"] = man["seeds"]
    packed["reconstructed_npz"] = True
    return packed


__all__ = [
    "generate_split_utilities",
    "generate_state_utilities",
    "load_ll_worker",
    "reconstruct_official_split",
    "replay_to_hold",
    "shard_path",
]
