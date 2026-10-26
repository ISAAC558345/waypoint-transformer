"""Check the frozen candidate table without recomputing J15."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import restore_state
from wt.benchmark_v2.phase6e.generators import build_pools, is_legal
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _key
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import GENERATOR, SELECTED_K

from .protocol import CANDIDATE_KEYS, assert_pilot_scenario
from .shards import ShardError, load_npz


def load_candidates(scenario_id: int, time_step: int, env_snap: dict, manifest: dict) -> dict:
    scenario = assert_pilot_scenario(scenario_id)
    arrays, digest = load_npz(scenario, int(time_step), "candidates")
    if tuple(arrays) != CANDIDATE_KEYS and set(arrays) != set(CANDIDATE_KEYS):
        raise ShardError("frozen candidate schema does not match the pilot contract")
    if digest != str(manifest["checksum_candidates"]):
        raise ShardError(f"candidate checksum mismatch at {scenario} t={int(time_step)}")
    assignment = np.asarray(arrays["assignment"], dtype=np.int64)
    candidate_id = np.asarray(arrays["candidate_id"], dtype=np.int64).reshape(-1)
    j15_bits = np.asarray(arrays["j15_bits"], dtype=np.float64).reshape(-1)
    utility_rank = np.asarray(arrays["utility_rank"], dtype=np.int64).reshape(-1)
    tie_group_id = np.asarray(arrays["tie_group_id"], dtype=np.int64).reshape(-1)
    if assignment.ndim != 2 or assignment.shape[0] != candidate_id.size:
        raise ShardError("candidate assignment shape does not match candidate ids")
    if int(candidate_id.size) != int(manifest["candidate_count"]) or int(manifest["illegal"]) != 0:
        raise ShardError("candidate count or illegal flag does not match the manifest")
    env = _env_at(scenario, env_snap)
    illegal = int(sum(not is_legal(row, env.config.num_uavs, env.config.num_buoys) for row in assignment))
    if illegal != 0:
        raise ShardError(f"{illegal} frozen candidates are illegal at {scenario} t={int(time_step)}")
    pools, _packed, pool_illegal = build_pools(env, ((GENERATOR, int(SELECTED_K)),))
    if int(pool_illegal) != 0:
        raise ShardError("candidate pool builder reported an illegal assignment")
    keys = sorted({_key(item) for item in pools[(GENERATOR, int(SELECTED_K))]})
    shard_keys = [tuple(int(item) for item in row) for row in assignment]
    order_match = keys == shard_keys
    if not order_match:
        raise ShardError(f"candidate order does not match the sorted G1_RATE k=6 pool at {scenario} t={int(time_step)}")
    order_hash = _order_hash(assignment)
    return {
        "scenario_id": scenario,
        "time_step": int(time_step),
        "candidate_id": candidate_id,
        "assignment": assignment,
        "j15_bits": j15_bits,
        "utility_rank": utility_rank,
        "tie_group_id": tie_group_id,
        "candidate_count": int(candidate_id.size),
        "illegal_count": illegal,
        "pool_order_match": bool(order_match),
        "pool_hash": order_hash,
        "checksum_candidates": digest,
    }


def _env_at(scenario: int, env_snap: dict) -> L3BBitQueueEnvV2:
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario))
    restore_state(env, env_snap)
    return env


def _order_hash(assignment: np.ndarray) -> str:
    import hashlib

    raw = np.ascontiguousarray(assignment, dtype=np.int64).tobytes()
    return hashlib.sha256(raw).hexdigest()
