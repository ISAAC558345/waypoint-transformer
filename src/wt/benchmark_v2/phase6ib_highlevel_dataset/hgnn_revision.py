"""HGNN-input revision of the frozen Phase-6I-B dataset.

The frozen shards stay byte-for-byte in place. This module replays the same
MAX_RATE hold and frozen MAPPO-V2 seed-0 execution, then writes a separate
revision that adds the two raw fields the encoder cannot recover:

- service_progress: L3BEnv.progress, exposed as info['service_progress']
- uav_step_displacement_m: position_t - position of the previous environment step

Candidate labels are copied from the frozen shards. J15 is not recomputed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import geometry
from wt.benchmark_v2.phase6c.rollout import choose_assignment
from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads
from wt.benchmark_v2.phase6h_executor_conditioned_macro.jobs import _live_step
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _frozen_actor, _key, _split_of
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import HOLD, OPEN_POOL, PHASE, PRIMARY_TIMES, assert_allowed
from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_arrays, state_hash
from wt.l3b.env import kappa_for, reference_rate


REVISION_NAME = "Phase-6I-B-HGNN-input-compatible"
FROZEN_DATASET = PHASE / "dataset"
REVISION_ROOT = PHASE / "revision_hgnn_input"
REVISION_DATASET = REVISION_ROOT / "dataset"
CHECKPOINT = PHASE.parents[3] / "phase5l_joint_encoder" / "joint_utility_encoder.pt"
DIAGNOSTIC_TIME = 45
CANDIDATE_KEYS = ("candidate_id", "assignment", "j15_bits", "utility_rank", "tie_group_id")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_paths(scenario: int, time_step: int) -> dict[str, Path]:
    folder = FROZEN_DATASET / _split_of(scenario) / f"scenario_{int(scenario)}"
    stem = f"t{int(time_step):03d}"
    return {"state": folder / f"{stem}_state.npz", "candidates": folder / f"{stem}_candidates.npz", "manifest": folder / f"{stem}_manifest.json"}


def _revision_paths(scenario: int, time_step: int) -> dict[str, Path]:
    folder = REVISION_DATASET / _split_of(scenario) / f"scenario_{int(scenario)}"
    stem = f"t{int(time_step):03d}"
    return {"state": folder / f"{stem}_state.npz", "candidates": folder / f"{stem}_candidates.npz", "manifest": folder / f"{stem}_manifest.json"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as shard:
        return {key: np.asarray(shard[key]) for key in shard.files}


def step_observation(env: L3BBitQueueEnvV2, position_prev: np.ndarray, predecessor_time: int | None) -> dict[str, np.ndarray | int | str | None]:
    """Read the live environment. Displacement is one transition, never a 15-step chord."""
    position = np.asarray(env.uav_positions, dtype=np.float64).copy()
    previous = np.asarray(position_prev, dtype=np.float64)
    if previous.shape != position.shape:
        raise RuntimeError("previous UAV position shape does not match the live state")
    progress = np.asarray(env.progress, dtype=np.float64).copy()
    # install_light_info replaces env.info during rollout. The class method still
    # exposes service_progress from L3BEnv.progress and does not alter that state.
    info_progress = np.asarray(L3BBitQueueEnvV2.info(env)["service_progress"], dtype=np.float64)
    if not np.array_equal(progress, info_progress):
        raise RuntimeError("env.progress and info service_progress diverged")
    displacement = position - previous
    limit = float(env.config.step_length_m) + 1.0e-8
    if predecessor_time is not None and float(np.max(np.linalg.norm(displacement, axis=1))) > limit:
        raise RuntimeError("recorded displacement is longer than one environment step")
    return {
        "service_progress": progress,
        "uav_step_displacement_m": displacement,
        "predecessor_time_step": None if predecessor_time is None else int(predecessor_time),
        "velocity_predecessor": "reset_pose_before_any_step" if predecessor_time is None else "previous_environment_step",
    }


def _arrays_match(live: dict[str, np.ndarray], stored: dict[str, np.ndarray]) -> bool:
    if set(live) != set(stored):
        return False
    return all(np.array_equal(np.asarray(live[key]), np.asarray(stored[key])) for key in live)


def _open_env(scenario: int) -> L3BBitQueueEnvV2:
    limit_threads()
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario))
    env._phase6h_last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    return env


def _choose(env: L3BBitQueueEnvV2):
    geo = geometry(env)
    max_rate, _meta = choose_assignment("max_rate_independent", env, np.random.default_rng(0), geo)
    return make_option(env, np.asarray(_key(max_rate), dtype=np.int64))


def replay_scenario(scenario: int, *, trace_until: int | None = None) -> dict:
    """Replay one frozen scenario and write the HGNN-input revision."""
    scenario = int(scenario)
    assert_allowed(scenario, split=_split_of(scenario))
    actor = _frozen_actor()
    env = _open_env(scenario)
    position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
    predecessor_time = None
    option = None
    pending = 0
    rows = []
    trace = []
    diagnostic_time = None if trace_until is None else int(trace_until)
    source_hashes = {}
    while int(env.t) < int(env.config.episode_horizon):
        current = int(env.t)
        observed = step_observation(env, position_prev, predecessor_time)
        if diagnostic_time is None or current <= diagnostic_time:
            displacement = np.asarray(observed["uav_step_displacement_m"], dtype=np.float64)
            trace.append(
                {
                    "time_step": current,
                    "predecessor_time_step": observed["predecessor_time_step"],
                    "velocity_predecessor": observed["velocity_predecessor"],
                    "max_displacement_m": float(np.max(np.linalg.norm(displacement, axis=1))),
                    "service_progress": [float(item) for item in np.asarray(observed["service_progress"])],
                    "saved_dataset_row": int(current in PRIMARY_TIMES),
                }
            )
        if pending <= 0:
            if current in PRIMARY_TIMES:
                rows.append(_store_revision(env, scenario, current, observed, source_hashes))
            if current >= int(PRIMARY_TIMES[-1]) and (diagnostic_time is None or current >= diagnostic_time):
                break
            if current < int(env.config.episode_horizon):
                option = _choose(env)
                pending = HOLD
        if int(env.t) >= int(env.config.episode_horizon):
            break
        if pending <= 0:
            raise RuntimeError("revision replay lost its option")
        position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
        predecessor_time = int(env.t)
        _live_step(env, actor, option, use_fly=False)
        pending -= 1
    if [int(row["time_step"]) for row in rows] != list(PRIMARY_TIMES):
        raise RuntimeError("revision did not write every primary state")
    for time_step, before in source_hashes.items():
        frozen = _frozen_paths(scenario, int(time_step))
        if _sha(frozen["state"]) != before["state"] or _sha(frozen["candidates"]) != before["candidates"]:
            raise RuntimeError("frozen Phase-6I-B shard changed while writing the revision")
    return {"scenario_id": scenario, "states": rows, "trace": trace}


def _store_revision(env: L3BBitQueueEnvV2, scenario: int, time_step: int, observed: dict, source_hashes: dict) -> dict:
    frozen = _frozen_paths(scenario, time_step)
    source_hashes[int(time_step)] = {"state": _sha(frozen["state"]), "candidates": _sha(frozen["candidates"])}
    stored_state = _load_npz(frozen["state"])
    live_state = state_arrays(env)
    if not _arrays_match(live_state, stored_state):
        raise RuntimeError(f"replay state diverged from the frozen shard at {scenario} t={time_step}")
    frozen_manifest = json.loads(frozen["manifest"].read_text(encoding="utf-8"))
    if state_hash(scenario, live_state) != str(frozen_manifest["state_hash"]):
        raise RuntimeError(f"frozen state hash mismatch at {scenario} t={time_step}")
    stored_candidates = _load_npz(frozen["candidates"])
    if tuple(stored_candidates) != CANDIDATE_KEYS and set(stored_candidates) != set(CANDIDATE_KEYS):
        raise RuntimeError("frozen candidate schema changed")
    assignment = np.asarray(stored_candidates["assignment"], dtype=np.int64)
    illegal = int(sum(not is_legal(row, env.config.num_uavs, env.config.num_buoys) for row in assignment))
    if illegal != 0 or int(frozen_manifest["illegal"]) != 0:
        raise RuntimeError(f"assignment legality changed at {scenario} t={time_step}")
    if int(assignment.shape[0]) != int(frozen_manifest["candidate_count"]):
        raise RuntimeError(f"candidate count changed at {scenario} t={time_step}")
    progress = np.asarray(observed["service_progress"], dtype=np.float64)
    displacement = np.asarray(observed["uav_step_displacement_m"], dtype=np.float64)
    if progress.shape != (env.config.num_uavs,) or displacement.shape != (env.config.num_uavs, 2):
        raise RuntimeError("new HGNN raw fields have the wrong shape")
    if not np.all(np.isfinite(progress)) or not np.all(np.isfinite(displacement)):
        raise RuntimeError("new HGNN raw fields are non-finite")
    augmented = dict(live_state)
    augmented["service_progress"] = progress
    augmented["uav_step_displacement_m"] = displacement
    paths = _revision_paths(scenario, time_step)
    paths["state"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["state"].with_suffix(paths["state"].suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **augmented)
    temporary.replace(paths["state"])
    shutil.copy2(frozen["candidates"], paths["candidates"])
    copied = _load_npz(paths["candidates"])
    label_gap = float(np.max(np.abs(np.asarray(copied["j15_bits"], dtype=np.float64) - np.asarray(stored_candidates["j15_bits"], dtype=np.float64))))
    if label_gap != 0.0 or not np.array_equal(copied["assignment"], stored_candidates["assignment"]):
        raise RuntimeError("copied J15 labels or assignments differ from the frozen shard")
    payload = {
        "revision": REVISION_NAME,
        "scenario_id": int(scenario),
        "time_step": int(time_step),
        "state_source": "MAX_RATE_INDEPENDENT held for 15 steps and executed by frozen MAPPO-V2 seed0",
        "service_progress_source": "L3BEnv.progress",
        "velocity_definition": "uav_position_m(t) - uav_position_m(previous environment step)",
        "velocity_unit": "meter per environment step",
        "velocity_predecessor": observed["velocity_predecessor"],
        "predecessor_time_step": observed["predecessor_time_step"],
        "candidate_count": int(frozen_manifest["candidate_count"]),
        "illegal": 0,
        "j15_max_abs_error_vs_frozen": 0.0,
        "source_state_hash": str(frozen_manifest["state_hash"]),
        "source_checksum_state": source_hashes[int(time_step)]["state"],
        "source_checksum_candidates": source_hashes[int(time_step)]["candidates"],
        "checksum_state": _sha(paths["state"]),
        "checksum_candidates": _sha(paths["candidates"]),
        "argmax_assignment": frozen_manifest["argmax_assignment"],
    }
    if payload["checksum_candidates"] != payload["source_checksum_candidates"]:
        raise RuntimeError("candidate copy is not byte-identical to the frozen shard")
    _write_json(paths["manifest"], payload)
    return payload


def checkpoint_input_dims() -> dict[str, list[int]]:
    import torch

    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    weights = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    return {
        "uav": [int(item) for item in weights["encoder.uav_enc.0.weight"].shape],
        "buoy": [int(item) for item in weights["encoder.buoy_enc.0.weight"].shape],
        "edge": [int(item) for item in weights["encoder.edge_enc.0.weight"].shape],
        "edge_uu": [int(item) for item in weights["encoder.uu_enc.0.weight"].shape],
    }


def hgnn_tensors(arrays: dict[str, np.ndarray], config) -> dict[str, np.ndarray]:
    """Build JointUtilityHGNN inputs from one revised state. No imputed channels."""
    position = np.asarray(arrays["uav_position_m"], dtype=np.float64)
    buoy_position = np.asarray(arrays["buoy_position_m"], dtype=np.float64)
    association = np.asarray(arrays["association"], dtype=np.int64)
    q_bits = np.asarray(arrays["q_bits"], dtype=np.float64)
    progress = np.asarray(arrays["service_progress"], dtype=np.float64)
    displacement = np.asarray(arrays["uav_step_displacement_m"], dtype=np.float64)
    distance_ub = np.asarray(arrays["uav_buoy_distance_m"], dtype=np.float64)
    distance_uu = np.asarray(arrays["uav_uav_distance_m"], dtype=np.float64)
    gain = np.asarray(arrays["uav_buoy_gain"], dtype=np.float64)
    rate = np.asarray(arrays["uav_buoy_hypothetical_rate"], dtype=np.float64)
    in_range = np.asarray(arrays["uav_buoy_in_range"], dtype=np.uint8)
    time_step = int(np.asarray(arrays["time_index"]).reshape(-1)[0])
    block = float(config.block_payload_bits)
    capacity = float(config.queue_capacity_blocks)
    queue = q_bits / block
    step = float(config.step_length_m)
    diag = float(np.hypot(config.area_width, config.area_height))
    reference = max(float(reference_rate(config)), 1.0e-9)
    velocity = np.clip(displacement / step, -1.0, 1.0)
    assigned = (association >= 0).astype(np.float64)
    assigned_queue = np.zeros(config.num_uavs, dtype=np.float64)
    for uav_id, buoy_id in enumerate(association):
        if int(buoy_id) >= 0:
            assigned_queue[uav_id] = queue[int(buoy_id)] / capacity
    uav = np.column_stack(
        (
            2.0 * position[:, 0] / float(config.area_width) - 1.0,
            2.0 * position[:, 1] / float(config.area_height) - 1.0,
            velocity[:, 0],
            velocity[:, 1],
            progress,
            assigned,
            assigned_queue,
            np.zeros(config.num_uavs, dtype=np.float64),
        )
    )
    occupied = np.zeros(config.num_buoys, dtype=np.float64)
    progress_on_buoy = np.zeros(config.num_buoys, dtype=np.float64)
    for uav_id, buoy_id in enumerate(association):
        if int(buoy_id) >= 0:
            occupied[int(buoy_id)] = 1.0
            progress_on_buoy[int(buoy_id)] = max(float(progress_on_buoy[int(buoy_id)]), float(progress[uav_id]))
    if time_step == 0:
        last_arrival = np.zeros(config.num_buoys, dtype=np.float64)
    else:
        clock = L3BBitQueueEnvV2()
        last_arrival = np.asarray(clock.arrivals_at(time_step - 1), dtype=np.float64)
    scale = float(np.max(queue)) + 1.0e-8
    urgency = np.clip(queue / scale, 0.0, 1.0)
    buoy = np.column_stack(
        (
            2.0 * buoy_position[:, 0] / float(config.area_width) - 1.0,
            2.0 * buoy_position[:, 1] / float(config.area_height) - 1.0,
            np.clip(queue / capacity, 0.0, 1.0),
            np.clip(last_arrival / capacity, 0.0, 1.0),
            progress_on_buoy,
            occupied,
            np.full(config.num_buoys, float(config.num_buoys) / 12.0, dtype=np.float64),
            urgency,
        )
    )
    edge = np.zeros((config.num_uavs, config.num_buoys, 4), dtype=np.float64)
    edge[:, :, 0] = distance_ub / diag
    edge[:, :, 1] = np.clip(rate / reference, 0.0, 2.0) / 2.0
    edge[:, :, 2] = (distance_ub / step) / float(config.episode_horizon)
    edge[:, :, 3] = np.clip(queue / capacity, 0.0, 1.0)[None, :]
    counts: dict[int, int] = {}
    chosen = []
    for uav_id, buoy_id in enumerate(association):
        target = int(buoy_id)
        if target < 0 or (not bool(in_range[uav_id, target])) or float(queue[target]) <= 1.0e-9:
            continue
        counts[target] = counts.get(target, 0) + 1
        chosen.append((uav_id, target))
    active = [(uav, target) for uav, target in chosen if counts[target] == 1]
    interference = np.zeros((config.num_uavs, config.num_uavs), dtype=np.float64)
    for left, buoy_left in active:
        for right, buoy_right in active:
            if int(left) == int(right):
                continue
            interference[int(left), int(right)] = float(gain[int(right), int(buoy_left)]) * float(config.tx_power) * kappa_for(config, int(buoy_right), int(buoy_left))
    edge_uu = np.zeros((config.num_uavs, config.num_uavs, 2), dtype=np.float64)
    edge_uu[:, :, 0] = distance_uu / diag
    edge_uu[:, :, 1] = np.clip(interference / float(config.tx_power), 0.0, 1.0)
    global_ = np.asarray(
        (
            float(time_step) / float(config.episode_horizon),
            float(np.sum(queue)) / (capacity * float(config.num_buoys)),
            float(config.num_uavs) / 6.0,
        ),
        dtype=np.float64,
    )
    return {"uav": uav, "buoy": buoy, "edge": edge, "edge_uu": edge_uu, "global": global_}


def _stats(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array, dtype=np.float64)
    return {"min": float(np.min(values)), "max": float(np.max(values)), "mean": float(np.mean(values)), "finite": int(np.all(np.isfinite(values)))}


def audit_scenario(scenario: int) -> dict:
    config = L3BBitQueueEnvV2().config
    expected = checkpoint_input_dims()
    states = []
    for time_step in PRIMARY_TIMES:
        paths = _revision_paths(scenario, int(time_step))
        arrays = _load_npz(paths["state"])
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        tensors = hgnn_tensors(arrays, config)
        dims_ok = (
            int(tensors["uav"].shape[-1]) == int(expected["uav"][1])
            and int(tensors["buoy"].shape[-1]) == int(expected["buoy"][1])
            and int(tensors["edge"].shape[-1]) == int(expected["edge"][1])
            and int(tensors["edge_uu"].shape[-1]) == int(expected["edge_uu"][1])
        )
        finite = all(int(_stats(tensors[key])["finite"]) == 1 for key in tensors)
        finite = finite and int(_stats(arrays["service_progress"])["finite"]) == 1 and int(_stats(arrays["uav_step_displacement_m"])["finite"]) == 1
        states.append(
            {
                "time_step": int(time_step),
                "fields": sorted(arrays),
                "uav_feature_shape": list(tensors["uav"].shape),
                "buoy_feature_shape": list(tensors["buoy"].shape),
                "uav_buoy_edge_shape": list(tensors["edge"].shape),
                "uav_uav_edge_shape": list(tensors["edge_uu"].shape),
                "service_progress_shape": list(np.asarray(arrays["service_progress"]).shape),
                "uav_velocity_shape": list(np.asarray(arrays["uav_step_displacement_m"]).shape),
                "predecessor_time_step": manifest["predecessor_time_step"],
                "velocity_predecessor": manifest["velocity_predecessor"],
                "candidate_count": int(manifest["candidate_count"]),
                "j15_max_abs_error_vs_frozen": float(manifest["j15_max_abs_error_vs_frozen"]),
                "dimension_match": int(dims_ok),
                "finite": int(finite),
                "feature_stats": {key: _stats(tensors[key]) for key in ("uav", "buoy", "edge", "edge_uu", "global")},
                "service_progress_stats": _stats(arrays["service_progress"]),
                "uav_step_displacement_stats": _stats(arrays["uav_step_displacement_m"]),
            }
        )
    return {"scenario_id": int(scenario), "state_count": len(states), "checkpoint_input_dims": expected, "states": states}


def revision_complete(scenario: int) -> bool:
    for time_step in PRIMARY_TIMES:
        paths = _revision_paths(scenario, int(time_step))
        if not paths["manifest"].is_file():
            return False
        payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if payload.get("revision") != REVISION_NAME:
            return False
        if _sha(paths["state"]) != payload["checksum_state"] or _sha(paths["candidates"]) != payload["checksum_candidates"]:
            return False
        frozen = _frozen_paths(scenario, int(time_step))
        if _sha(frozen["state"]) != payload["source_checksum_state"] or _sha(frozen["candidates"]) != payload["source_checksum_candidates"]:
            return False
    return True


def scenarios_to_revise() -> list[int]:
    return [int(scenario) for scenario in OPEN_POOL if not revision_complete(scenario)]
