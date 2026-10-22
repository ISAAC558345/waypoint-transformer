"""Compare the formal k=5 pool with the frozen k=6 pool. No environment step."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6e.generators import build_pools, feature_matrices, is_legal, proposal_scores
from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import FROZEN_DATASET, REVISION_DATASET
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import PRIMARY_TIMES
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import assert_open
from wt.benchmark_v2.phase6is_assignment_head.audit import _rss_bytes
from wt.benchmark_v2.phase6iu_pairwise_head.audit import verify_split
from wt.benchmark_v2.phase6ix_candidate_alignment.protocol import (
    EPOCH6_CKPT,
    FORMAL_K,
    FROZEN_K,
    GENERATOR,
    OUTPUT,
    PROBE,
    RATE_ATOL,
    SCHEME_C_CKPT,
    VALIDATION_POOL,
    assert_output_path,
    assign_gate,
)
from wt.benchmark_v2.phase6iw_inference_adapter.protocol import MAPPO_CKPT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _folder(root: Path, scenario: int) -> Path:
    split = "train" if int(scenario) < 322048 else "validation"
    return root / split / f"scenario_{int(scenario)}"


def _load_state(scenario: int, time_step: int) -> dict[str, np.ndarray]:
    path = _folder(FROZEN_DATASET, scenario) / f"t{int(time_step):03d}_state.npz"
    with np.load(path) as handle:
        return {key: np.asarray(handle[key]) for key in handle.files}


def _load_assignments(scenario: int, time_step: int) -> np.ndarray:
    path = _folder(REVISION_DATASET, scenario) / f"t{int(time_step):03d}_candidates.npz"
    with np.load(path) as handle:
        return np.asarray(handle["assignment"], dtype=np.int64)


def _manifest(scenario: int, time_step: int) -> dict:
    path = _folder(FROZEN_DATASET, scenario) / f"t{int(time_step):03d}_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _as_rows(pool: list[np.ndarray]) -> np.ndarray:
    if not pool:
        return np.zeros((0, 6), dtype=np.int64)
    return np.asarray([np.asarray(row, dtype=np.int64).reshape(-1) for row in pool], dtype=np.int64)


def scheme_c_order(pool: list[np.ndarray]) -> np.ndarray:
    """Phase-6I-B stores the unique assignments in lexicographic order."""
    keys = sorted({tuple(int(item) for item in np.asarray(row, dtype=np.int64).reshape(-1)) for row in pool})
    if not keys:
        return np.zeros((0, 6), dtype=np.int64)
    return np.asarray(keys, dtype=np.int64)


def formal_order(pool: list[np.ndarray]) -> np.ndarray:
    """Phase-6F keeps the depth-first order returned by build_pools."""
    return _as_rows(pool)


def _restore(scenario: int, time_step: int) -> tuple[L3BBitQueueEnvV2, dict[str, np.ndarray]]:
    """Copy a saved snapshot onto a fresh environment. The clock is not advanced."""
    saved = _load_state(scenario, time_step)
    env = L3BBitQueueEnvV2()
    env.uav_positions = np.asarray(saved["uav_position_m"], dtype=np.float64).copy()
    env.buoy_positions = np.asarray(saved["buoy_position_m"], dtype=np.float64).copy()
    env.q_bits = np.asarray(saved["q_bits"], dtype=np.float64).copy()
    env.queue = env.q_bits / float(env.block_bits)
    env.assignment = np.asarray(saved["association"], dtype=np.int64).copy()
    env.t = int(np.asarray(saved["time_index"]).reshape(-1)[0])
    return env, saved


def _additive_choice(rows: np.ndarray, normalized: np.ndarray) -> list[int]:
    if int(rows.shape[0]) == 0:
        return []
    valid = rows >= 0
    safe = np.where(valid, rows, 0)
    uav = np.arange(rows.shape[1], dtype=np.int64)[None, :]
    scores = np.where(valid, normalized[uav, safe], 0.0).sum(axis=1)
    return [int(item) for item in rows[int(np.argmax(scores))]]


def _contains(rows: np.ndarray, assignment: list[int]) -> bool:
    if not assignment or int(rows.shape[0]) == 0:
        return False
    target = np.asarray(assignment, dtype=np.int64)
    return bool(np.any(np.all(rows == target[None, :], axis=1)))


def _index_of(rows: np.ndarray, assignment: list[int]) -> int:
    if not assignment or int(rows.shape[0]) == 0:
        return -1
    target = np.asarray(assignment, dtype=np.int64)
    hits = np.flatnonzero(np.all(rows == target[None, :], axis=1))
    return int(hits[0]) if hits.size else -1


def _illegal_count(rows: np.ndarray) -> int:
    return int(sum(not is_legal(row, 6, 12) for row in rows))


def _idle_count(rows: np.ndarray) -> int:
    if int(rows.shape[0]) == 0:
        return 0
    return int(np.sum(np.any(rows < 0, axis=1)))


def _compare_rows(reference: np.ndarray, other: np.ndarray) -> dict:
    same_order = bool(reference.shape == other.shape and np.array_equal(reference, other))
    reference_keys = {tuple(int(item) for item in row) for row in reference}
    other_keys = {tuple(int(item) for item in row) for row in other}
    only_reference = reference_keys - other_keys
    only_other = other_keys - reference_keys
    prefix = min(int(reference.shape[0]), int(other.shape[0]))
    first_order_mismatch = -1
    if prefix:
        mismatch = np.flatnonzero(np.any(reference[:prefix] != other[:prefix], axis=1))
        if mismatch.size:
            first_order_mismatch = int(mismatch[0])
        elif int(reference.shape[0]) != int(other.shape[0]):
            first_order_mismatch = prefix
    example = list(next(iter(only_reference))) if only_reference else []
    return {
        "same_count": int(reference.shape[0] == other.shape[0]),
        "same_set": int(not only_reference and not only_other),
        "same_order": int(same_order),
        "only_reference": int(len(only_reference)),
        "only_other": int(len(only_other)),
        "first_order_mismatch": int(first_order_mismatch),
        "example_only_reference": example,
    }


def _scheme_choices() -> dict[tuple[int, int], list[int]]:
    path = Path(__file__).resolve().parents[4] / "phase6iw_inference_interface" / "summary.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {(int(row["scenario_id"]), int(row["time_step"])): [int(item) for item in row["scheme_c"]] for row in payload["rows"]}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _one(scenario: int, time_step: int, split: str, choices: dict[tuple[int, int], list[int]]) -> dict:
    assert_open(int(scenario), split)
    env, saved = _restore(int(scenario), int(time_step))
    if int(env.t) != int(time_step):
        raise RuntimeError("snapshot time was not restored")
    recomputed = np.asarray(feature_matrices(env)["rate"], dtype=np.float64)
    saved_rate = np.asarray(saved["uav_buoy_hypothetical_rate"], dtype=np.float64)
    rate_gap = float(np.max(np.abs(recomputed - saved_rate))) if saved_rate.size else 0.0
    formal_pool, _packed, formal_illegal = build_pools(env, ((GENERATOR, FORMAL_K),))
    frozen_pool, _packed_frozen, frozen_illegal = build_pools(env, ((GENERATOR, FROZEN_K),))
    formal_rows = formal_order(formal_pool[(GENERATOR, FORMAL_K)])
    frozen_rows = scheme_c_order(frozen_pool[(GENERATOR, FROZEN_K)])
    stored = _load_assignments(int(scenario), int(time_step))
    normalized = proposal_scores(feature_matrices(env))[GENERATOR]
    manifest = _manifest(int(scenario), int(time_step))
    recorded_additive = [int(item) for item in str(manifest["additive_assignment"]).split(",")]
    recorded_j15 = [int(item) for item in str(manifest["argmax_assignment"]).split(",")]
    scheme_choice = choices.get((int(scenario), int(time_step)), [])
    frozen_choice = _additive_choice(frozen_rows, normalized)
    formal_choice = _additive_choice(formal_rows, normalized)
    compared = _compare_rows(frozen_rows, formal_rows)
    compared["stored_matches_frozen_path"] = int(np.array_equal(stored, frozen_rows))
    compared["rate_gap"] = rate_gap
    compared["rate_matches_snapshot"] = int(rate_gap <= RATE_ATOL)
    compared["formal_illegal_counter"] = int(formal_illegal)
    compared["frozen_illegal_counter"] = int(frozen_illegal)
    compared["formal_illegal_rows"] = _illegal_count(formal_rows)
    compared["frozen_illegal_rows"] = _illegal_count(frozen_rows)
    compared["formal_idle_rows"] = _idle_count(formal_rows)
    compared["frozen_idle_rows"] = _idle_count(frozen_rows)
    compared["formal_count"] = int(formal_rows.shape[0])
    compared["frozen_count"] = int(frozen_rows.shape[0])
    compared["stored_count"] = int(stored.shape[0])
    compared["frozen_additive"] = frozen_choice
    compared["formal_additive"] = formal_choice
    compared["additive_same"] = int(frozen_choice == formal_choice)
    compared["recorded_additive_matches"] = int(frozen_choice == recorded_additive)
    compared["recorded_j15_in_formal"] = int(_contains(formal_rows, recorded_j15))
    compared["recorded_additive_in_formal"] = int(_contains(formal_rows, recorded_additive))
    compared["scheme_choice_in_formal"] = int(_contains(formal_rows, scheme_choice)) if scheme_choice else -1
    compared["scheme_choice_formal_index"] = _index_of(formal_rows, scheme_choice) if scheme_choice else -1
    compared["scheme_choice_frozen_index"] = _index_of(frozen_rows, scheme_choice) if scheme_choice else -1
    if compared["first_order_mismatch"] >= 0 and int(formal_rows.shape[0]) and int(frozen_rows.shape[0]):
        index = int(compared["first_order_mismatch"])
        compared["frozen_row_at_mismatch"] = [int(item) for item in frozen_rows[min(index, frozen_rows.shape[0] - 1)]]
        compared["formal_row_at_mismatch"] = [int(item) for item in formal_rows[min(index, formal_rows.shape[0] - 1)]]
    return {
        "scenario_id": int(scenario),
        "time_step": int(time_step),
        "split": split,
        **compared,
    }


def run_audit() -> dict:
    before = {
        "scheme_c": _sha256(SCHEME_C_CKPT),
        "epoch6": _sha256(EPOCH6_CKPT),
        "mappo": _sha256(MAPPO_CKPT),
    }
    split = verify_split()
    choices = _scheme_choices()
    rows = []
    for scenario, time_step in PROBE:
        rows.append(_one(int(scenario), int(time_step), "train", choices))
        print(f"probe {scenario} t={time_step} formal={rows[-1]['formal_count']} frozen={rows[-1]['frozen_count']}", flush=True)
    for scenario in VALIDATION_POOL:
        for time_step in PRIMARY_TIMES:
            rows.append(_one(int(scenario), int(time_step), "validation", choices))
            print(f"validation {scenario} t={time_step} same_order={rows[-1]['same_order']}", flush=True)
    comparable = all(int(row["rate_matches_snapshot"]) == 1 and int(row["stored_matches_frozen_path"]) == 1 for row in rows)
    identical = comparable and all(int(row["same_order"]) == 1 and int(row["same_set"]) == 1 for row in rows)
    after = {
        "scheme_c": _sha256(SCHEME_C_CKPT),
        "epoch6": _sha256(EPOCH6_CKPT),
        "mappo": _sha256(MAPPO_CKPT),
    }
    if before != after:
        comparable = False
    gate = assign_gate(comparable=comparable, identical=identical)
    validation = [row for row in rows if row["split"] == "validation"]
    result = {
        "gate": gate,
        "comparable": int(comparable),
        "identical": int(identical),
        "formal_generator": GENERATOR,
        "formal_k": FORMAL_K,
        "frozen_generator": GENERATOR,
        "frozen_k": FROZEN_K,
        "formal_order": "enumerate_legal depth-first, idle first at each UAV",
        "frozen_order": "unique assignments sorted lexicographically, as stored for scheme C",
        "j15_recomputed": 0,
        "environment_steps": 0,
        "hashes_unchanged": before == after,
        "checkpoint_hashes": before,
        "split": split,
        "states": int(len(rows)),
        "validation_states": int(len(validation)),
        "same_order_states": int(sum(int(row["same_order"]) for row in rows)),
        "same_set_states": int(sum(int(row["same_set"]) for row in rows)),
        "stored_match_states": int(sum(int(row["stored_matches_frozen_path"]) for row in rows)),
        "rate_match_states": int(sum(int(row["rate_matches_snapshot"]) for row in rows)),
        "scheme_choice_outside_formal": int(sum(int(row["scheme_choice_in_formal"]) == 0 for row in validation)),
        "recorded_j15_outside_formal": int(sum(int(row["recorded_j15_in_formal"]) == 0 for row in rows)),
        "median_only_frozen": float(np.median([row["only_reference"] for row in rows])),
        "max_rate_gap": float(max(row["rate_gap"] for row in rows)),
        "rss_bytes": _rss_bytes(),
        "reserved_scenarios_read": 0,
        "planner_modified": 0,
        "k_modified": 0,
        "planner_ready": False,
        "deployment_verified": False,
        "rows": rows,
    }
    folder = assert_output_path(OUTPUT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "comparable": comparable, "identical": identical}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    run_audit()
