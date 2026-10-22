"""Row-by-row check of the isolated k=6 adapter. No planner call and no rollout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import FROZEN_DATASET, REVISION_DATASET
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import PRIMARY_TIMES
from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import load_decision
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import assert_open
from wt.benchmark_v2.phase6is_assignment_head.audit import _rss_bytes
from wt.benchmark_v2.phase6iu_pairwise_head.audit import verify_split
from wt.benchmark_v2.phase6iw_inference_adapter.interface import SchemeCAdapter
from wt.benchmark_v2.phase6iw_inference_adapter.protocol import MAPPO_CKPT
from wt.benchmark_v2.phase6iy_candidate_adapter.adapter import g1_choice, generate_frozen_pool, snapshot_rate_gap
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import (
    CANDIDATE_COUNT,
    EPOCH6_CKPT,
    FORMAL_ENTRY_K,
    OUTPUT,
    PROBE,
    RATE_ATOL,
    SCHEME_C_CKPT,
    VALIDATION_POOL,
    assert_output_path,
    assign_gate,
    config_payload,
    config_sha256,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assignments(root: Path, scenario: int, time_step: int) -> np.ndarray:
    split = "train" if int(scenario) < 322048 else "validation"
    path = root / split / f"scenario_{int(scenario)}" / f"t{int(time_step):03d}_candidates.npz"
    with np.load(path) as handle:
        return np.asarray(handle["assignment"], dtype=np.int64)


def _manifest(scenario: int, time_step: int) -> dict:
    split = "train" if int(scenario) < 322048 else "validation"
    path = FROZEN_DATASET / split / f"scenario_{int(scenario)}" / f"t{int(time_step):03d}_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _published_choices() -> dict[tuple[int, int], list[int]]:
    path = Path(__file__).resolve().parents[4] / "phase6iw_inference_interface" / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {(int(row["scenario_id"]), int(row["time_step"])): [int(item) for item in row["scheme_c"]] for row in payload["rows"]}


def _set_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left_keys = {tuple(int(item) for item in row) for row in left}
    right_keys = {tuple(int(item) for item in row) for row in right}
    return left_keys == right_keys


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


def _one(scenario: int, time_step: int, split: str, scorer: SchemeCAdapter, published: dict[tuple[int, int], list[int]]) -> dict:
    assert_open(int(scenario), split)
    rate_gap = snapshot_rate_gap(int(scenario), int(time_step))
    first = generate_frozen_pool(int(scenario), int(time_step))
    second = generate_frozen_pool(int(scenario), int(time_step))
    matrix = first.assignment_matrix()
    repeated = second.assignment_matrix()
    stored = _assignments(REVISION_DATASET, int(scenario), int(time_step))
    frozen_file = _assignments(FROZEN_DATASET, int(scenario), int(time_step))
    manifest = _manifest(int(scenario), int(time_step))
    sample = load_decision(int(scenario), int(time_step))
    loaded = np.asarray(sample["assignments"], dtype=np.int64)
    result = scorer.infer(sample["graph"], matrix)
    bound = list(first.assignment_at(result.chosen_index))
    recorded = [int(item) for item in str(manifest["additive_assignment"]).split(",")]
    chosen_g1 = [int(item) for item in g1_choice(int(scenario), int(time_step), first)]
    published_choice = published.get((int(scenario), int(time_step)))
    order_mismatch = -1
    if matrix.shape == stored.shape:
        mismatch = np.flatnonzero(np.any(matrix != stored, axis=1))
        if mismatch.size:
            order_mismatch = int(mismatch[0])
    elif int(matrix.shape[0]) or int(stored.shape[0]):
        order_mismatch = 0
    row = {
        "scenario_id": int(scenario),
        "time_step": int(time_step),
        "split": split,
        "state_hash": first.state_hash,
        "manifest_state_hash": str(manifest["state_hash"]),
        "state_hash_match": int(first.state_hash == str(manifest["state_hash"])),
        "count": int(matrix.shape[0]),
        "raw_count": int(first.raw_count),
        "set_equal": int(_set_equal(matrix, stored) and _set_equal(matrix, frozen_file)),
        "order_equal": int(np.array_equal(matrix, stored) and np.array_equal(matrix, frozen_file) and np.array_equal(matrix, loaded)),
        "repeat_equal": int(np.array_equal(matrix, repeated) and first.state_hash == second.state_hash),
        "index_bound": int(all(first.assignment_at(binding.score_index) == binding.assignment for binding in first.bindings)),
        "legal_rows": int(sum(binding.legal for binding in first.bindings)),
        "idle_rows": int(sum(binding.contains_idle for binding in first.bindings)),
        "illegal_rows": int(sum(not binding.legal for binding in first.bindings)),
        "rate_gap": float(rate_gap),
        "rate_match": int(rate_gap <= RATE_ATOL),
        "scheme_bound": int(bound == result.chosen_assignment and list(matrix[result.chosen_index]) == result.chosen_assignment),
        "scheme_index": int(result.chosen_index),
        "scheme_assignment": result.chosen_assignment,
        "published_scheme_match": int(published_choice is None or published_choice == result.chosen_assignment),
        "additive_match": int(chosen_g1 == recorded),
        "first_order_mismatch": int(order_mismatch),
    }
    if order_mismatch >= 0:
        row["adapter_assignment"] = [int(item) for item in matrix[min(order_mismatch, matrix.shape[0] - 1)]]
        row["stored_assignment"] = [int(item) for item in stored[min(order_mismatch, stored.shape[0] - 1)]]
    return row


def _failed(row: dict) -> bool:
    required = (
        "set_equal",
        "order_equal",
        "repeat_equal",
        "index_bound",
        "rate_match",
        "scheme_bound",
        "published_scheme_match",
        "additive_match",
        "state_hash_match",
    )
    if any(int(row[name]) != 1 for name in required):
        return True
    return int(row["count"]) != CANDIDATE_COUNT or int(row["illegal_rows"]) != 0 or int(row["legal_rows"]) != CANDIDATE_COUNT


def run_audit() -> dict:
    before = {
        "scheme_c": _sha256(SCHEME_C_CKPT),
        "epoch6": _sha256(EPOCH6_CKPT),
        "mappo": _sha256(MAPPO_CKPT),
        "config": config_sha256(),
    }
    split = verify_split()
    scorer = SchemeCAdapter()
    scorer.checkpoint_sha256 = before["scheme_c"]
    published = _published_choices()
    rows = []
    for scenario, time_step in PROBE:
        rows.append(_one(int(scenario), int(time_step), "train", scorer, published))
        print(f"probe {scenario} t={time_step} order={rows[-1]['order_equal']} set={rows[-1]['set_equal']}", flush=True)
    for scenario in VALIDATION_POOL:
        for time_step in PRIMARY_TIMES:
            rows.append(_one(int(scenario), int(time_step), "validation", scorer, published))
            print(f"validation {scenario} t={time_step} order={rows[-1]['order_equal']}", flush=True)
    comparable = all(int(row["rate_match"]) == 1 and int(row["state_hash_match"]) == 1 for row in rows) and split == {"train": 48, "validation": 16}
    identical = comparable and not any(_failed(row) for row in rows)
    after = {
        "scheme_c": _sha256(SCHEME_C_CKPT),
        "epoch6": _sha256(EPOCH6_CKPT),
        "mappo": _sha256(MAPPO_CKPT),
        "config": config_sha256(),
    }
    if before != after:
        comparable = False
        identical = False
    gate = assign_gate(comparable=comparable, identical=identical)
    first_failure = next((row for row in rows if _failed(row)), None)
    result = {
        "gate": gate,
        "comparable": int(comparable),
        "identical": int(identical),
        "config": config_payload(),
        "config_sha256": before["config"],
        "checkpoint_hashes": {key: value for key, value in before.items() if key != "config"},
        "hashes_unchanged": before == after,
        "formal_planner_k_unchanged": FORMAL_ENTRY_K,
        "states": int(len(rows)),
        "passed_states": int(sum(not _failed(row) for row in rows)),
        "set_pass": int(sum(int(row["set_equal"]) for row in rows)),
        "order_pass": int(sum(int(row["order_equal"]) for row in rows)),
        "index_pass": int(sum(int(row["index_bound"]) for row in rows)),
        "repeat_pass": int(sum(int(row["repeat_equal"]) for row in rows)),
        "scheme_pass": int(sum(int(row["scheme_bound"]) for row in rows)),
        "published_scheme_pass": int(sum(int(row["published_scheme_match"]) for row in rows)),
        "additive_pass": int(sum(int(row["additive_match"]) for row in rows)),
        "first_failure": first_failure,
        "call_chain": [
            "phase6iy_candidate_adapter.adapter.generate_frozen_pool",
            "phase6ix_candidate_alignment.audit._restore",
            "phase6e.generators.build_pools",
            "phase6ib_highlevel_dataset.jobs._key",
            "sorted unique keys from jobs._score_current",
        ],
        "environment_steps": 0,
        "j15_recomputed": 0,
        "planner_modified": 0,
        "rss_bytes": _rss_bytes(),
        "reserved_scenarios_read": 0,
        "planner_ready": False,
        "deployment_verified": False,
        "rows": rows,
    }
    folder = assert_output_path(OUTPUT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "passed": result["passed_states"], "states": result["states"]}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    run_audit()
