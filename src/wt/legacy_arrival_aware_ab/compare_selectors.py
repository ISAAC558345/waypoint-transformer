"""Compare cached assignment scores with one fixed idle reference. No model forward."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_COMMIT = "d25e97fec756028a394a1cca67edb6d29c47c137"
INSTRUCTION_COMMIT = "964280298f8d45704c654c0aef41484ff3035366"
MODELS = ("E6", "A10", "B10")
TRAIN_SCENARIOS = tuple(range(322016, 322022))
EXPLORE_SCENARIOS = (322022, 322023)
TIMES = (15, 30)
SPLITS = ("train_t15", "train_t30", "explore_t15", "explore_t30")
EXPECTED_BLOBS = {
    "raw_matrices.npz": "6bfcfe7dc08f9c22a7d442ce4585586f7acb8b5e",
    "raw_scores.npz": "c56c25d17ba9359a1c0d85fb2b429666e715f6fd",
    "state_records.json": "feadba3e397234d46d1a6f8ac842dd6498e60ccd",
    "rows.csv": "ec048e6a3d6542bd201571f4e0c5b43ddadef768",
    "groups.json": "95f1158973c8eb079b5b588c8815cb0110c7da84",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(payload) + payload).hexdigest()


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _file_identity(path: Path, expected_blob: str) -> dict:
    workspace = path.read_bytes()
    workspace_blob = _git_blob_sha1(workspace)
    text_suffix = path.suffix.lower() in {".json", ".csv", ".md", ".py"}
    normalized = workspace.replace(b"\r\n", b"\n") if text_suffix else workspace
    normalized_blob = _git_blob_sha1(normalized)
    if workspace_blob == expected_blob:
        repository = workspace
        normalized_applied = False
    elif text_suffix and normalized_blob == expected_blob:
        repository = normalized
        normalized_applied = True
    else:
        raise RuntimeError(f"{path.name} git blob does not match {expected_blob}")
    return {
        "path": path.name,
        "git_blob_sha1": expected_blob,
        "repository_bytes_sha256": _sha256(repository),
        "workspace_bytes_sha256": _sha256(workspace),
        "workspace_git_blob_sha1": workspace_blob,
        "newline_normalization_applied": normalized_applied,
    }


def _active_mask(assignment: np.ndarray) -> str:
    return "".join("1" if int(value) >= 0 else "0" for value in assignment)


def _assignment_text(assignment: np.ndarray) -> str:
    return "|".join(str(int(value)) for value in assignment)


def _parse_assignment(text: str) -> np.ndarray:
    parts = str(text).split("|")
    if len(parts) != 6:
        raise RuntimeError(f"assignment does not describe six UAVs: {text}")
    values = [int(part) for part in parts]
    if any(value < -1 for value in values):
        raise RuntimeError(f"assignment has an unexpected idle marker: {text}")
    return np.asarray(values, dtype=np.int64)


def _choose_index(values: np.ndarray, candidate_ids: np.ndarray) -> int:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    identities = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    if scores.shape != identities.shape or int(scores.size) != 128:
        raise RuntimeError("scores are not aligned to the 128 local positions")
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("non-finite score")
    tied = np.flatnonzero(scores == np.max(scores))
    return int(tied[int(np.argmin(identities[tied]))])


def _choose_pool(values: np.ndarray, candidate_ids: np.ndarray, pool: np.ndarray) -> int:
    pool = np.asarray(pool, dtype=np.int64)
    if pool.size == 0:
        raise RuntimeError("candidate pool is empty")
    subset = np.asarray(values, dtype=np.float64)[pool]
    tied = pool[np.flatnonzero(subset == np.max(subset))]
    return int(tied[int(np.argmin(candidate_ids[tied]))])


def _used_edge_means(matrix: np.ndarray, assignments: np.ndarray) -> tuple[list[float], list[list[int]]]:
    means: list[float] = []
    used: list[list[int]] = []
    for uav in range(assignments.shape[1]):
        buoys = sorted({int(value) for value in assignments[:, uav].tolist() if int(value) >= 0})
        if any(buoy >= matrix.shape[1] for buoy in buoys):
            raise RuntimeError("assignment buoy is outside the cached matrix")
        used.append(buoys)
        if not buoys:
            means.append(0.0)
            continue
        values = np.asarray([matrix[uav, buoy] for buoy in buoys], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("non-finite used edge")
        means.append(float(np.mean(values)))
    return means, used


def _mask_offset(mask: str, mu: list[float]) -> float:
    total = 0.0
    for uav, bit in enumerate(mask):
        if bit == "1":
            total += float(mu[uav])
    return total


def _unique_used_edges(matrix: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    seen: set[tuple[int, int]] = set()
    values = []
    for assignment in assignments:
        for uav, buoy in enumerate(assignment.tolist()):
            pair = (int(uav), int(buoy))
            if pair[1] >= 0 and pair not in seen:
                seen.add(pair)
                values.append(float(matrix[pair]))
    return np.asarray(values, dtype=np.float64)


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise RuntimeError("edge distribution is empty")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "negative_fraction": float(np.mean(array < 0.0)),
    }


def select_rules(
    matrix: np.ndarray,
    scores: np.ndarray,
    assignments: np.ndarray,
    candidate_ids: np.ndarray,
) -> dict:
    """Choose with cached scores only. Payloads and label ranks are not arguments."""
    mu, used_buoys = _used_edge_means(matrix, assignments)
    masks = [_active_mask(row) for row in assignments]
    offset_table: dict[str, float] = {}
    offsets = np.zeros(assignments.shape[0], dtype=np.float64)
    for index, mask in enumerate(masks):
        if mask not in offset_table:
            offset_table[mask] = _mask_offset(mask, mu)
        offsets[index] = offset_table[mask]
    q_values = np.asarray(scores, dtype=np.float64) - offsets
    raw_local = _choose_index(scores, candidate_ids)
    idle_local = _choose_index(q_values, candidate_ids)
    return {
        "mu": mu,
        "used_buoys": used_buoys,
        "mask_offsets": offset_table,
        "offsets": offsets,
        "q_values": q_values,
        "RAW_SUM": raw_local,
        "ROW_MEAN_IDLE": idle_local,
    }


def _choice_record(
    local: int,
    scores: np.ndarray,
    q_values: np.ndarray,
    assignments: np.ndarray,
    candidate_ids: np.ndarray,
    decision_values: np.ndarray,
) -> dict:
    assignment = assignments[local]
    decision = np.asarray(decision_values, dtype=np.float64)
    return {
        "local_position": int(local),
        "candidate_id": int(candidate_ids[local]),
        "assignment": _assignment_text(assignment),
        "active_uavs": int(np.sum(assignment >= 0)),
        "active_mask": _active_mask(assignment),
        "raw_score": float(scores[local]),
        "q": float(q_values[local]),
        "tie_count": int(np.sum(decision == np.max(decision))),
    }


def _read_rows(path: Path) -> dict[tuple[int, int], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((int(row["scenario_id"]), int(row["time_step"])), []).append(row)
    expected = {(scenario, time_step) for scenario in TRAIN_SCENARIOS + EXPLORE_SCENARIOS for time_step in TIMES}
    if set(grouped) != expected or any(len(group) != 128 for group in grouped.values()):
        raise RuntimeError("paired rows are not the fixed 16 states of 128 candidates")
    return grouped


def _state_key(model: str, scenario: int, time_step: int) -> str:
    return f"{model}_{scenario}_t{time_step:03d}"


def _split_name(group: str, time_step: int) -> str:
    if group == "train":
        prefix = "train"
    elif group == "explore_eval":
        prefix = "explore"
    else:
        raise RuntimeError(f"unexpected candidate group {group}")
    return f"{prefix}_t{time_step}"


def _load_records(path: Path) -> dict[str, dict[tuple[int, int], dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    indexed = {}
    for model in MODELS:
        rows = payload[model]
        if len(rows) != 16:
            raise RuntimeError(f"{model} does not have 16 cached state records")
        indexed[model] = {(int(row["scenario_id"]), int(row["time_step"])): row for row in rows}
        if len(indexed[model]) != 16:
            raise RuntimeError(f"{model} state records repeat a scenario time")
    return indexed


def _best_payload(values: np.ndarray, candidate_ids: np.ndarray, pool: np.ndarray) -> int:
    return _choose_pool(values, candidate_ids, pool)


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _group_rows(states: list[dict], model: str, split: str | None = None) -> list[dict]:
    rows = [row for row in states if row["model"] == model and (split is None or row["split"] == split)]
    return rows


def _compact(row: dict) -> dict:
    raw = row["rules"]["RAW_SUM"]
    new = row["rules"]["ROW_MEAN_IDLE"]
    return {
        "scenario_id": row["scenario_id"],
        "time_step": row["time_step"],
        "split": row["split"],
        "raw_candidate_id": raw["candidate_id"],
        "new_candidate_id": new["candidate_id"],
        "raw_assignment": raw["assignment"],
        "new_assignment": new["assignment"],
        "raw_active_uavs": raw["active_uavs"],
        "new_active_uavs": new["active_uavs"],
        "raw_active_mask": raw["active_mask"],
        "new_active_mask": new["active_mask"],
        "raw_score": raw["raw_score"],
        "new_raw_score": new["raw_score"],
        "raw_q": raw["q"],
        "new_q": new["q"],
        "raw_real_payload_bits": raw["real_payload_bits"],
        "new_real_payload_bits": new["real_payload_bits"],
        "raw_real_regret_bits": raw["real_regret_bits"],
        "new_real_regret_bits": new["real_regret_bits"],
        "raw_zero_payload_bits": raw["zero_payload_bits"],
        "new_zero_payload_bits": new["zero_payload_bits"],
        "model": row["model"],
        "real_payload_difference_new_minus_old_bits": row["real_payload_difference_new_minus_old_bits"],
        "selection_changed": row["selection_changed"],
    }


def _split_summary(rows: list[dict]) -> dict:
    negatives = [row for row in rows if float(row["real_payload_difference_new_minus_old_bits"]) < 0.0]
    return {
        "states": len(rows),
        "selection_changed": int(sum(bool(row["selection_changed"]) for row in rows)),
        "mean_raw_active_uavs": _mean(rows, "raw_active_uavs"),
        "mean_new_active_uavs": _mean(rows, "new_active_uavs"),
        "mean_raw_real_payload_bits": _mean(rows, "raw_real_payload_bits"),
        "mean_new_real_payload_bits": _mean(rows, "new_real_payload_bits"),
        "mean_real_payload_difference_new_minus_old_bits": _mean(rows, "real_payload_difference_new_minus_old_bits"),
        "mean_raw_real_regret_bits": _mean(rows, "raw_real_regret_bits"),
        "mean_new_real_regret_bits": _mean(rows, "new_real_regret_bits"),
        "mean_raw_zero_payload_bits": _mean(rows, "raw_zero_payload_bits"),
        "mean_new_zero_payload_bits": _mean(rows, "new_zero_payload_bits"),
        "payload_improved": int(sum(float(row["real_payload_difference_new_minus_old_bits"]) > 0.0 for row in rows)),
        "payload_worsened": len(negatives),
        "payload_unchanged": int(sum(float(row["real_payload_difference_new_minus_old_bits"]) == 0.0 for row in rows)),
        "active_increased": int(sum(int(row["new_active_uavs"]) > int(row["raw_active_uavs"]) for row in rows)),
        "active_decreased": int(sum(int(row["new_active_uavs"]) < int(row["raw_active_uavs"]) for row in rows)),
        "active_unchanged": int(sum(int(row["new_active_uavs"]) == int(row["raw_active_uavs"]) for row in rows)),
        "active_increased_and_payload_not_improved": int(
            sum(
                int(row["new_active_uavs"]) > int(row["raw_active_uavs"])
                and float(row["real_payload_difference_new_minus_old_bits"]) <= 0.0
                for row in rows
            )
        ),
        "negative_real_payload_differences": negatives,
    }


def compare(root: Path, cache: Path, paired: Path, output: Path) -> dict:
    if output == cache or cache in output.parents:
        raise RuntimeError("refusing to write into the L4 cache")
    if (output / "manifest.json").exists() or (output / "failure.json").exists():
        raise RuntimeError("refusing to repeat an existing L5 output")
    output.mkdir(parents=True, exist_ok=True)
    consumption = {
        "environment_steps": 0,
        "low_level_inferences": 0,
        "scoring_forwards": 0,
        "training_updates": 0,
        "backward_passes": 0,
    }
    try:
        return _compare(root, cache, paired, output, consumption)
    except Exception as exc:
        if not (output / "failure.json").exists():
            _write_json(output / "failure.json", {"error": f"{type(exc).__name__}: {exc}", **consumption})
        raise


def _compare(root: Path, cache: Path, paired: Path, output: Path, consumption: dict) -> dict:
    rows_path = paired / "rows.csv" if paired.is_dir() else paired
    groups_path = rows_path.with_name("groups.json")
    identities = {
        "raw_matrices.npz": _file_identity(cache / "raw_matrices.npz", EXPECTED_BLOBS["raw_matrices.npz"]),
        "raw_scores.npz": _file_identity(cache / "raw_scores.npz", EXPECTED_BLOBS["raw_scores.npz"]),
        "state_records.json": _file_identity(cache / "state_records.json", EXPECTED_BLOBS["state_records.json"]),
        "rows.csv": _file_identity(rows_path, EXPECTED_BLOBS["rows.csv"]),
        "groups.json": _file_identity(groups_path, EXPECTED_BLOBS["groups.json"]),
    }
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    if groups["train_scenarios"] != list(TRAIN_SCENARIOS) or groups["explore_eval_scenarios"] != list(EXPLORE_SCENARIOS):
        raise RuntimeError("group index does not match the fixed scenario split")
    if groups["times"] != list(TIMES):
        raise RuntimeError("group index does not match times 15 and 30")
    records = _load_records(cache / "state_records.json")
    grouped_rows = _read_rows(rows_path)
    with np.load(cache / "raw_matrices.npz") as matrix_file, np.load(cache / "raw_scores.npz") as score_file:
        expected_keys = [
            _state_key(model, scenario, time_step)
            for model in MODELS
            for scenario in TRAIN_SCENARIOS + EXPLORE_SCENARIOS
            for time_step in TIMES
        ]
        if set(matrix_file.files) != set(expected_keys) or set(score_file.files) != set(expected_keys):
            raise RuntimeError("cache keys are not the 48 model-state pairs")
        prepared = []
        selections = []
        q_store = {}
        for model in MODELS:
            for scenario in TRAIN_SCENARIOS + EXPLORE_SCENARIOS:
                for time_step in TIMES:
                    key = _state_key(model, scenario, time_step)
                    group_rows = grouped_rows[(scenario, time_step)]
                    if any(row["group"] != group_rows[0]["group"] for row in group_rows):
                        raise RuntimeError(f"{key} mixes groups")
                    assignments = np.stack([_parse_assignment(row["assignment"]) for row in group_rows])
                    candidate_ids = np.asarray([int(row["candidate_id"]) for row in group_rows], dtype=np.int64)
                    if len(set(candidate_ids.tolist())) != 128:
                        raise RuntimeError(f"{key} repeats a candidate id")
                    matrix = np.array(matrix_file[key], dtype=np.float32, copy=True)
                    scores = np.array(score_file[key], dtype=np.float32, copy=True)
                    if matrix.shape != (6, 12) or matrix.dtype != np.float32 or scores.shape != (128,) or scores.dtype != np.float32:
                        raise RuntimeError(f"{key} cache arrays do not have the saved shapes")
                    if np.any(assignments < -1) or np.any(assignments >= matrix.shape[1]):
                        raise RuntimeError(f"{key} assignment is outside the cached matrix")
                    record = records[model][(scenario, time_step)]
                    local = int(record["local_position"])
                    if not 0 <= local < 128:
                        raise RuntimeError(f"{key} local position is outside 0..127")
                    if int(candidate_ids[local]) != int(record["candidate_id"]) or _assignment_text(assignments[local]) != record["assignment"]:
                        raise RuntimeError(f"{key} candidate identity is not aligned")
                    if float(scores[local]) != float(record["score"]):
                        raise RuntimeError(f"{key} cached score is not aligned")
                    chosen = select_rules(matrix, scores, assignments, candidate_ids)
                    _check_reference(key, matrix, assignments, candidate_ids, scores, chosen)
                    raw_choice = _choice_record(
                        chosen["RAW_SUM"], scores, chosen["q_values"], assignments, candidate_ids, scores
                    )
                    idle_choice = _choice_record(
                        chosen["ROW_MEAN_IDLE"], scores, chosen["q_values"], assignments, candidate_ids, chosen["q_values"]
                    )
                    selections.append(
                        {
                            "key": key,
                            "model": model,
                            "scenario_id": scenario,
                            "time_step": time_step,
                            "group": group_rows[0]["group"],
                            "split": _split_name(group_rows[0]["group"], time_step),
                            "mu": chosen["mu"],
                            "used_buoys": chosen["used_buoys"],
                            "mask_offsets": chosen["mask_offsets"],
                            "rules": {"RAW_SUM": raw_choice, "ROW_MEAN_IDLE": idle_choice},
                        }
                    )
                    prepared.append(
                        {
                            "key": key,
                            "matrix": matrix,
                            "scores": scores,
                            "assignments": assignments,
                            "candidate_ids": candidate_ids,
                            "rows": group_rows,
                            "record": record,
                            "selection": selections[-1],
                        }
                    )
                    q_store[f"{key}__q"] = chosen["q_values"]
                    q_store[f"{key}__offset"] = chosen["offsets"]
        code_sha = _sha256(Path(__file__).read_bytes())
        pre_label = {
            "package": "legacy_arrival_aware_ab_l5_pre_label",
            "source_results_commit": SOURCE_COMMIT,
            "instruction_commit": INSTRUCTION_COMMIT,
            "code_sha256": code_sha,
            "labels_used_for_selection": False,
            "rules": {
                "RAW_SUM": "maximum cached score; exact ties keep the smaller candidate_id",
                "ROW_MEAN_IDLE": "Q = float64(cached score) - sum of fixed per-UAV means over unique legal buoys used by the 128 candidates",
            },
            "mu_definition": "ascending buoy index, float64 mean; empty used-buoy set has mean 0; mask offset summed in UAV index order",
            "states": selections,
            **consumption,
        }
        _write_json(output / "pre_label_selections.json", pre_label)
        np.savez(output / "q_and_offsets.npz", **q_store)

        corrected_states = []
        evaluated = []
        for item in prepared:
            zero, real = _payloads(item["rows"])
            _check_labels(item, zero, real)
            selection = item["selection"]
            for rule in selection["rules"].values():
                local = int(rule["local_position"])
                rule["real_payload_bits"] = float(real[local])
                rule["real_regret_bits"] = float(np.max(real) - real[local])
                rule["zero_payload_bits"] = float(zero[local])
            raw = selection["rules"]["RAW_SUM"]
            new = selection["rules"]["ROW_MEAN_IDLE"]
            if int(raw["candidate_id"]) != int(item["record"]["candidate_id"]) or raw["assignment"] != item["record"]["assignment"]:
                raise RuntimeError(f"{item['key']} RAW_SUM candidate does not reproduce L4")
            if raw["real_payload_bits"] != float(item["record"]["real_payload_bits"]) or raw["zero_payload_bits"] != float(item["record"]["zero_payload_bits"]):
                raise RuntimeError(f"{item['key']} RAW_SUM payload does not reproduce L4")
            difference = float(new["real_payload_bits"] - raw["real_payload_bits"])
            evaluated.append({**selection, "selection_changed": raw["candidate_id"] != new["candidate_id"], "real_payload_difference_new_minus_old_bits": difference})
            corrected_states.append(_corrected_state(item, real, zero))
        corrected = _corrected_summary(corrected_states)
        compact_rows = [_compact(row) for row in evaluated]
        summary = {
            "primary_comparison": "B10 RAW_SUM versus ROW_MEAN_IDLE",
            "controls": ["E6", "A10"],
            "selection_change_is_not_a_payload_gain": True,
            "models": {},
        }
        for model in MODELS:
            model_rows = [row for row in compact_rows if row["model"] == model]
            summary["models"][model] = {split: _split_summary([row for row in model_rows if row["split"] == split]) for split in SPLITS}
            explore_rows = [row for row in model_rows if row["split"].startswith("explore")]
            summary["models"][model]["explore_states_combined_descriptive"] = _split_summary(explore_rows)
            summary["models"][model]["per_scenario"] = {
                str(scenario): {
                    "group": "train" if scenario in TRAIN_SCENARIOS else "explore_eval",
                    "t15": next(row for row in model_rows if row["scenario_id"] == scenario and row["time_step"] == 15),
                    "t30": next(row for row in model_rows if row["scenario_id"] == scenario and row["time_step"] == 30),
                }
                for scenario in TRAIN_SCENARIOS + EXPLORE_SCENARIOS
            }
        _write_json(output / "l4_corrected_summary.json", corrected)
        _write_json(output / "state_comparisons.json", {"states": evaluated})
        _write_json(output / "group_summary.json", summary)
        diagnose_sha = _sha256((root / "src/wt/legacy_arrival_aware_ab/diagnose.py").read_bytes())
        manifest = {
            "package": "legacy_arrival_aware_ab_l5",
            "source_results_commit": SOURCE_COMMIT,
            "instruction_commit": INSTRUCTION_COMMIT,
            "command": "py -3 -m wt.legacy_arrival_aware_ab.compare_selectors --run --root . --cache results/legacy_arrival_aware_ab_l4 --paired results/legacy_arrival_aware_ab_l2/paired --output results/legacy_arrival_aware_ab_l5",
            "code_sha256": code_sha,
            "diagnose_summary_code_sha256": diagnose_sha,
            "input_identity": identities,
            "checks": {
                "cache_keys": 48,
                "candidates_per_state": 128,
                "raw_sum_reproduced_l4": 48,
                "used_edges_per_state": 36,
                "auxiliary_matrix_entries_per_state": 72,
                "labels_joined_after_selection_file": True,
            },
            **consumption,
        }
        _write_json(output / "checks.json", manifest["checks"])
        _write_json(output / "manifest.json", manifest)
        print(json.dumps({"reproduction": "passed", **consumption}, sort_keys=True), flush=True)
        return manifest


def _payloads(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    zero = np.asarray([float(row["zero_payload_bits"]) for row in rows], dtype=np.float64)
    real = np.asarray([float(row["real_payload_bits"]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(zero)) or not np.all(np.isfinite(real)):
        raise RuntimeError("non-finite payload label")
    return zero, real


def _check_reference(key: str, matrix: np.ndarray, assignments: np.ndarray, candidate_ids: np.ndarray, scores: np.ndarray, chosen: dict) -> None:
    recomputed, used = _used_edge_means(matrix, assignments)
    if recomputed != chosen["mu"] or used != chosen["used_buoys"]:
        raise RuntimeError(f"{key} mu was not recomputed from the used buoys")
    if not all(np.isfinite(value) for value in recomputed):
        raise RuntimeError(f"{key} mu is non-finite")
    for uav, buoys in enumerate(used):
        observed = sorted({int(value) for value in assignments[:, uav].tolist() if int(value) >= 0})
        if observed != buoys:
            raise RuntimeError(f"{key} mu uses a buoy outside E_u")
        if not buoys and np.any(assignments[:, uav] >= 0):
            raise RuntimeError(f"{key} active UAV has an empty used-buoy set")
    delta = chosen["q_values"] - np.asarray(scores, dtype=np.float64)
    masks = [_active_mask(row) for row in assignments]
    for mask in set(masks):
        pool = np.asarray([index for index, value in enumerate(masks) if value == mask], dtype=np.int64)
        expected = -_mask_offset(mask, chosen["mu"])
        if not np.all(delta[pool] == expected):
            raise RuntimeError(f"{key} mask offset is not constant")
        if _choose_pool(scores, candidate_ids, pool) != _choose_pool(chosen["q_values"], candidate_ids, pool):
            raise RuntimeError(f"{key} same-mask ranking changed")
    for local in (chosen["RAW_SUM"], chosen["ROW_MEAN_IDLE"]):
        if not 0 <= int(local) < 128:
            raise RuntimeError(f"{key} selection is outside the 128 candidates")


def _check_labels(item: dict, zero: np.ndarray, real: np.ndarray) -> None:
    record = item["record"]
    local = int(record["local_position"])
    if float(real[local]) != float(record["real_payload_bits"]) or float(zero[local]) != float(record["zero_payload_bits"]):
        raise RuntimeError(f"{item['key']} label row is not aligned with L4")
    assignments = item["assignments"]
    candidate_ids = item["candidate_ids"]
    active = np.sum(assignments >= 0, axis=1)
    chosen = assignments[local]
    mask = _active_mask(chosen)
    same_mask = np.asarray([index for index, row in enumerate(assignments) if _active_mask(row) == mask], dtype=np.int64)
    same_count = np.flatnonzero(active == int(active[local]))
    if int(same_mask.size) != int(record["within_mask_candidates"]):
        raise RuntimeError(f"{item['key']} mask group size does not match L4")
    mask_regret = float(real[_best_payload(real, candidate_ids, same_mask)] - real[local])
    count_regret = float(real[_best_payload(real, candidate_ids, same_count)] - real[local])
    if mask_regret != float(record["within_mask_regret_bits"]) or count_regret != float(record["within_active_count_regret_bits"]):
        raise RuntimeError(f"{item['key']} within-group regret does not match L4")


def _corrected_state(item: dict, real: np.ndarray, zero: np.ndarray) -> dict:
    del zero
    record = item["record"]
    used = _distribution(_unique_used_edges(item["matrix"], item["assignments"]))
    stored = record["candidate_edges"]
    if int(used["count"]) != 36 or int(stored["count"]) != 36:
        raise RuntimeError(f"{item['key']} used-edge count is not 36")
    if used["mean"] != float(stored["mean"]) or used["negative_fraction"] != float(stored["negative_fraction"]):
        raise RuntimeError(f"{item['key']} recomputed used edges do not match the stored candidate edges")
    matrix = item["matrix"]
    if int(matrix.size) != 72:
        raise RuntimeError(f"{item['key']} full matrix does not have 72 entries")
    auxiliary_mean = float(np.mean(matrix))
    auxiliary_negative = float(np.mean(matrix < 0.0))
    if auxiliary_mean != float(record["matrix_mean"]) or auxiliary_negative != float(record["matrix_negative_fraction"]):
        raise RuntimeError(f"{item['key']} auxiliary matrix statistic does not match L4")
    active = np.sum(item["assignments"] >= 0, axis=1)
    local = int(record["local_position"])
    count_size = int(np.sum(active == int(active[local])))
    return {
        "model": item["selection"]["model"],
        "scenario_id": item["selection"]["scenario_id"],
        "time_step": item["selection"]["time_step"],
        "split": item["selection"]["split"],
        "used_candidate_edges": used,
        "auxiliary_full_matrix": {
            "count": 72,
            "mean": auxiliary_mean,
            "negative_fraction": auxiliary_negative,
            "definition": "every matrix entry, including edges the 128 candidates never use",
        },
        "within_mask_candidates": int(record["within_mask_candidates"]),
        "within_mask_regret_bits": float(record["within_mask_regret_bits"]),
        "within_mask_singleton": int(record["within_mask_candidates"]) == 1,
        "within_active_count_candidates": count_size,
        "within_active_count_regret_bits": float(record["within_active_count_regret_bits"]),
        "within_active_count_singleton": count_size == 1,
        "real_payload_checked_bits": float(real[local]),
    }


def _size_block(rows: list[dict], size_key: str, regret_key: str, singleton_key: str) -> dict:
    comparable = [row for row in rows if not row[singleton_key]]
    return {
        "group_sizes": [int(row[size_key]) for row in rows],
        "singleton_states": int(sum(bool(row[singleton_key]) for row in rows)),
        "comparable_states": len(comparable),
        "positive_regret_states": int(sum(float(row[regret_key]) > 0.0 for row in rows)),
        "positive_regret_among_comparable_states": int(sum(float(row[regret_key]) > 0.0 for row in comparable)),
        "singleton_zero_regret_is_not_ranking_evidence": True,
    }


def _corrected_summary(states: list[dict]) -> dict:
    summary = {
        "definition": "primary edge statistics use the 36 unique valid edges that appear in the 128 candidates; the 72-entry matrix is auxiliary",
        "stored_candidate_edges_match_recompute": True,
        "models": {},
    }
    for model in MODELS:
        summary["models"][model] = {}
        for split in SPLITS:
            rows = [row for row in states if row["model"] == model and row["split"] == split]
            summary["models"][model][split] = {
                "states": len(rows),
                "primary_used_candidate_edges": {
                    "mean_count": float(np.mean([row["used_candidate_edges"]["count"] for row in rows])),
                    "mean_of_state_means": float(np.mean([row["used_candidate_edges"]["mean"] for row in rows])),
                    "mean_negative_fraction": float(np.mean([row["used_candidate_edges"]["negative_fraction"] for row in rows])),
                },
                "auxiliary_full_matrix": {
                    "mean_count": float(np.mean([row["auxiliary_full_matrix"]["count"] for row in rows])),
                    "mean_of_state_means": float(np.mean([row["auxiliary_full_matrix"]["mean"] for row in rows])),
                    "mean_negative_fraction": float(np.mean([row["auxiliary_full_matrix"]["negative_fraction"] for row in rows])),
                },
                "within_mask": _size_block(rows, "within_mask_candidates", "within_mask_regret_bits", "within_mask_singleton"),
                "within_active_count": _size_block(
                    rows, "within_active_count_candidates", "within_active_count_regret_bits", "within_active_count_singleton"
                ),
            }
    summary["states"] = states
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=Path("results/legacy_arrival_aware_ab_l4"))
    parser.add_argument("--paired", type=Path, default=Path("results/legacy_arrival_aware_ab_l2/paired"))
    parser.add_argument("--output", type=Path, default=Path("results/legacy_arrival_aware_ab_l5"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve() if args.root is not None else _repo_root()
    if not args.run:
        print(
            json.dumps(
                {
                    "mode": "config_only",
                    "scoring_forwards": 0,
                    "training_updates": 0,
                    "rules": ["RAW_SUM", "ROW_MEAN_IDLE"],
                },
                indent=2,
            ),
            flush=True,
        )
        return
    compare(
        root,
        _resolve(root, args.cache),
        _resolve(root, args.paired),
        _resolve(root, args.output),
    )


if __name__ == "__main__":
    main()
